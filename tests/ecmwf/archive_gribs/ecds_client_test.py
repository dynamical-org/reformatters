import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import requests

from reformatters.ecmwf.archive_gribs import ecds_client
from reformatters.ecmwf.archive_gribs.ecds_client import (
    EcdsRequest,
    RequestState,
    StateStore,
    constraints,
    costing,
    process_url,
)

from .grib_inventory_test import grib_message


@pytest.fixture(autouse=True)
def _isolated_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CDSAPI_RC", str(tmp_path / "absent.cdsapirc"))
    monkeypatch.delenv("ECDS_API_ENDPOINT", raising=False)
    monkeypatch.delenv("ECDS_API_KEY", raising=False)


def response(body: dict[str, Any], status_code: int = 200) -> Mock:
    result = Mock()
    result.json.return_value = body
    result.status_code = status_code
    result.headers = {}
    result.raise_for_status.return_value = None
    return result


def session_mock() -> Mock:
    session = Mock()
    session.headers = {"PRIVATE-TOKEN": "test-token"}
    return session


class FakeClock:
    """Stands in for `time`, so what polling sleeps can be measured without waiting."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake_clock = FakeClock()
    monkeypatch.setattr(ecds_client, "time", fake_clock)
    return fake_clock


def test_process_url_defaults_to_the_ecds_s2s_forecasts_process() -> None:
    assert (
        process_url()
        == "https://ecds.ecmwf.int/api/retrieve/v1/processes/s2s-forecasts"
    )


def test_endpoint_and_token_come_from_cdsapirc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / ".cdsapirc"
    config_path.write_text(
        "url: https://example.test/api\nkey: personal-access-token\n"
    )
    monkeypatch.setenv("CDSAPI_RC", str(config_path))

    request = EcdsRequest(StateStore(tmp_path / "state.json"))

    assert request.execution_url == (
        "https://example.test/api/retrieve/v1/processes/s2s-forecasts/execution"
    )
    assert request.session.headers["PRIVATE-TOKEN"] == "personal-access-token"
    assert "Authorization" not in request.session.headers


def test_environment_overrides_cdsapirc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / ".cdsapirc"
    config_path.write_text("url: https://from-file.test/api\nkey: file-token\n")
    monkeypatch.setenv("CDSAPI_RC", str(config_path))
    monkeypatch.setenv("ECDS_API_ENDPOINT", "https://from-env.test/api/")
    monkeypatch.setenv("ECDS_API_KEY", "env-token")

    request = EcdsRequest(StateStore(tmp_path / "state.json"))

    assert request.execution_url.startswith("https://from-env.test/api/")
    assert request.session.headers["PRIVATE-TOKEN"] == "env-token"


def test_submitting_without_credentials_raises(tmp_path: Path) -> None:
    request = EcdsRequest(StateStore(tmp_path / "state.json"), session=Mock(headers={}))

    with pytest.raises(AssertionError, match="ECDS_API_KEY"):
        request.submit({"variable": ["total_precipitation"]})


def test_state_is_written_atomically(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")

    state_store.write(RequestState("id", {"day": ["24"]}, "now", "status"))

    assert json.loads(state_store.path.read_text())["request_id"] == "id"
    assert not state_store.path.with_suffix(".json.tmp").exists()
    assert state_store.read().payload == {"day": ["24"]}


def test_a_submitted_request_can_be_polled_by_another_process(tmp_path: Path) -> None:
    submit_response = response({"jobID": "job-1"})
    submit_response.headers = {"Location": "https://example.test/jobs/job-1"}
    submitting_session = session_mock()
    submitting_session.post.return_value = submit_response
    state_store = StateStore(tmp_path / "state.json")
    EcdsRequest(state_store, session=submitting_session).submit({"variable": ["tp"]})

    resumed_session = session_mock()
    resumed_session.get.side_effect = [
        response(
            {"status": "successful", "links": [{"rel": "results", "href": "results"}]}
        ),
        response({"asset": {"value": {"href": "https://example.test/blob"}}}),
    ]

    state, result_url = EcdsRequest(state_store, session=resumed_session).poll_once()

    assert state.request_id == "job-1"
    assert result_url == "https://example.test/blob"
    assert state_store.read().result_url == "https://example.test/blob"
    assert resumed_session.get.call_args_list[0].args == (
        "https://example.test/jobs/job-1",
    )
    assert resumed_session.get.call_args_list[1].args == ("results",)


def test_a_failed_poll_records_the_results_error(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("id", {}, "now", "status"))
    failure: dict[str, Any] = {"status": 400, "title": "The job has failed"}
    session = session_mock()
    session.get.side_effect = [
        response(
            {"status": "failed", "links": [{"rel": "results", "href": "results"}]}
        ),
        response(failure, status_code=400),
    ]

    state, result_url = EcdsRequest(state_store, session=session).poll_once()

    assert result_url is None
    assert state.errors == [json.dumps(failure, sort_keys=True)]


def test_polling_raises_on_a_terminal_failure(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("id", {}, "now", "status"))
    session = session_mock()
    session.get.return_value = response({"status": "dismissed"})

    with pytest.raises(RuntimeError, match="ended with status dismissed"):
        EcdsRequest(state_store, session=session).poll_until_complete(0, 3)


def test_polling_raises_when_the_job_never_completes(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("id", {}, "now", "status"))
    session = session_mock()
    session.get.return_value = response({"status": "running"})

    with pytest.raises(TimeoutError, match="did not complete within 2 polls"):
        EcdsRequest(state_store, session=session).poll_until_complete(0, 2)


def test_polling_waits_for_success_even_once_the_body_carries_a_url(
    tmp_path: Path, clock: FakeClock
) -> None:
    """A queued job's status document can carry hrefs of its own; only success returns one."""
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("id", {}, "now", "status"))
    session = session_mock()
    session.get.return_value = response(
        {"status": "running", "asset": {"value": {"href": "https://example.test/blob"}}}
    )

    with pytest.raises(TimeoutError):
        EcdsRequest(state_store, session=session).poll_until_complete(30, 3)


def test_polling_rejects_a_status_response_without_a_status(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("id", {}, "now", "status"))
    session = session_mock()
    session.get.return_value = response({"jobID": "job-1"})

    with pytest.raises(AssertionError, match="has no status"):
        EcdsRequest(state_store, session=session).poll_until_complete(30, 3)


def test_poll_backoff_follows_consecutive_failures_not_the_poll_count(
    tmp_path: Path, clock: FakeClock
) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("id", {}, "now", "status"))
    session = session_mock()
    running = response({"status": "running"})
    session.get.side_effect = [
        running,
        running,
        running,
        requests.ConnectionError("transient"),
        requests.ConnectionError("transient"),
        response({"status": "successful", "asset": {"value": {"href": "blob"}}}),
    ]

    EcdsRequest(state_store, session=session).poll_until_complete(30, 240)

    assert clock.sleeps == [30, 30, 30, 30, 60]
    assert state_store.read().poll_failures == 0


def test_a_persistently_failing_status_url_times_out_within_its_poll_budget(
    tmp_path: Path, clock: FakeClock
) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("id", {}, "now", "status"))
    session = session_mock()
    session.get.side_effect = requests.ConnectionError("down")

    with pytest.raises(TimeoutError):
        EcdsRequest(state_store, session=session).poll_until_complete(30, 240)

    assert clock.now <= 30 * 240
    assert state_store.read().poll_failures > 1


def test_download_uses_the_result_url_saved_by_poll(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(
        RequestState("id", {}, "now", "status", result_url="https://example.test/blob")
    )
    message = grib_message()
    download_response = response({})
    download_response.iter_content.return_value = [message]
    session = session_mock()
    session.get.return_value = download_response

    state = EcdsRequest(state_store, session=session).download(tmp_path / "blob.grib2")

    assert (tmp_path / "blob.grib2").read_bytes() == message
    assert state.downloaded_bytes == len(message)
    assert state.grib_messages == 1
    assert state.status == "downloaded"


def test_download_resumes_a_partial_file_after_http_206(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("id", {}, "now", "status", result_url="result"))
    target = tmp_path / "blob.grib2"
    message = grib_message()
    target.with_suffix(".grib2.partial").write_bytes(message[:4])
    download_response = response({}, status_code=requests.codes.partial)
    download_response.iter_content.return_value = [message[4:]]
    session = session_mock()
    session.get.return_value = download_response

    EcdsRequest(state_store, session=session).download(target)

    assert target.read_bytes() == message
    assert session.get.call_args.kwargs["headers"] == {"Range": "bytes=4-"}


def test_download_restarts_a_partial_file_after_http_200(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("id", {}, "now", "status", result_url="result"))
    target = tmp_path / "blob.grib2"
    message = grib_message()
    target.with_suffix(".grib2.partial").write_bytes(b"stale bytes")
    download_response = response({})
    download_response.iter_content.return_value = [message]
    session = session_mock()
    session.get.return_value = download_response

    EcdsRequest(state_store, session=session).download(target)

    assert target.read_bytes() == message
    assert session.get.call_args.kwargs["headers"] == {"Range": "bytes=11-"}


def test_download_rejects_a_truncated_blob(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("id", {}, "now", "status", result_url="result"))
    download_response = response({})
    download_response.iter_content.return_value = [grib_message()[:-4]]
    session = session_mock()
    session.get.return_value = download_response

    with pytest.raises(AssertionError, match="Truncated message"):
        EcdsRequest(state_store, session=session).download(tmp_path / "blob.grib2")

    assert not (tmp_path / "blob.grib2").exists()


def test_retrieve_resumes_an_in_flight_request_without_resubmitting(
    tmp_path: Path,
) -> None:
    payload = {"variable": ["total_precipitation"]}
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("job-1", payload, "now", "status"))
    message = grib_message()
    download_response = response({})
    download_response.iter_content.return_value = [message]
    session = session_mock()
    session.get.side_effect = [
        response({"status": "successful", "asset": {"value": {"href": "blob"}}}),
        download_response,
    ]

    EcdsRequest(state_store, session=session).retrieve(
        payload, tmp_path / "blob.grib2", poll_seconds=0
    )

    session.post.assert_not_called()
    assert (tmp_path / "blob.grib2").read_bytes() == message


def test_retrieve_keeps_a_blob_it_has_already_downloaded(tmp_path: Path) -> None:
    """An ECDS result expires without an SLA, so a downloaded blob must not be discarded."""
    payload = {"variable": ["total_precipitation"]}
    target = tmp_path / "blob.grib2"
    message = grib_message()
    target.write_bytes(message)
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(
        RequestState(
            "job-1",
            payload,
            "now",
            "status",
            status="downloaded",
            downloaded_bytes=len(message),
            grib_messages=1,
        )
    )
    session = session_mock()

    EcdsRequest(state_store, session=session).retrieve(payload, target, poll_seconds=0)

    session.post.assert_not_called()
    session.get.assert_not_called()
    assert target.read_bytes() == message


def test_retrieve_downloads_again_when_the_staged_blob_does_not_match(
    tmp_path: Path,
) -> None:
    payload = {"variable": ["total_precipitation"]}
    target = tmp_path / "blob.grib2"
    message = grib_message()
    target.write_bytes(message)
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(
        RequestState(
            "job-1",
            payload,
            "now",
            "status",
            status="downloaded",
            downloaded_bytes=len(message),
            grib_messages=2,
        )
    )
    download_response = response({})
    download_response.iter_content.return_value = [message, message]
    session = session_mock()
    session.get.side_effect = [
        response({"status": "successful", "asset": {"value": {"href": "blob"}}}),
        download_response,
    ]

    EcdsRequest(state_store, session=session).retrieve(payload, target, poll_seconds=0)

    session.post.assert_not_called()
    assert state_store.read().grib_messages == 2


def test_retrieve_resubmits_after_a_terminal_failure(tmp_path: Path) -> None:
    payload = {"variable": ["total_precipitation"]}
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("job-1", payload, "now", "status", status="failed"))
    submit_response = response({"jobID": "job-2"})
    session = session_mock()
    session.post.return_value = submit_response
    download_response = response({})
    download_response.iter_content.return_value = [grib_message()]
    session.get.side_effect = [
        response({"status": "successful", "asset": {"value": {"href": "blob"}}}),
        download_response,
    ]

    EcdsRequest(state_store, session=session).retrieve(
        payload, tmp_path / "blob.grib2", poll_seconds=0
    )

    session.post.assert_called_once()
    assert state_store.read().request_id == "job-2"


def test_constraints_retries_a_transient_server_error() -> None:
    """ECDS intermittently 502s these endpoints; one blip must not abort a staging run."""
    session = Mock()
    failure = Mock()
    failure.raise_for_status.side_effect = requests.HTTPError("502 Bad Gateway")
    session.post.side_effect = [failure, response({"variable": ["surface_pressure"]})]

    assert constraints({"origin": "ecmwf"}, session=session) == {
        "variable": ["surface_pressure"]
    }
    assert session.post.call_count == 2


def test_constraints_raises_once_retries_are_exhausted() -> None:
    session = Mock()
    session.post.return_value.raise_for_status.side_effect = requests.HTTPError("502")

    with pytest.raises(requests.HTTPError):
        constraints({"origin": "ecmwf"}, session=session)


def test_constraints_and_costing_post_to_the_unauthenticated_endpoints() -> None:
    session = Mock()
    session.post.return_value = response({"variable": ["total_precipitation"]})

    assert constraints({"origin": "ecmwf"}, session=session) == {
        "variable": ["total_precipitation"]
    }
    assert session.post.call_args.args[0].endswith(
        "/retrieve/v1/processes/s2s-forecasts/constraints"
    )
    assert session.post.call_args.kwargs["json"] == {"inputs": {"origin": "ecmwf"}}

    session.post.return_value = response({"id": "size", "cost": 202.0, "limit": 1e6})

    assert costing({"origin": "ecmwf"}, session=session) == (202.0, 1e6)
    assert session.post.call_args.args[0].endswith(
        "/retrieve/v1/processes/s2s-forecasts/costing"
    )
