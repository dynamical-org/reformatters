import itertools
import json
import threading
import time
from collections.abc import Callable, Iterator
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import Mock, call

import pytest
import requests
from requests.adapters import BaseAdapter

from reformatters.ecmwf.archive_gribs import ecds_client
from reformatters.ecmwf.archive_gribs.ecds_client import (
    EcdsJobFailedError,
    EcdsRangeMismatchError,
    EcdsRequest,
    RequestState,
    StateStore,
    constraints,
    costing,
    process_url,
)

from .grib_inventory_test import grib_message

SUBMITTED_AT = "2026-10-02T06:00:00+00:00"


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
    result.url = None
    result.raise_for_status.return_value = None
    return result


def session_mock() -> Mock:
    session = Mock()
    session.headers = {"PRIVATE-TOKEN": "marker-a"}
    return session


class OfflineAdapter(BaseAdapter):
    def __init__(self) -> None:
        self.responses: list[requests.Response] = []
        self.sent: list[requests.PreparedRequest] = []

    def send(
        self, request: requests.PreparedRequest, *_args: object, **_kwargs: object
    ) -> requests.Response:
        self.sent.append(request.copy())
        assert self.responses, "Unexpected offline request"
        result = self.responses.pop(0)
        assert request.url is not None
        result.url = request.url
        result.request = request
        return result

    def close(self) -> None:
        pass


def wire_response(
    body: dict[str, Any] | bytes = b"",
    status_code: int = 200,
    location: str | None = None,
) -> requests.Response:
    result = requests.Response()
    result.status_code = status_code
    result.raw = BytesIO(json.dumps(body).encode() if isinstance(body, dict) else body)
    if location is not None:
        result.headers["Location"] = location
    return result


@pytest.fixture
def offline_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[EcdsRequest, OfflineAdapter, OfflineAdapter]:
    monkeypatch.setenv("ECDS_API_KEY", "marker-a")
    monkeypatch.setattr(
        requests.sessions,
        "get_netrc_auth",
        Mock(return_value=None),
    )
    request = EcdsRequest(
        StateStore(tmp_path / "state.json"), api_url="https://api.test/api"
    )
    request.session.trust_env = False
    api_adapter = OfflineAdapter()
    download_adapter = OfflineAdapter()
    for scheme in ("https://", "http://"):
        request.session.mount(scheme, api_adapter)
        request.download_session.mount(scheme, download_adapter)
    return request, api_adapter, download_adapter


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
    config_path.write_text("url: https://example.test/api\nkey: marker-b\n")
    monkeypatch.setenv("CDSAPI_RC", str(config_path))

    request = EcdsRequest(StateStore(tmp_path / "state.json"))

    assert request.execution_url == (
        "https://example.test/api/retrieve/v1/processes/s2s-forecasts/execution"
    )
    assert request.session.headers["PRIVATE-TOKEN"] == "marker-b"
    assert "Authorization" not in request.session.headers


def test_environment_overrides_cdsapirc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / ".cdsapirc"
    config_path.write_text("url: https://from-file.test/api\nkey: marker-c\n")
    monkeypatch.setenv("CDSAPI_RC", str(config_path))
    monkeypatch.setenv("ECDS_API_ENDPOINT", "https://from-env.test/api/")
    monkeypatch.setenv("ECDS_API_KEY", "marker-d")

    request = EcdsRequest(StateStore(tmp_path / "state.json"))

    assert request.execution_url.startswith("https://from-env.test/api/")
    assert request.session.headers["PRIVATE-TOKEN"] == "marker-d"


def test_submitting_without_credentials_raises(tmp_path: Path) -> None:
    request = EcdsRequest(StateStore(tmp_path / "state.json"), session=Mock(headers={}))

    with pytest.raises(AssertionError, match="ECDS_API_KEY"):
        request.submit({"variable": ["total_precipitation"]})


def test_state_is_written_atomically(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")

    state_store.write(RequestState("id", {"day": ["24"]}, SUBMITTED_AT, "status"))

    assert json.loads(state_store.path.read_text())["request_id"] == "id"
    assert not state_store.path.with_suffix(".json.tmp").exists()
    assert state_store.read().payload == {"day": ["24"]}


def test_a_submitted_request_can_be_polled_by_another_process(tmp_path: Path) -> None:
    submit_response = response({"jobID": "job-1"})
    submit_response.headers = {"Location": "https://ecds.ecmwf.int/jobs/job-1"}
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
        "https://ecds.ecmwf.int/jobs/job-1",
    )
    assert resumed_session.get.call_args_list[1].args == (
        "https://ecds.ecmwf.int/jobs/results",
    )


def test_a_failed_poll_records_the_results_error(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("id", {}, SUBMITTED_AT, "status"))
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
    state_store.write(RequestState("id", {}, SUBMITTED_AT, "status"))
    session = session_mock()
    session.get.return_value = response({"status": "dismissed"})

    with pytest.raises(EcdsJobFailedError, match="ended with status dismissed"):
        EcdsRequest(state_store, session=session).poll_until_complete(0, 3)


def test_polling_raises_when_the_job_never_completes(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("id", {}, SUBMITTED_AT, "status"))
    session = session_mock()
    session.get.return_value = response({"status": "running"})

    with pytest.raises(TimeoutError, match="did not complete within 2 polls"):
        EcdsRequest(state_store, session=session).poll_until_complete(0, 2)


def test_polling_waits_for_success_even_once_the_body_carries_a_url(
    tmp_path: Path, clock: FakeClock
) -> None:
    """A queued job's status document can carry hrefs of its own; only success returns one."""
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("id", {}, SUBMITTED_AT, "status"))
    session = session_mock()
    session.get.return_value = response(
        {"status": "running", "asset": {"value": {"href": "https://example.test/blob"}}}
    )

    with pytest.raises(TimeoutError):
        EcdsRequest(state_store, session=session).poll_until_complete(30, 3)


def test_polling_rejects_a_status_response_without_a_status(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("id", {}, SUBMITTED_AT, "status"))
    session = session_mock()
    session.get.return_value = response({"jobID": "job-1"})

    with pytest.raises(AssertionError, match="has no status"):
        EcdsRequest(state_store, session=session).poll_until_complete(30, 3)


def test_poll_backoff_follows_consecutive_failures_not_the_poll_count(
    tmp_path: Path, clock: FakeClock
) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("id", {}, SUBMITTED_AT, "status"))
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
    state_store.write(RequestState("id", {}, SUBMITTED_AT, "status"))
    session = session_mock()
    session.get.side_effect = requests.ConnectionError("down")

    with pytest.raises(TimeoutError):
        EcdsRequest(state_store, session=session).poll_until_complete(30, 240)

    assert clock.now <= 30 * 240
    assert state_store.read().poll_failures > 1


def test_download_uses_the_result_url_saved_by_poll(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(
        RequestState(
            "id", {}, SUBMITTED_AT, "status", result_url="https://example.test/blob"
        )
    )
    message = grib_message()
    download_response = response({})
    download_response.iter_content.return_value = [message]
    session = session_mock()
    session.get.return_value = download_response

    state = EcdsRequest(state_store, download_session=session).download(
        tmp_path / "blob.grib2"
    )

    assert (tmp_path / "blob.grib2").read_bytes() == message
    assert state.downloaded_bytes == len(message)
    assert state.grib_messages == 1
    assert state.status == "downloaded"


def test_download_resumes_a_partial_file_after_http_206(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(
        RequestState(
            "id", {}, SUBMITTED_AT, "status", result_url="https://example.test/blob"
        )
    )
    target = tmp_path / "blob.grib2"
    message = grib_message()
    target.with_suffix(".grib2.partial").write_bytes(message[:4])
    download_response = response({}, status_code=requests.codes.partial)
    download_response.iter_content.return_value = [message[4:]]
    session = session_mock()
    session.get.return_value = download_response

    EcdsRequest(state_store, download_session=session).download(target)

    assert target.read_bytes() == message
    assert session.get.call_args.kwargs["headers"] == {"Range": "bytes=4-"}


def test_download_restarts_a_partial_file_after_http_200(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(
        RequestState(
            "id", {}, SUBMITTED_AT, "status", result_url="https://example.test/blob"
        )
    )
    target = tmp_path / "blob.grib2"
    message = grib_message()
    target.with_suffix(".grib2.partial").write_bytes(b"stale bytes")
    download_response = response({})
    download_response.iter_content.return_value = [message]
    session = session_mock()
    session.get.return_value = download_response

    EcdsRequest(state_store, download_session=session).download(target)

    assert target.read_bytes() == message
    assert session.get.call_args.kwargs["headers"] == {"Range": "bytes=11-"}


def test_download_refetches_the_whole_blob_after_http_416(tmp_path: Path) -> None:
    """A partial file longer than the result makes the ranged request unsatisfiable."""
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(
        RequestState(
            "id", {}, SUBMITTED_AT, "status", result_url="https://example.test/blob"
        )
    )
    target = tmp_path / "blob.grib2"
    message = grib_message()
    target.with_suffix(".grib2.partial").write_bytes(b"stale bytes")
    range_response = response(
        {}, status_code=requests.codes.requested_range_not_satisfiable
    )
    range_response.raise_for_status.side_effect = requests.HTTPError(
        "416 Client Error: Requested Range Not Satisfiable"
    )
    full_response = response({})
    full_response.iter_content.return_value = [message]
    session = session_mock()
    session.get.side_effect = [range_response, full_response]

    EcdsRequest(state_store, download_session=session).download(target)

    assert target.read_bytes() == message
    range_call, full_call = session.get.call_args_list
    assert range_call.kwargs["headers"] == {"Range": "bytes=11-"}
    assert full_call.kwargs["headers"] == {}
    range_response.close.assert_called_once_with()


def test_download_rejects_a_truncated_blob(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(
        RequestState(
            "id", {}, SUBMITTED_AT, "status", result_url="https://example.test/blob"
        )
    )
    download_response = response({})
    download_response.iter_content.return_value = [grib_message()[:-4]]
    session = session_mock()
    session.get.return_value = download_response

    with pytest.raises(AssertionError, match="Truncated message"):
        EcdsRequest(state_store, download_session=session).download(
            tmp_path / "blob.grib2"
        )

    assert not (tmp_path / "blob.grib2").exists()


def test_retrieve_resumes_an_in_flight_request_without_resubmitting(
    tmp_path: Path,
) -> None:
    payload = {"variable": ["total_precipitation"]}
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("job-1", payload, SUBMITTED_AT, "status"))
    message = grib_message()
    download_response = response({})
    download_response.iter_content.return_value = [message]
    session = session_mock()
    download_session = Mock(headers={})
    session.get.side_effect = [
        response({"status": "successful", "asset": {"value": {"href": "blob"}}}),
    ]

    download_session.get.return_value = download_response

    EcdsRequest(
        state_store, session=session, download_session=download_session
    ).retrieve(payload, tmp_path / "blob.grib2", poll_seconds=0)

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
            SUBMITTED_AT,
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


def test_retrieve_downloads_again_when_the_archived_blob_does_not_match(
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
            SUBMITTED_AT,
            "status",
            status="downloaded",
            downloaded_bytes=len(message),
            grib_messages=2,
        )
    )
    download_response = response({})
    download_response.iter_content.return_value = [message, message]
    session = session_mock()
    download_session = Mock(headers={})
    session.get.side_effect = [
        response({"status": "successful", "asset": {"value": {"href": "blob"}}}),
    ]

    download_session.get.return_value = download_response

    EcdsRequest(
        state_store, session=session, download_session=download_session
    ).retrieve(payload, target, poll_seconds=0)

    session.post.assert_not_called()
    assert state_store.read().grib_messages == 2


def test_retrieve_resubmits_after_a_terminal_failure(tmp_path: Path) -> None:
    payload = {"variable": ["total_precipitation"]}
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(
        RequestState("job-1", payload, SUBMITTED_AT, "status", status="failed")
    )
    submit_response = response({"jobID": "job-2"})
    session = session_mock()
    download_session = Mock(headers={})
    session.post.return_value = submit_response
    download_response = response({})
    download_response.iter_content.return_value = [grib_message()]
    session.get.side_effect = [
        response({"status": "successful", "asset": {"value": {"href": "blob"}}}),
    ]

    download_session.get.return_value = download_response

    EcdsRequest(
        state_store, session=session, download_session=download_session
    ).retrieve(payload, tmp_path / "blob.grib2", poll_seconds=0)

    session.post.assert_called_once()
    assert state_store.read().request_id == "job-2"


def test_retrieve_submits_a_failed_job_again_after_a_wait(
    tmp_path: Path, clock: FakeClock
) -> None:
    payload = {"variable": ["total_precipitation"]}
    state_store = StateStore(tmp_path / "state.json")
    session = session_mock()
    download_session = Mock(headers={})
    session.post.side_effect = [
        response({"jobID": "job-1"}),
        response({"jobID": "job-2"}),
    ]
    download_response = response({})
    download_response.iter_content.return_value = [grib_message()]
    session.get.side_effect = [
        response({"status": "failed"}),
        response({"status": "successful", "asset": {"value": {"href": "blob"}}}),
    ]

    download_session.get.return_value = download_response

    EcdsRequest(
        state_store, session=session, download_session=download_session
    ).retrieve(
        payload, tmp_path / "blob.grib2", poll_seconds=0, resubmit_wait_seconds=60
    )

    assert clock.sleeps == [60]
    assert session.post.call_count == 2
    assert state_store.read().request_id == "job-2"
    assert (tmp_path / "blob.grib2").exists()


def test_retrieve_waits_longer_after_each_failed_job_until_the_budget_is_spent(
    tmp_path: Path, clock: FakeClock
) -> None:
    payload = {"variable": ["total_precipitation"]}
    state_store = StateStore(tmp_path / "state.json")
    session = session_mock()
    session.post.side_effect = [response({"jobID": f"job-{n}"}) for n in range(1, 10)]
    session.get.return_value = response({"status": "failed"})

    with pytest.raises(EcdsJobFailedError, match="job-4 ended with status failed"):
        EcdsRequest(state_store, session=session).retrieve(
            payload,
            tmp_path / "blob.grib2",
            poll_seconds=0,
            resubmit_wait_seconds=10,
            resubmit_budget_seconds=100,
        )

    assert clock.sleeps == [10, 20, 40]
    assert session.post.call_count == 4
    assert state_store.read().status == "failed"
    assert not (tmp_path / "blob.grib2").exists()


def test_constraints_retries_a_transient_server_error() -> None:
    """ECDS intermittently 502s these endpoints; one blip must not abort an archiving run."""
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


def test_a_job_whose_results_are_gone_is_submitted_again(
    tmp_path: Path, clock: FakeClock
) -> None:
    """ECDS purges a finished job's result; a resumed job whose result has expired must be replaced, not polled."""
    payload = {"variable": ["total_precipitation"]}
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("job-1", payload, SUBMITTED_AT, "status"))
    session = session_mock()
    download_session = Mock(headers={})
    session.post.return_value = response({"jobID": "job-2"})
    download_response = response({})
    download_response.iter_content.return_value = [grib_message()]
    session.get.side_effect = [
        response(
            {"status": "successful", "links": [{"rel": "results", "href": "results"}]}
        ),
        response({"title": "Not Found"}, status_code=404),
        response({"status": "successful", "asset": {"value": {"href": "blob"}}}),
    ]

    download_session.get.return_value = download_response

    EcdsRequest(
        state_store, session=session, download_session=download_session
    ).retrieve(
        payload, tmp_path / "blob.grib2", poll_seconds=30, resubmit_wait_seconds=60
    )

    assert clock.sleeps == [60]
    session.post.assert_called_once()
    assert state_store.read().request_id == "job-2"
    assert (tmp_path / "blob.grib2").exists()


def test_a_job_ecds_no_longer_knows_is_submitted_again(
    tmp_path: Path, clock: FakeClock
) -> None:
    payload = {"variable": ["total_precipitation"]}
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("job-1", payload, SUBMITTED_AT, "status"))
    session = session_mock()
    download_session = Mock(headers={})
    session.post.return_value = response({"jobID": "job-2"})
    download_response = response({})
    download_response.iter_content.return_value = [grib_message()]
    session.get.side_effect = [
        response({"title": "Not Found"}, status_code=404),
        response({"status": "successful", "asset": {"value": {"href": "blob"}}}),
    ]

    download_session.get.return_value = download_response

    EcdsRequest(
        state_store, session=session, download_session=download_session
    ).retrieve(
        payload, tmp_path / "blob.grib2", poll_seconds=30, resubmit_wait_seconds=60
    )

    session.post.assert_called_once()
    assert state_store.read().request_id == "job-2"


def test_an_expired_job_is_recorded_as_terminal(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("job-1", {}, SUBMITTED_AT, "status"))
    session = session_mock()
    session.get.return_value = response({"title": "Not Found"}, status_code=404)

    with pytest.raises(EcdsJobFailedError, match="job-1 ended with status expired"):
        EcdsRequest(state_store, session=session).poll_until_complete(30, 240)

    state = state_store.read()
    assert state.status == "expired"
    assert "404" in state.errors[-1]


def test_retrieve_replaces_a_job_whose_polling_is_exhausted(
    tmp_path: Path, clock: FakeClock
) -> None:
    payload = {"variable": ["total_precipitation"]}
    state_store = StateStore(tmp_path / "state.json")
    session = session_mock()
    download_session = Mock(headers={})
    session.post.side_effect = [
        response({"jobID": "job-1"}),
        response({"jobID": "job-2"}),
    ]
    download_response = response({})
    download_response.iter_content.return_value = [grib_message()]
    running = response({"status": "running"})
    session.get.side_effect = [running] * 240 + [
        response({"status": "successful", "asset": {"value": {"href": "blob"}}}),
    ]

    download_session.get.return_value = download_response

    EcdsRequest(
        state_store, session=session, download_session=download_session
    ).retrieve(payload, tmp_path / "blob.grib2", poll_seconds=30, maximum_polls=240)

    assert clock.now >= 30 * 240
    assert session.post.call_count == 2
    assert state_store.read().request_id == "job-2"
    assert (tmp_path / "blob.grib2").exists()


def test_retrieve_gives_up_after_the_replacement_job_also_exhausts_polling(
    tmp_path: Path, clock: FakeClock
) -> None:
    payload = {"variable": ["total_precipitation"]}
    state_store = StateStore(tmp_path / "state.json")
    session = session_mock()
    session.post.side_effect = [response({"jobID": f"job-{n}"}) for n in range(1, 6)]
    session.get.return_value = response({"status": "running"})

    with pytest.raises(TimeoutError, match="did not complete within 3 polls"):
        EcdsRequest(state_store, session=session).retrieve(
            payload, tmp_path / "blob.grib2", poll_seconds=30, maximum_polls=3
        )

    assert session.post.call_count == 2
    assert state_store.read().status == "abandoned"

    session.reset_mock()
    with pytest.raises(TimeoutError):
        EcdsRequest(state_store, session=session).retrieve(
            payload, tmp_path / "blob.grib2", poll_seconds=30, maximum_polls=3
        )

    # An abandoned job is replaced before any poll, not resumed for another budget.
    assert session.mock_calls[0] == call.post(
        "https://ecds.ecmwf.int/api/retrieve/v1/processes/s2s-forecasts/execution",
        json={"inputs": payload},
        timeout=60,
    )


def test_a_replacement_job_does_not_resume_the_abandoned_jobs_partial_download(
    tmp_path: Path, clock: FakeClock
) -> None:
    payload = {"variable": ["total_precipitation"]}
    target = tmp_path / "blob.grib2"
    target.with_suffix(".grib2.partial").write_bytes(b"bytes of job-1")
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(
        RequestState("job-1", payload, SUBMITTED_AT, "status", status="failed")
    )
    session = session_mock()
    download_session = Mock(headers={})
    session.post.return_value = response({"jobID": "job-2"})
    download_response = response({})
    download_response.iter_content.return_value = [grib_message()]
    session.get.side_effect = [
        response({"status": "successful", "asset": {"value": {"href": "blob"}}}),
    ]

    download_session.get.return_value = download_response

    EcdsRequest(
        state_store, session=session, download_session=download_session
    ).retrieve(payload, target, poll_seconds=0)

    assert download_session.get.call_args.kwargs["headers"] == {}
    assert target.read_bytes() == grib_message()


def test_a_result_that_is_gone_at_download_time_is_submitted_again(
    tmp_path: Path, clock: FakeClock
) -> None:
    payload = {"variable": ["total_precipitation"]}
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("job-1", payload, SUBMITTED_AT, "status"))
    session = session_mock()
    download_session = Mock(headers={})
    session.post.return_value = response({"jobID": "job-2"})
    successful = response(
        {"status": "successful", "asset": {"value": {"href": "blob"}}}
    )
    gone = response({}, status_code=404)
    download_response = response({})
    download_response.iter_content.return_value = [grib_message()]
    session.get.side_effect = [successful, successful]
    download_session.get.side_effect = [gone, response({}), download_response]

    EcdsRequest(
        state_store, session=session, download_session=download_session
    ).retrieve(
        payload, tmp_path / "blob.grib2", poll_seconds=30, resubmit_wait_seconds=60
    )

    gone.close.assert_called_once_with()
    assert clock.sleeps == [60]
    assert state_store.read().request_id == "job-2"
    assert (tmp_path / "blob.grib2").read_bytes() == grib_message()


def test_a_result_that_keeps_disappearing_exhausts_the_budget(
    tmp_path: Path, clock: FakeClock
) -> None:
    payload = {"variable": ["total_precipitation"]}
    state_store = StateStore(tmp_path / "state.json")
    session = session_mock()
    download_session = Mock(headers={})
    session.post.side_effect = [response({"jobID": f"job-{n}"}) for n in range(1, 10)]
    successful = response(
        {"status": "successful", "asset": {"value": {"href": "blob"}}}
    )
    session.get.return_value = successful
    download_session.get.return_value = response({}, status_code=404)

    with pytest.raises(EcdsJobFailedError, match="job-4 ended with status expired"):
        EcdsRequest(
            state_store, session=session, download_session=download_session
        ).retrieve(
            payload,
            tmp_path / "blob.grib2",
            poll_seconds=0,
            resubmit_wait_seconds=10,
            resubmit_budget_seconds=100,
        )

    assert clock.sleeps == [10, 20, 40]
    assert state_store.read().status == "expired"
    assert not any("blob" in error for error in state_store.read().errors)


@pytest.mark.parametrize(
    "location",
    ["/jobs/next", "next", "https://API.test:443/jobs/next"],
)
def test_api_redirects_keep_the_token_within_the_origin(
    offline_request: tuple[EcdsRequest, OfflineAdapter, OfflineAdapter],
    location: str,
) -> None:
    request, api, downloads = offline_request
    request.state_store.write(RequestState("id", {}, SUBMITTED_AT, "/jobs/id"))
    api.responses = [
        wire_response(status_code=307, location=location),
        wire_response({"status": "running"}),
    ]

    request.poll_once()

    assert len(api.sent) == 2
    assert all(sent.headers["PRIVATE-TOKEN"] == "marker-a" for sent in api.sent)
    assert not downloads.sent


@pytest.mark.parametrize(
    "location",
    [
        "https://foreign.test/jobs/id",
        "//foreign.test/jobs/id",
        "https://api.test:444/jobs/id",
        "http://api.test/jobs/id",
        "https://marker-b@api.test/jobs/id",
    ],
)
def test_api_redirects_are_blocked_before_a_foreign_or_insecure_send(
    offline_request: tuple[EcdsRequest, OfflineAdapter, OfflineAdapter],
    location: str,
) -> None:
    request, api, downloads = offline_request
    request.state_store.write(RequestState("id", {}, SUBMITTED_AT, "/jobs/id"))
    api.responses = [wire_response(status_code=307, location=location)]

    with pytest.raises(ValueError, match="HTTPS"):
        request.poll_until_complete(0, 2)

    assert len(api.sent) == 1
    assert api.sent[0].url == "https://api.test/jobs/id"
    assert api.sent[0].headers["PRIVATE-TOKEN"] == "marker-a"
    assert request.state_store.read().poll_failures == 0
    assert not downloads.sent


@pytest.mark.parametrize(
    "url",
    [
        "https://foreign.test/jobs/id",
        "//foreign.test/jobs/id",
        "https://api.test:444/jobs/id",
        "http://api.test/jobs/id",
        "https://marker-b@api.test/jobs/id",
        "https://api.test:invalid/jobs/id",
    ],
)
def test_persisted_status_urls_are_validated_before_sending(
    offline_request: tuple[EcdsRequest, OfflineAdapter, OfflineAdapter],
    url: str,
) -> None:
    request, api, downloads = offline_request
    request.state_store.write(RequestState("id", {}, SUBMITTED_AT, url))

    with pytest.raises(ValueError, match="HTTPS"):
        request.poll_once()

    assert not api.sent
    assert not downloads.sent


@pytest.mark.parametrize("status", ["successful", "failed"])
@pytest.mark.parametrize(
    "url",
    [
        "https://foreign.test/results",
        "http://api.test/results",
        "https://api.test:444/results",
    ],
)
def test_api_results_links_are_validated_for_success_and_failure(
    offline_request: tuple[EcdsRequest, OfflineAdapter, OfflineAdapter],
    status: str,
    url: str,
) -> None:
    request, api, downloads = offline_request
    request.state_store.write(RequestState("id", {}, SUBMITTED_AT, "/jobs/id"))
    api.responses = [
        wire_response({"status": status, "links": [{"rel": "results", "href": url}]})
    ]

    with pytest.raises(ValueError, match="HTTPS"):
        request.poll_once()

    assert len(api.sent) == 1
    assert not downloads.sent


@pytest.mark.parametrize("in_header", [True, False])
@pytest.mark.parametrize(
    "location", ["/jobs/id", "jobs/id", "https://foreign.test/jobs/id"]
)
def test_submitted_status_locations_are_resolved_and_validated(
    offline_request: tuple[EcdsRequest, OfflineAdapter, OfflineAdapter],
    in_header: bool,
    location: str,
) -> None:
    request, api, downloads = offline_request
    body = {"jobID": "id"} if in_header else {"jobID": "id", "location": location}
    api.responses = [wire_response(body, location=location if in_header else None)]

    if "foreign.test" in location:
        with pytest.raises(ValueError, match="configured HTTPS origin"):
            request.submit({})
        assert not request.state_store.path.exists()
    else:
        state = request.submit({})
        assert state.status_url == (
            "https://api.test/jobs/id"
            if location.startswith("/")
            else "https://api.test/api/retrieve/v1/processes/s2s-forecasts/jobs/id"
        )
    assert len(api.sent) == 1
    assert not downloads.sent


def test_relative_links_follow_the_final_api_response_url(
    offline_request: tuple[EcdsRequest, OfflineAdapter, OfflineAdapter],
    tmp_path: Path,
) -> None:
    request, api, downloads = offline_request
    request.state_store.write(RequestState("id", {}, SUBMITTED_AT, "/jobs/id"))
    api.responses = [
        wire_response(status_code=307, location="/jobs/id/status"),
        wire_response(
            {"status": "successful", "links": [{"rel": "results", "href": "results"}]}
        ),
        wire_response(status_code=307, location="/results/id/metadata"),
        wire_response({"asset": {"value": {"href": "blob?value=marker-b"}}}),
    ]
    downloads.responses = [wire_response(grib_message()), wire_response(grib_message())]

    _, result_url = request.poll_once()
    request.download(tmp_path / "blob.grib2", result_url)

    assert api.sent[2].url == "https://api.test/jobs/id/results"
    assert all(sent.headers["PRIVATE-TOKEN"] == "marker-a" for sent in api.sent)
    assert all(
        sent.url == "https://api.test/results/id/blob?value=marker-b"
        for sent in downloads.sent
    )
    assert all("PRIVATE-TOKEN" not in sent.headers for sent in downloads.sent)


@pytest.mark.parametrize("port", [443, 8443])
def test_configured_api_ports_define_the_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, port: int
) -> None:
    monkeypatch.setenv("ECDS_API_KEY", "marker-a")
    request = EcdsRequest(
        StateStore(tmp_path / "state.json"), api_url=f"https://api.test:{port}/api"
    )
    request.session.trust_env = False
    api = OfflineAdapter()
    request.session.mount("https://", api)
    request.state_store.write(
        RequestState("id", {}, SUBMITTED_AT, f"https://api.test:{port}/jobs/id")
    )
    api.responses = [wire_response({"status": "running"})]

    request.poll_once()

    assert api.sent[0].headers["PRIVATE-TOKEN"] == "marker-a"
    if port != 443:
        request.state_store.write(
            RequestState("id", {}, SUBMITTED_AT, "https://api.test/jobs/id")
        )
        with pytest.raises(ValueError, match="configured HTTPS origin"):
            request.poll_once()
        assert len(api.sent) == 1


@pytest.mark.parametrize(
    "url", ["http://api.test/api", "https://marker-b@api.test/api"]
)
def test_insecure_api_configuration_is_rejected(tmp_path: Path, url: str) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        EcdsRequest(StateStore(tmp_path / "state.json"), api_url=url)


@pytest.mark.parametrize(
    "result_url", ["https://api.test/blob", "https://files.test/blob?value=marker-b"]
)
def test_download_redirects_use_an_unauthenticated_transport(
    offline_request: tuple[EcdsRequest, OfflineAdapter, OfflineAdapter],
    tmp_path: Path,
    result_url: str,
) -> None:
    request, api, downloads = offline_request
    request.state_store.write(
        RequestState("id", {}, SUBMITTED_AT, "/jobs/id", result_url=result_url)
    )
    downloads.responses = [
        wire_response(
            status_code=307, location="https://other.test:8443/blob?value=marker-c"
        ),
        wire_response(grib_message()),
        wire_response(grib_message()),
    ]

    state = request.download(tmp_path / "blob.grib2")

    assert state.status == "downloaded"
    assert len(downloads.sent) == 3
    assert all("PRIVATE-TOKEN" not in sent.headers for sent in downloads.sent)
    assert request.download_session.trust_env
    assert not api.sent


@pytest.mark.parametrize("redirect", [True, False])
def test_downloads_reject_http_before_sending(
    offline_request: tuple[EcdsRequest, OfflineAdapter, OfflineAdapter],
    tmp_path: Path,
    redirect: bool,
) -> None:
    request, api, downloads = offline_request
    request.state_store.write(RequestState("id", {}, SUBMITTED_AT, "/jobs/id"))
    url = "http://files.test/blob?value=marker-b"
    if redirect:
        downloads.responses = [wire_response(status_code=307, location=url)]
        url = "https://files.test/blob?value=marker-c"

    with pytest.raises(ValueError, match="HTTPS") as error:
        request.download(tmp_path / "blob.grib2", url)

    assert "marker-b" not in str(error.value)
    assert "marker-c" not in str(error.value)
    assert len(downloads.sent) == int(redirect)
    assert all("PRIVATE-TOKEN" not in sent.headers for sent in downloads.sent)
    assert not api.sent


@pytest.mark.parametrize("status_code", [206, 200, 416])
def test_download_range_recovery_uses_the_unauthenticated_session(
    offline_request: tuple[EcdsRequest, OfflineAdapter, OfflineAdapter],
    tmp_path: Path,
    status_code: int,
) -> None:
    request, api, downloads = offline_request
    request.state_store.write(
        RequestState(
            "id",
            {},
            SUBMITTED_AT,
            "/jobs/id",
            result_url="https://files.test/blob?value=marker-b",
        )
    )
    target = tmp_path / "blob.grib2"
    message = grib_message()
    target.with_suffix(".grib2.partial").write_bytes(message[:4])
    downloads.responses = [
        wire_response(message[4:] if status_code == 206 else message, status_code)
    ]
    if status_code == 416:
        downloads.responses.append(wire_response(message))

    request.download(target)

    assert target.read_bytes() == message
    assert downloads.sent[0].headers["Range"] == "bytes=4-"
    if status_code == 416:
        assert len(downloads.sent) == 2
        assert "Range" not in downloads.sent[1].headers
    assert all("PRIVATE-TOKEN" not in sent.headers for sent in downloads.sent)
    assert not api.sent


class InterruptedBody(BytesIO):
    def stream(self, chunk_size: int, decode_content: bool) -> Iterator[bytes]:
        assert decode_content
        yield self.read(chunk_size)
        raise requests.ConnectionError("transient")


def test_download_retry_resumes_without_api_authentication(
    offline_request: tuple[EcdsRequest, OfflineAdapter, OfflineAdapter],
    tmp_path: Path,
) -> None:
    request, api, downloads = offline_request
    request.state_store.write(
        RequestState(
            "id",
            {},
            SUBMITTED_AT,
            "/jobs/id",
            result_url="https://files.test/blob?value=marker-b",
        )
    )
    target = tmp_path / "blob.grib2"
    message = grib_message()
    interrupted = wire_response()
    interrupted.raw = InterruptedBody(message[:4])
    downloads.responses = [
        wire_response(message),
        interrupted,
        wire_response(message[4:], status_code=206),
    ]

    with pytest.raises(requests.ConnectionError, match="transient"):
        request.download(target)
    assert request.state_store.read().status == "submitted"
    assert target.with_suffix(".grib2.partial").read_bytes() == message[:4]
    request.download(target)

    assert target.read_bytes() == message
    assert downloads.sent[2].headers["Range"] == "bytes=4-"
    assert all("PRIVATE-TOKEN" not in sent.headers for sent in downloads.sent)
    assert not api.sent


def test_api_poll_retry_stays_authenticated_and_download_does_not(
    offline_request: tuple[EcdsRequest, OfflineAdapter, OfflineAdapter],
    tmp_path: Path,
    clock: FakeClock,
) -> None:
    request, api, downloads = offline_request
    request.state_store.write(RequestState("id", {}, SUBMITTED_AT, "/jobs/id"))
    api.responses = [
        wire_response(status_code=503),
        wire_response(
            {
                "status": "successful",
                "asset": {"value": {"href": "https://files.test/blob?value=marker-b"}},
            }
        ),
    ]
    downloads.responses = [wire_response(grib_message()), wire_response(grib_message())]

    _, result_url = request.poll_until_complete(30, 3)
    request.download(tmp_path / "blob.grib2", result_url)

    assert clock.sleeps == [30]
    assert len(api.sent) == 2
    assert all(sent.headers["PRIVATE-TOKEN"] == "marker-a" for sent in api.sent)
    assert "PRIVATE-TOKEN" not in downloads.sent[0].headers
    assert request.state_store.read().poll_failures == 0


def test_download_http_errors_omit_signed_urls_and_keep_the_status(
    offline_request: tuple[EcdsRequest, OfflineAdapter, OfflineAdapter],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    request, api, downloads = offline_request
    request.state_store.write(
        RequestState(
            "id",
            {},
            SUBMITTED_AT,
            "/jobs/id",
            result_url="https://files.test/blob?value=marker-b",
        )
    )
    downloads.responses = [wire_response(status_code=503)]

    with pytest.raises(requests.HTTPError, match="HTTP 503") as error:
        request.download(tmp_path / "blob.grib2")

    assert "marker-b" not in str(error.value)
    assert "files.test" not in str(error.value)
    assert error.value.__suppress_context__
    assert "marker-a" not in caplog.text
    assert "marker-b" not in caplog.text
    assert request.state_store.read().status == "submitted"
    assert not api.sent


@pytest.mark.parametrize("netrc", [True, False])
def test_download_prepared_requests_remove_default_and_netrc_authentication(
    offline_request: tuple[EcdsRequest, OfflineAdapter, OfflineAdapter],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    netrc: bool,
) -> None:
    request, api, downloads = offline_request
    request.state_store.write(
        RequestState(
            "id",
            {},
            SUBMITTED_AT,
            "/jobs/id",
            result_url="https://files.test/blob?value=marker-b",
        )
    )
    request.download_session.headers["Authorization"] = "marker-c"
    request.download_session.headers["PRIVATE-TOKEN"] = "marker-d"
    if netrc:
        monkeypatch.setattr(
            requests.sessions,
            "get_netrc_auth",
            Mock(return_value=("marker-e", "marker-f")),
        )
    downloads.responses = [
        wire_response(status_code=307, location="https://other.test/blob"),
        wire_response(grib_message()),
        wire_response(grib_message()),
    ]

    request.download(tmp_path / "blob.grib2")

    assert len(downloads.sent) == 3
    assert all("Authorization" not in sent.headers for sent in downloads.sent)
    assert all("PRIVATE-TOKEN" not in sent.headers for sent in downloads.sent)
    assert request.download_session.trust_env
    assert not api.sent


@pytest.mark.parametrize("prepared", [True, False])
def test_default_api_session_blocks_initial_foreign_requests(
    offline_request: tuple[EcdsRequest, OfflineAdapter, OfflineAdapter],
    prepared: bool,
) -> None:
    request, api, downloads = offline_request
    url = "https://foreign.test/jobs/id"

    if prepared:
        with pytest.raises(ValueError, match="configured HTTPS origin"):
            request.session.send(requests.Request("GET", url).prepare())
    else:
        with pytest.raises(ValueError, match="configured HTTPS origin"):
            request.session.get(url)

    assert not api.sent
    assert not downloads.sent


def test_submitted_relative_location_uses_the_final_response_url(
    offline_request: tuple[EcdsRequest, OfflineAdapter, OfflineAdapter],
) -> None:
    request, api, downloads = offline_request
    api.responses = [
        wire_response(status_code=307, location="/submission/start"),
        wire_response({"jobID": "id"}, status_code=201, location="jobs/id"),
    ]

    state = request.submit({"variable": ["tp"]})

    assert state.status_url == "https://api.test/submission/jobs/id"
    assert len(api.sent) == 2
    assert all(sent.method == "POST" for sent in api.sent)
    assert all(sent.headers["PRIVATE-TOKEN"] == "marker-a" for sent in api.sent)
    assert not downloads.sent


SIGNED_RESULT_URL = "https://files.test/blob?value=marker-b"
RANGED_BLOB = b"".join(grib_message(bytes([n]) * 50) for n in range(8))


class RangeServer(BaseAdapter):
    """Serves one object like the ECDS result store, answering Range requests with 206."""

    def __init__(
        self, body: bytes = RANGED_BLOB, etag: str | None = '"etag-1"'
    ) -> None:
        self.body = body
        self.etag = etag
        self.lock = threading.Lock()
        self.ranges: list[str | None] = []
        self.if_matches: list[str | None] = []
        self.ignore_range = False
        self.reply: Callable[[int, int, requests.Response], requests.Response] = (
            lambda start, end, result: result
        )

    def send(
        self, request: requests.PreparedRequest, *_args: object, **_kwargs: object
    ) -> requests.Response:
        assert "PRIVATE-TOKEN" not in request.headers
        range_header = request.headers.get("Range")
        with self.lock:
            self.ranges.append(range_header)
            self.if_matches.append(request.headers.get("If-Match"))
        if range_header is None or self.ignore_range:
            result = wire_response(self.body)
        else:
            first, _, last = range_header.removeprefix("bytes=").partition("-")
            start, end = int(first), min(int(last), len(self.body) - 1)
            result = wire_response(self.body[start : end + 1], status_code=206)
            result.headers["Content-Range"] = f"bytes {start}-{end}/{len(self.body)}"
            result = self.reply(start, end, result)
        if self.etag is not None:
            result.headers.setdefault("ETag", self.etag)
        assert request.url is not None
        result.url = request.url
        result.request = request
        return result

    def close(self) -> None:
        pass


@pytest.fixture
def range_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[EcdsRequest, RangeServer]:
    monkeypatch.setattr(ecds_client, "MINIMUM_RANGE_BYTES", 1)
    request = EcdsRequest(
        StateStore(tmp_path / "state.json"), api_url="https://api.test/api"
    )
    request.state_store.write(
        RequestState("id", {}, SUBMITTED_AT, "/jobs/id", result_url=SIGNED_RESULT_URL)
    )
    server = RangeServer()
    request.download_session.mount("https://", server)
    return request, server


def part_paths(target: Path) -> list[Path]:
    return sorted(target.parent.glob(f"{target.name}.partial.*"))


def test_a_ranged_download_assembles_the_parts_in_order(
    range_request: tuple[EcdsRequest, RangeServer], tmp_path: Path
) -> None:
    request, server = range_request
    target = tmp_path / "blob.grib2"

    state = request.download(target, download_ranges=4)

    assert target.read_bytes() == RANGED_BLOB
    assert state.downloaded_bytes == len(RANGED_BLOB)
    assert state.grib_messages == 8
    assert server.ranges[0] == "bytes=0-0"
    size = len(RANGED_BLOB)
    bounds = [size * index // 4 for index in range(5)]
    assert sorted(map(str, server.ranges[1:])) == sorted(
        f"bytes={start}-{end - 1}" for start, end in itertools.pairwise(bounds)
    )
    assert server.if_matches == [None] + ['"etag-1"'] * 4
    assert part_paths(target) == []
    assert not target.with_suffix(".grib2.partial").exists()


def test_retrieve_passes_the_range_count_to_the_download(
    range_request: tuple[EcdsRequest, RangeServer], tmp_path: Path
) -> None:
    request, server = range_request
    request.session = session_mock()
    request.session.get.return_value = response(
        {"status": "successful", "asset": {"value": {"href": SIGNED_RESULT_URL}}}
    )

    request.retrieve({}, tmp_path / "blob.grib2", poll_seconds=0, download_ranges=1)

    assert server.ranges == ["bytes=0-0", f"bytes=0-{len(RANGED_BLOB) - 1}"]


def test_a_result_below_the_minimum_range_size_uses_one_range(
    range_request: tuple[EcdsRequest, RangeServer],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, server = range_request
    monkeypatch.setattr(ecds_client, "MINIMUM_RANGE_BYTES", len(RANGED_BLOB) + 1)

    request.download(tmp_path / "blob.grib2", download_ranges=4)

    assert (tmp_path / "blob.grib2").read_bytes() == RANGED_BLOB
    assert server.ranges == ["bytes=0-0", f"bytes=0-{len(RANGED_BLOB) - 1}"]


def no_etag(server: RangeServer) -> None:
    server.etag = None


def weak_etag(server: RangeServer) -> None:
    server.etag = 'W/"etag-1"'


def ignored_range(server: RangeServer) -> None:
    server.ignore_range = True


def gzip_encoded(server: RangeServer) -> None:
    def reply(start: int, end: int, result: requests.Response) -> requests.Response:
        result.headers["Content-Encoding"] = "gzip"
        return result

    server.reply = reply


@pytest.mark.parametrize("configure", [ignored_range, no_etag, weak_etag, gzip_encoded])
def test_a_result_not_served_in_verifiable_ranges_is_read_as_one_stream(
    range_request: tuple[EcdsRequest, RangeServer],
    tmp_path: Path,
    configure: Callable[[RangeServer], None],
) -> None:
    request, server = range_request
    configure(server)

    request.download(tmp_path / "blob.grib2", download_ranges=4)

    assert (tmp_path / "blob.grib2").read_bytes() == RANGED_BLOB
    assert server.ranges == ["bytes=0-0", None]


def wrong_content_range(start: int, end: int, result: requests.Response) -> None:
    result.headers["Content-Range"] = f"bytes {start}-{end}/{len(RANGED_BLOB) + 1}"


def wrong_length(start: int, end: int, result: requests.Response) -> None:
    result.raw = BytesIO(RANGED_BLOB[start:end])


def changed_etag(start: int, end: int, result: requests.Response) -> None:
    result.headers["ETag"] = '"etag-2"'


def precondition_failed(start: int, end: int, result: requests.Response) -> None:
    result.status_code = 412
    result.raw = BytesIO(b"")
    del result.headers["Content-Range"]


def whole_body(start: int, end: int, result: requests.Response) -> None:
    result.status_code = 200
    result.raw = BytesIO(RANGED_BLOB)
    del result.headers["Content-Range"]


@pytest.mark.parametrize(
    "tamper",
    [wrong_content_range, wrong_length, changed_etag, precondition_failed, whole_body],
)
def test_a_part_that_does_not_match_the_result_discards_every_part(
    range_request: tuple[EcdsRequest, RangeServer],
    tmp_path: Path,
    tamper: Callable[[int, int, requests.Response], None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    request, server = range_request
    target = tmp_path / "blob.grib2"

    def reply(start: int, end: int, result: requests.Response) -> requests.Response:
        if start == len(RANGED_BLOB) // 4:
            tamper(start, end, result)
        return result

    server.reply = reply

    with pytest.raises(EcdsRangeMismatchError) as error:
        request.download(target, download_ranges=4)

    assert isinstance(error.value, RuntimeError)
    assert "marker-b" not in str(error.value)
    assert "marker-b" not in caplog.text
    assert part_paths(target) == []
    assert not target.exists()
    assert not target.with_suffix(".grib2.partial").exists()
    assert request.state_store.read().status == "submitted"


@pytest.mark.parametrize(
    "leftover",
    [
        {"blob.grib2.partial.0": RANGED_BLOB[:7], "blob.grib2.partial.1": b"stale"},
        {"blob.grib2.partial.assembly": RANGED_BLOB * 2},
    ],
)
def test_files_left_by_an_earlier_ranged_attempt_are_discarded(
    range_request: tuple[EcdsRequest, RangeServer],
    tmp_path: Path,
    leftover: dict[str, bytes],
) -> None:
    request, server = range_request
    target = tmp_path / "blob.grib2"
    for name, contents in leftover.items():
        (tmp_path / name).write_bytes(contents)

    request.download(target, download_ranges=2)

    assert target.read_bytes() == RANGED_BLOB
    size = len(RANGED_BLOB)
    assert sorted(map(str, server.ranges[1:])) == [
        f"bytes=0-{size // 2 - 1}",
        f"bytes={size // 2}-{size - 1}",
    ]
    assert part_paths(target) == []


def test_a_result_gone_during_a_part_is_expired(
    range_request: tuple[EcdsRequest, RangeServer], tmp_path: Path
) -> None:
    request, server = range_request

    def reply(start: int, end: int, result: requests.Response) -> requests.Response:
        return wire_response(status_code=404) if start > 0 else result

    server.reply = reply

    with pytest.raises(EcdsJobFailedError, match="ended with status expired"):
        request.download(tmp_path / "blob.grib2", download_ranges=4)

    state = request.state_store.read()
    assert state.status == "expired"
    assert not any("marker-b" in error for error in state.errors)


def test_a_part_connection_error_discards_parts_and_omits_the_signed_url(
    range_request: tuple[EcdsRequest, RangeServer],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    request, server = range_request
    target = tmp_path / "blob.grib2"

    def reply(start: int, end: int, result: requests.Response) -> requests.Response:
        if start == len(RANGED_BLOB) // 2:
            raise requests.ConnectionError(
                f"Max retries exceeded with url: {SIGNED_RESULT_URL}"
            )
        return result

    server.reply = reply

    with pytest.raises(requests.ConnectionError) as error:
        request.download(target, download_ranges=2)

    assert "marker-b" not in str(error.value)
    assert "files.test" not in str(error.value)
    assert error.value.__suppress_context__
    assert "marker-b" not in caplog.text
    assert part_paths(target) == []
    assert request.state_store.read().status == "submitted"


def test_resubmitting_discards_the_part_files_of_the_old_result(
    tmp_path: Path,
) -> None:
    payload = {"variable": ["total_precipitation"]}
    target = tmp_path / "blob.grib2"
    (tmp_path / "blob.grib2.partial.0").write_bytes(b"bytes of job-1")
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(
        RequestState("job-1", payload, SUBMITTED_AT, "status", status="failed")
    )
    session = session_mock()
    session.post.return_value = response({"jobID": "job-2"})
    session.get.return_value = response(
        {"status": "successful", "asset": {"value": {"href": "blob"}}}
    )
    download_session = Mock(headers={})
    download_response = response({})
    download_response.iter_content.return_value = [grib_message()]
    download_session.get.return_value = download_response

    EcdsRequest(
        state_store, session=session, download_session=download_session
    ).retrieve(payload, target, poll_seconds=0)

    assert part_paths(target) == []
    assert target.read_bytes() == grib_message()


def test_a_ranged_download_logs_network_time_apart_from_the_grib_scan(
    range_request: tuple[EcdsRequest, RangeServer],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    request, _ = range_request
    caplog.set_level("INFO")

    request.download(tmp_path / "blob.grib2", download_ranges=4)

    [record] = [r for r in caplog.records if r.getMessage().startswith("Downloaded")]
    assert "4 ranges" in record.getMessage()
    assert "MB/s" in record.getMessage()
    assert "counted 8 GRIB messages" in record.getMessage()
    assert "marker-b" not in caplog.text


def test_polling_logs_each_status_change_once(
    tmp_path: Path, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    state_store = StateStore(tmp_path / "state.json")
    state_store.write(RequestState("job-1", {}, SUBMITTED_AT, "status"))
    session = session_mock()
    session.get.side_effect = [
        response({"status": "accepted"}),
        response({"status": "accepted"}),
        response({"status": "running"}),
        response({"status": "running"}),
        response({"status": "successful", "asset": {"value": {"href": "blob"}}}),
    ]

    EcdsRequest(state_store, session=session).poll_until_complete(30, 10)

    transitions = [
        record.getMessage()
        for record in caplog.records
        if "after submission" in record.getMessage()
    ]
    assert [message.split()[4] for message in transitions] == [
        "accepted",
        "running",
        "successful",
    ]
    assert all(message.startswith("ECDS job job-1 is ") for message in transitions)


def test_a_mismatched_part_is_downloaded_again_without_resubmitting(
    range_request: tuple[EcdsRequest, RangeServer], tmp_path: Path
) -> None:
    request, server = range_request
    request.session = session_mock()
    request.session.get.return_value = response(
        {"status": "successful", "asset": {"value": {"href": SIGNED_RESULT_URL}}}
    )
    target = tmp_path / "blob.grib2"

    def reply(start: int, end: int, result: requests.Response) -> requests.Response:
        if start > 0:
            changed_etag(start, end, result)
        return result

    server.reply = reply

    with pytest.raises(EcdsRangeMismatchError):
        request.retrieve({}, target, poll_seconds=0, download_ranges=2)
    server.reply = lambda start, end, result: result
    request.retrieve({}, target, poll_seconds=0, download_ranges=2)

    request.session.post.assert_not_called()
    assert target.read_bytes() == RANGED_BLOB


class SlowBody(BytesIO):
    """A body that trickles out one byte at a time, as a part still mid-stream does."""

    def __init__(self, contents: bytes) -> None:
        super().__init__(contents)
        self.bytes_streamed = 0

    def stream(self, *_args: object, **_kwargs: object) -> Iterator[bytes]:
        while chunk := self.read(1):
            self.bytes_streamed += 1
            yield chunk
            time.sleep(0.01)


def test_a_part_failure_stops_and_joins_every_part_before_cleanup(
    range_request: tuple[EcdsRequest, RangeServer], tmp_path: Path
) -> None:
    request, server = range_request
    target = tmp_path / "blob.grib2"
    size = len(RANGED_BLOB)
    slow_bodies: list[SlowBody] = []
    first_part_streaming = threading.Event()

    def reply(start: int, end: int, result: requests.Response) -> requests.Response:
        if start == 0 and end > 0:
            body = SlowBody(RANGED_BLOB[start : end + 1])
            slow_bodies.append(body)
            result.raw = body
            first_part_streaming.set()
        elif start > 0:
            first_part_streaming.wait(5)
            time.sleep(0.05)
            changed_etag(start, end, result)
        return result

    server.reply = reply

    with pytest.raises(EcdsRangeMismatchError):
        request.download(target, download_ranges=2)
    time.sleep(0.1)

    assert part_paths(target) == []
    [slow_body] = slow_bodies
    assert 0 < slow_body.bytes_streamed < size // 2
