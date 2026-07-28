import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import requests
import xarray as xr

from scripts.ecmwf_extended_range_spike import (
    EcdsRequest,
    RequestMeasurement,
    StateStore,
)


def response(body: dict[str, object], status_code: int = 200) -> Mock:
    result = Mock()
    result.json.return_value = body
    result.status_code = status_code
    result.headers = {}
    result.raise_for_status.return_value = None
    return result


def grib_message(payload: bytes = b"contents") -> bytes:
    message_size = 16 + len(payload) + 4
    return (
        b"GRIB"
         b"\x00\x00\x00\x02"
        + message_size.to_bytes(8, byteorder="big")
        + payload
        + b"7777"
    )


def test_request_state_can_resume_in_another_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ECDS_API_KEY", "test-key")
    session = Mock()
    session.headers = {}
    submit_response = response({"jobID": "job-1"})
    submit_response.headers = {"Location": "status"}
    session.post.return_value = submit_response
    state_store = StateStore(tmp_path / "state.json")
    request = EcdsRequest(state_store, session=session)

    request.submit(
        {
            "number": [0, 1],
            "leadtime_hour": ["024"],
            "variable": ["10_m_u_component_of_wind"],
        }
    )

    resumed_session = Mock()
    resumed_session.headers = {}
    resumed_session.get.side_effect = [
        response(
            {
                "status": "successful",
                "created": "2026-07-28T17:46:07",
                "started": "2026-07-28T17:47:07",
                "finished": "2026-07-28T17:48:07",
                "links": [{"rel": "results", "href": "results"}],
            }
        ),
        response({"asset": {"value": {"href": "result"}}}),
    ]
    resumed = EcdsRequest(state_store, session=resumed_session)
    measurement, result_url = resumed.poll_once()
    assert measurement.request_id == "job-1"
    assert measurement.expected_members == [0, 1]
    assert measurement.expected_steps == [24]
    assert measurement.expected_variables == ["10_m_u_component_of_wind"]
    assert measurement.queue_seconds == 60
    assert measurement.server_processing_seconds == 60
    assert measurement.completed_at == "2026-07-28T17:48:07+00:00"
    assert result_url == "result"
    assert state_store.read().result_url == "result"
    assert resumed_session.get.call_args_list[1].args == ("results",)


def test_configures_endpoint_and_basic_credentials_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ECDS_API_ENDPOINT", "https://example.test/api/")
    monkeypatch.setenv("ECDS_API_KEY", "user:key")
    request = EcdsRequest(StateStore(tmp_path / "state.json"))

    assert request.api_url == (
        "https://example.test/api/retrieve/v1/processes/s2s-forecasts/execution"
    )
    assert request.session.auth == ("user", "key")


def test_configures_endpoint_and_token_from_cdsapirc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / ".cdsapirc"
    config_path.write_text(
        "url: https://example.test/api\nkey: personal-access-token\n"
    )
    monkeypatch.setenv("CDSAPI_RC", str(config_path))
    monkeypatch.delenv("ECDS_API_ENDPOINT", raising=False)
    monkeypatch.delenv("ECDS_API_KEY", raising=False)
    request = EcdsRequest(StateStore(tmp_path / "state.json"))

    assert request.api_url == (
        "https://example.test/api/retrieve/v1/processes/s2s-forecasts/execution"
    )
    assert request.session.headers["PRIVATE-TOKEN"] == "personal-access-token"
    assert "Authorization" not in request.session.headers


def test_state_writes_machine_readable_measurements_atomically(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "measurement.json")
    state_store.write(RequestMeasurement("id", {"date": "2026-07-24"}, "now", "status"))
    assert json.loads(state_store.path.read_text())["request_id"] == "id"
    assert not state_store.path.with_suffix(".json.tmp").exists()


def test_failed_poll_records_results_error(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "measurement.json")
    state_store.write(RequestMeasurement("id", {}, "now", "status"))
    failure: dict[str, object] = {
        "status": 400,
        "title": "The job has failed",
        "traceback": "failure detail",
    }
    session = Mock()
    session.headers = {}
    session.auth = ("user", "key")
    session.get.side_effect = [
        response(
            {
                "status": "failed",
                "links": [{"rel": "results", "href": "results"}],
            }
        ),
        response(failure, status_code=400),
    ]

    measurement, result_url = EcdsRequest(state_store, session=session).poll_once()

    assert result_url is None
    assert measurement.errors == [json.dumps(failure, sort_keys=True)]


def test_download_uses_result_url_saved_by_poll(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "measurement.json")
    state_store.write(
        RequestMeasurement(
            "id", {}, "now", "status", result_url="https://example.test/result"
        )
    )
    download_response = response({})
    message = grib_message()
    download_response.iter_content.return_value = [message]
    session = Mock()
    session.headers = {}
    session.auth = ("user", "key")
    session.get.return_value = download_response

    measurement = EcdsRequest(state_store, session=session).download(
        tmp_path / "result.grib"
    )

    assert measurement.downloaded_bytes == len(message)
    assert measurement.grib_messages == 1
    assert (tmp_path / "result.grib.complete").exists()


def test_download_resumes_partial_file_after_http_206(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "measurement.json")
    state_store.write(
        RequestMeasurement("id", {}, "now", "status", result_url="result")
    )
    target = tmp_path / "result.grib"
    message = grib_message()
    target.with_suffix(".grib.partial").write_bytes(message[:4])
    download_response = response({}, status_code=requests.codes.partial)
    download_response.iter_content.return_value = [message[4:]]
    session = Mock()
    session.headers = {}
    session.auth = ("user", "key")
    session.get.return_value = download_response

    measurement = EcdsRequest(state_store, session=session).download(target)

    assert target.read_bytes() == message
    assert measurement.interrupted_download_resumable
    assert session.get.call_args.kwargs["headers"] == {"Range": "bytes=4-"}


def test_download_restarts_partial_file_after_http_200(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "measurement.json")
    state_store.write(
        RequestMeasurement("id", {}, "now", "status", result_url="result")
    )
    target = tmp_path / "result.grib"
    target.with_suffix(".grib.partial").write_bytes(b"incomplete")
    download_response = response({})
    message = grib_message()
    download_response.iter_content.return_value = [message]
    session = Mock()
    session.headers = {}
    session.auth = ("user", "key")
    session.get.return_value = download_response

    measurement = EcdsRequest(state_store, session=session).download(target)

    assert target.read_bytes() == message
    assert not measurement.interrupted_download_resumable
    assert session.get.call_args.kwargs["headers"] == {"Range": "bytes=10-"}


def test_download_counts_structural_messages_not_embedded_grib_bytes(
    tmp_path: Path,
) -> None:
    state_store = StateStore(tmp_path / "measurement.json")
    state_store.write(
        RequestMeasurement("id", {}, "now", "status", result_url="result")
    )
    message = grib_message(b"compressed GRIB payload")
    download_response = response({})
    download_response.iter_content.return_value = [message]
    session = Mock()
    session.headers = {}
    session.auth = ("user", "key")
    session.get.return_value = download_response

    measurement = EcdsRequest(state_store, session=session).download(
        tmp_path / "result.grib"
    )

    assert measurement.grib_messages == 1


def test_complete_initialization_can_be_written_and_queried(tmp_path: Path) -> None:
    ensemble_members = np.arange(101)
    lead_times = np.arange(24, 46 * 24 + 1, 24).astype("timedelta64[h]")
    latitude = np.array([5.0, 0.0, -5.0])
    longitude = np.array([35.0, 40.0, 45.0])
    values = np.zeros((1, 46, 101, 3, 3), dtype=np.float32)
    dataset = xr.Dataset(
        {
            "temperature_2m": (
                ("init_time", "lead_time", "ensemble_member", "latitude", "longitude"),
                values,
            )
        },
        coords={
            "init_time": [np.datetime64("2026-07-24")],
            "lead_time": lead_times,
            "ensemble_member": ensemble_members,
            "latitude": latitude,
            "longitude": longitude,
        },
    ).assign_coords(valid_time=lambda ds: ds.init_time + ds.lead_time)
    store = tmp_path / "complete_initialization.zarr"
    dataset.to_zarr(store, mode="w")
    dataset.to_zarr(store, mode="w")

    reopened = xr.open_zarr(store)
    east_africa = reopened.sel(latitude=slice(5, -5), longitude=slice(35, 45))
    assert east_africa.sizes == {
        "init_time": 1,
        "lead_time": 46,
        "ensemble_member": 101,
        "latitude": 3,
        "longitude": 3,
    }
    assert np.array_equal(reopened.valid_time, reopened.init_time + reopened.lead_time)
