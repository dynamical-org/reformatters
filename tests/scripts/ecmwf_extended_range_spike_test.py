import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
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


def test_request_state_can_resume_in_another_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ECDS_API_KEY", "test-key")
    session = Mock()
    session.headers = {}
    session.post.return_value = response({"jobID": "job-1", "location": "status"})
    state_store = StateStore(tmp_path / "state.json")
    request = EcdsRequest(state_store, session=session)

    request.submit({"number": [0, 1], "step": [24], "param": ["2t"]})

    resumed_session = Mock()
    resumed_session.headers = {}
    resumed_session.get.return_value = response(
        {"status": "successful", "result": {"href": "result"}}
    )
    resumed = EcdsRequest(state_store, session=resumed_session)
    measurement, result_url = resumed.poll_once()
    assert measurement.request_id == "job-1"
    assert measurement.expected_members == [0, 1]
    assert result_url == "result"
    assert state_store.read().result_url == "result"


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


def test_state_writes_machine_readable_measurements_atomically(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "measurement.json")
    state_store.write(RequestMeasurement("id", {"date": "2026-07-24"}, "now", "status"))
    assert json.loads(state_store.path.read_text())["request_id"] == "id"
    assert not state_store.path.with_suffix(".json.tmp").exists()


def test_download_uses_result_url_saved_by_poll(tmp_path: Path) -> None:
    state_store = StateStore(tmp_path / "measurement.json")
    state_store.write(
        RequestMeasurement(
            "id", {}, "now", "status", result_url="https://example.test/result"
        )
    )
    download_response = response({})
    download_response.iter_content.return_value = [b"GRIB", b"contents", b"7777"]
    session = Mock()
    session.headers = {}
    session.auth = ("user", "key")
    session.get.return_value = download_response

    measurement = EcdsRequest(state_store, session=session).download(
        tmp_path / "result.grib"
    )

    assert measurement.downloaded_bytes == len(b"GRIBcontents7777")
    assert measurement.grib_messages == 1
    assert (tmp_path / "result.grib.complete").exists()


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
