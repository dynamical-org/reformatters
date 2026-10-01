from collections.abc import Callable, Sequence
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import requests
import xarray as xr
from requests.adapters import HTTPAdapter

from reformatters.common.region_job import SourceFileCoord, SourceFileStatus
from reformatters.contrib.nasa.smap.level3_36km_v9.region_job import (
    NasaSmapLevel336KmV9RegionJob,
    NasaSmapLevel336KmV9SourceFileCoord,
)
from reformatters.contrib.nasa.smap.level3_36km_v9.template_config import (
    NasaSmapLevel336KmV9TemplateConfig,
)
from reformatters.nasa import nasa_auth
from reformatters.nasa.imerg.analysis_early.region_job import (
    NasaImergAnalysisEarlyRegionJob,
)
from reformatters.nasa.imerg.analysis_early.template_config import (
    NasaImergAnalysisEarlyTemplateConfig,
)
from reformatters.nasa.imerg.region_job import (
    _HDF5_MAGIC,
    NasaImergAnalysisSourceFileCoord,
)

_GRANULE = _HDF5_MAGIC + b"granule"
_OAUTH_URL = "https://urs.earthdata.nasa.gov/oauth/authorize"


def _response(status: int, body: bytes = b"") -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.raw = BytesIO(body)
    return response


def _token_response(value: str = "test-token") -> requests.Response:
    return _response(200, f'{{"access_token": "{value}"}}'.encode())


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Mock:
    monkeypatch.setattr(nasa_auth, "_thread_local", nasa_auth._ThreadLocalStorage())
    monkeypatch.setattr(
        nasa_auth,
        "load_secret",
        Mock(return_value={"username": "test-user", "password": "test-password"}),
    )
    monkeypatch.setattr("reformatters.common.retry.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("reformatters.common.download.DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr("requests.sessions.get_netrc_auth", lambda url: None)
    responses = Mock()

    def send(
        self: HTTPAdapter, request: requests.PreparedRequest, **kwargs: object
    ) -> requests.Response:
        response = responses(request, **kwargs)
        response.request = request
        response.url = request.url
        return response

    monkeypatch.setattr(HTTPAdapter, "send", send)
    return responses


@pytest.fixture(params=["imerg", "smap"])
def download_group(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[Callable[[], Sequence[SourceFileCoord]], int]:
    template_ds = xr.DataTree(xr.Dataset(attrs={"dataset_id": f"test-{request.param}"}))
    time = pd.Timestamp("2015-04-01")
    if request.param == "imerg":
        imerg_job = NasaImergAnalysisEarlyRegionJob(
            tmp_store=tmp_path,
            template_ds=template_ds,
            data_vars=list(NasaImergAnalysisEarlyTemplateConfig().data_vars),
            append_dim="time",
            region=slice(0, 1),
            reformat_job_name="test",
            download_parallelism=1,
        )
        imerg_coord = NasaImergAnalysisSourceFileCoord(run="early", time=time)
        return lambda: imerg_job._download_processing_group([imerg_coord], []), 6
    smap_job = NasaSmapLevel336KmV9RegionJob(
        tmp_store=tmp_path,
        template_ds=template_ds,
        data_vars=NasaSmapLevel336KmV9TemplateConfig().data_vars[:1],
        append_dim="time",
        region=slice(0, 1),
        reformat_job_name="test",
        download_parallelism=1,
    )
    smap_coord = NasaSmapLevel336KmV9SourceFileCoord(time=time)
    return lambda: smap_job._download_processing_group([smap_coord], []), 10


def test_unauthorized_response_evicts_and_closes_session(transport: Mock) -> None:
    unauthorized = _response(401)
    transport.side_effect = [_token_response(), unauthorized, _token_response()]
    session = nasa_auth.get_earthdata_session()
    with pytest.MonkeyPatch.context() as monkeypatch:
        close_session = Mock(wraps=session.close)
        monkeypatch.setattr(session, "close", close_session)
        response = session.get(_OAUTH_URL, stream=True)

    assert response is unauthorized
    assert response.raw.closed
    close_session.assert_called_once()
    assert not hasattr(nasa_auth._thread_local, "earthdata_session")
    with pytest.raises(requests.HTTPError):
        response.raise_for_status()
    assert nasa_auth.get_earthdata_session() is not session


@pytest.mark.parametrize("status", [200, 404, 403])
def test_other_responses_keep_session(transport: Mock, status: int) -> None:
    transport.side_effect = [_token_response(), _response(status)]
    session = nasa_auth.get_earthdata_session()
    response = session.get(_OAUTH_URL, stream=True)
    assert response.status_code == status
    assert not response.raw.closed
    assert nasa_auth.get_earthdata_session() is session
    assert transport.call_count == 2


def test_stale_hook_does_not_evict_newer_session(transport: Mock) -> None:
    transport.side_effect = [_token_response(), _token_response(), _response(401)]
    stale = nasa_auth.get_earthdata_session()
    newer = nasa_auth._create_earthdata_session()
    nasa_auth._thread_local.earthdata_session = newer

    with pytest.MonkeyPatch.context() as monkeypatch:
        close_newer = Mock(wraps=newer.close)
        monkeypatch.setattr(newer, "close", close_newer)
        response = stale.get(_OAUTH_URL, stream=True)

    assert response.raw.closed
    close_newer.assert_not_called()
    assert nasa_auth.get_earthdata_session() is newer


def test_pps_unauthorized_keeps_session(transport: Mock) -> None:
    transport.return_value = _response(401)
    session = nasa_auth.get_pps_session()
    response = session.get(
        "https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg", stream=True
    )
    assert session.hooks["response"] == []
    assert not response.raw.closed
    assert nasa_auth.get_pps_session() is session
    assert not hasattr(nasa_auth._thread_local, "earthdata_session")
    with pytest.raises(requests.HTTPError):
        response.raise_for_status()


def test_download_recovers_from_unauthorized_oauth_redirect(
    transport: Mock,
    download_group: tuple[Callable[[], Sequence[SourceFileCoord]], int],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    redirect = _response(302)
    redirect.headers["Location"] = _OAUTH_URL
    unauthorized = _response(401)
    transport.side_effect = [
        _token_response("old-token"),
        redirect,
        unauthorized,
        _token_response("new-token"),
        _response(200, _GRANULE),
    ]

    download, _ = download_group
    (coord,) = download()

    assert coord.downloaded_path is not None
    assert coord.downloaded_path.read_bytes() == _GRANULE
    assert unauthorized.raw.closed
    requests_sent = [call.args[0] for call in transport.call_args_list]
    assert [request.method for request in requests_sent] == [
        "POST",
        "GET",
        "GET",
        "POST",
        "GET",
    ]
    assert requests_sent[2].url == _OAUTH_URL
    assert requests_sent[1].url == requests_sent[4].url
    assert requests_sent[1].headers["Authorization"] == "Bearer old-token"
    assert "Authorization" not in requests_sent[2].headers
    assert requests_sent[4].headers["Authorization"] == "Bearer new-token"
    assert not [record for record in caplog.records if record.levelname == "ERROR"]
    assert (
        caplog.messages.count(
            "Earthdata session rejected (401); discarding session before retry"
        )
        == 1
    )


def test_download_404_fallback_keeps_session(
    transport: Mock, download_group: tuple[Callable[[], Sequence[SourceFileCoord]], int]
) -> None:
    transport.side_effect = [
        _token_response(),
        _response(404),
        _response(200, _GRANULE),
    ]
    download, _ = download_group
    (coord,) = download()
    assert coord.downloaded_path is not None
    assert coord.downloaded_path.read_bytes() == _GRANULE
    requests_sent = [call.args[0] for call in transport.call_args_list]
    assert [request.method for request in requests_sent] == ["POST", "GET", "GET"]
    assert requests_sent[1].url != requests_sent[2].url


@pytest.mark.parametrize("failure", ["unauthorized", "refresh"])
def test_download_exhausts_retries_and_logs_failure(
    transport: Mock,
    download_group: tuple[Callable[[], Sequence[SourceFileCoord]], int],
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    download, attempts = download_group
    if failure == "unauthorized":
        transport.side_effect = [
            response
            for _ in range(attempts)
            for response in (_token_response(), _response(401))
        ]
        expected_requests = attempts * 2
    else:
        transport.side_effect = [
            _token_response(),
            _response(401),
            *[_response(401) for _ in range((attempts - 1) * 6)],
        ]
        expected_requests = 2 + (attempts - 1) * 6

    (coord,) = download()

    assert coord.status == SourceFileStatus.DownloadFailed
    assert coord.downloaded_path is None
    assert transport.call_count == expected_requests
    (error,) = [record for record in caplog.records if record.levelname == "ERROR"]
    assert error.message == f"Download failed {coord.get_url()}"
    assert error.exc_info is not None
    exception = error.exc_info[1]
    if failure == "unauthorized":
        assert isinstance(exception, requests.HTTPError)
        assert exception.response is not None
        assert exception.response.status_code == 401
    else:
        assert isinstance(exception, RuntimeError)
        assert "Failed to get token from NASA Earthdata: 401" in str(exception)
