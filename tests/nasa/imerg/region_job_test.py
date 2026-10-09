import logging
from pathlib import Path
from unittest.mock import MagicMock, call

import numpy as np
import pandas as pd
import pytest
import requests
import xarray as xr

from reformatters.common.pydantic import replace
from reformatters.nasa.imerg.analysis_early.region_job import (
    NasaImergAnalysisEarlyRegionJob,
)
from reformatters.nasa.imerg.analysis_early.template_config import (
    NasaImergAnalysisEarlyTemplateConfig,
)
from reformatters.nasa.imerg.analysis_late.region_job import (
    NasaImergAnalysisLateRegionJob,
)
from reformatters.nasa.imerg.region_job import (
    _HDF5_MAGIC,
    _JSIMPSON_MAX_AGE,
    DownloadSource,
    NasaImergAnalysisSourceFileCoord,
)
from reformatters.nasa.imerg.template_config import (
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    MM_PER_HR_TO_KG_M2_S,
    SOURCE_FILL_VALUE,
)


def test_variant_region_jobs_carry_run() -> None:
    assert NasaImergAnalysisEarlyRegionJob.model_fields["run"].default == "early"
    assert NasaImergAnalysisLateRegionJob.model_fields["run"].default == "late"


def test_version_computed_from_time() -> None:
    # The V07B->V07C switchover time differs per run.
    early_before = NasaImergAnalysisSourceFileCoord(
        run="early", time=pd.Timestamp("2026-03-03T23:30")
    )
    early_on = NasaImergAnalysisSourceFileCoord(
        run="early", time=pd.Timestamp("2026-03-04T00:00")
    )
    assert early_before.version == "V07B"
    assert early_on.version == "V07C"

    late_before = NasaImergAnalysisSourceFileCoord(
        run="late", time=pd.Timestamp("2026-03-03T13:30")
    )
    late_on = NasaImergAnalysisSourceFileCoord(
        run="late", time=pd.Timestamp("2026-03-03T14:00")
    )
    assert late_before.version == "V07B"
    assert late_on.version == "V07C"


def test_gesdisc_url_early_v07c() -> None:
    coord = NasaImergAnalysisSourceFileCoord(
        run="early", time=pd.Timestamp("2026-05-26T00:00")
    )
    assert coord.get_url("gesdisc") == (
        "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHHE.07/2026/146/"
        "3B-HHR-E.MS.MRG.3IMERG.20260526-S000000-E002959.0000.V07C.HDF5"
    )


def test_gesdisc_url_late_v07b_day_of_year() -> None:
    coord = NasaImergAnalysisSourceFileCoord(
        run="late", time=pd.Timestamp("2024-01-15T00:00")
    )
    assert coord.get_url("gesdisc") == (
        "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHHL.07/2024/015/"
        "3B-HHR-L.MS.MRG.3IMERG.20240115-S000000-E002959.0000.V07B.HDF5"
    )


def test_jsimpson_url_and_half_hour_fields() -> None:
    coord = NasaImergAnalysisSourceFileCoord(
        run="early", time=pd.Timestamp("2026-05-26T12:30")
    )
    # 12:30 -> S123000-E125959, minutes-into-day 12*60+30 = 750
    assert coord.get_url("jsimpson") == (
        "https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/early/202605/"
        "3B-HHR-E.MS.MRG.3IMERG.20260526-S123000-E125959.0750.V07C.RT-H5"
    )


def test_get_url_version_override() -> None:
    coord = NasaImergAnalysisSourceFileCoord(
        run="early", time=pd.Timestamp("2026-05-26T00:00")
    )
    assert ".V07B.HDF5" in coord.get_url("gesdisc", version="V07B")


def test_candidate_urls_recent_prefers_jsimpson() -> None:
    time = pd.Timestamp.now().floor("30min") - pd.Timedelta(hours=6)
    candidates = NasaImergAnalysisSourceFileCoord(
        run="early", time=time
    ).candidate_urls()
    sources = [source for source, _ in candidates]
    # Recent granule: jsimpson first, GES DISC fallback; each with both versions.
    assert sources == ["jsimpson", "jsimpson", "gesdisc", "gesdisc"]


def test_candidate_urls_old_prefers_gesdisc() -> None:
    time = pd.Timestamp("2001-06-15T00:00")
    assert time < pd.Timestamp.now() - _JSIMPSON_MAX_AGE
    candidates = NasaImergAnalysisSourceFileCoord(
        run="late", time=time
    ).candidate_urls()
    sources = [source for source, _ in candidates]
    # Old granule: GES DISC only (both versions); jsimpson's rolling window can't hold
    # data this old, so it is never tried.
    assert sources == ["gesdisc", "gesdisc"]


def _job() -> NasaImergAnalysisEarlyRegionJob:
    return NasaImergAnalysisEarlyRegionJob(
        tmp_store=Path("unused.zarr"),
        template_ds=xr.DataTree(
            xr.Dataset(attrs={"dataset_id": "nasa-imerg-analysis-early"})
        ),
        data_vars=list(NasaImergAnalysisEarlyTemplateConfig().data_vars),
        append_dim="time",
        region=slice(0, 1),
        reformat_job_name="test",
    )


def _fake_response(
    body: bytes, *, status: int = 200, content_type: str = "application/x-hdf5"
) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.headers = {"Content-Type": content_type}
    response.raise_for_status.return_value = None
    response.iter_content.return_value = [body]
    return response


def _recent_coord() -> NasaImergAnalysisSourceFileCoord:
    return NasaImergAnalysisSourceFileCoord(
        run="early", time=pd.Timestamp.now().floor("30min") - pd.Timedelta(hours=6)
    )


def _patch_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    pps: MagicMock,
    earthdata: MagicMock,
) -> None:
    monkeypatch.setattr("reformatters.common.retry.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("reformatters.common.download.DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(
        "reformatters.nasa.imerg.region_job.get_pps_session", lambda: pps
    )
    monkeypatch.setattr(
        "reformatters.nasa.imerg.region_job.get_earthdata_session", lambda: earthdata
    )


def test_download_file_falls_through_connection_error_to_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # jsimpson resets the connection for a not-yet-published granule; the GES DISC
    # archive candidate must still be tried rather than the whole attempt aborting.
    jsimpson = MagicMock()
    jsimpson.get.side_effect = requests.ConnectionError("Connection reset by peer")
    gesdisc = MagicMock()
    gesdisc.get.return_value = _fake_response(_HDF5_MAGIC + b"granule-bytes")
    _patch_sessions(monkeypatch, tmp_path, pps=jsimpson, earthdata=gesdisc)

    path = _job().download_file(_recent_coord())

    assert path.read_bytes().startswith(_HDF5_MAGIC)
    assert jsimpson.get.call_count == 2
    assert gesdisc.get.call_count == 1
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "outcome=recovered last_source=gesdisc" in caplog.text
    assert "'connection_error': 2" in caplog.text


def test_download_file_summarizes_mixed_failures_on_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    coord = _recent_coord()
    candidates = coord.candidate_urls()
    session = MagicMock()
    session.get.side_effect = [
        _fake_response(b"", status=404, content_type="text/plain"),
        requests.ConnectionError("Connection reset by peer"),
        _fake_response(
            b"<html>authentication or upstream failure</html>", content_type="text/html"
        ),
        _fake_response(_HDF5_MAGIC + b"granule-bytes"),
    ]
    _patch_sessions(monkeypatch, tmp_path, pps=session, earthdata=session)

    path = _job().download_file(coord)

    assert path.read_bytes() == _HDF5_MAGIC + b"granule-bytes"
    assert session.get.call_args_list == [
        call(url, timeout=30, stream=True, allow_redirects=True)
        for _, url in candidates
    ]
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert "outcome=recovered last_source=gesdisc" in record.getMessage()
    assert record.args == (
        coord.run,
        coord.time,
        "recovered",
        "gesdisc",
        {"not_found": 1, "connection_error": 1, "non_hdf5": 1},
        {
            "not_found": "source=jsimpson status=404 content_type='text/plain'",
            "connection_error": "source=jsimpson",
            "non_hdf5": "source=gesdisc status=200 content_type='text/html'",
        },
        [candidates[0][1], candidates[2][1]],
    )
    assert "authentication or upstream failure" not in caplog.text


def test_download_file_first_success_has_no_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = MagicMock()
    session.get.return_value = _fake_response(_HDF5_MAGIC)
    _patch_sessions(monkeypatch, tmp_path, pps=session, earthdata=session)

    assert _job().download_file(_recent_coord()).read_bytes() == _HDF5_MAGIC
    assert session.get.call_count == 1
    assert caplog.records == []


@pytest.mark.parametrize(
    "category", ["not_found", "connection_error", "non_hdf5", "mixed"]
)
def test_download_file_exhaustion_summarizes_each_sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    category: str,
) -> None:
    coord = _recent_coord()
    session = MagicMock()
    not_found = _fake_response(b"", status=404)
    non_hdf5 = _fake_response(b"<html>error</html>", content_type="text/html")
    connection_error = requests.ConnectionError("Connection reset by peer")
    failures = {
        "not_found": [not_found] * 4,
        "connection_error": [connection_error] * 4,
        "non_hdf5": [non_hdf5] * 4,
        "mixed": [not_found, connection_error, non_hdf5, not_found],
    }
    session.get.side_effect = failures[category] * 6
    _patch_sessions(monkeypatch, tmp_path, pps=session, earthdata=session)
    sleeps: list[float] = []
    monkeypatch.setattr("reformatters.common.retry.time.sleep", sleeps.append)
    monkeypatch.setattr("reformatters.common.retry.exponential_backoff_time", float)

    with pytest.raises(FileNotFoundError) as raised:
        _job().download_file(coord)

    assert str(raised.value) == f"No IMERG granule found for {coord.run} {coord.time}"
    assert (
        session.get.call_args_list
        == [
            call(url, timeout=30, stream=True, allow_redirects=True)
            for _, url in coord.candidate_urls()
        ]
        * 6
    )
    assert sleeps == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert len(caplog.records) == 6
    counts = {"not_found": 0, "connection_error": 0, "non_hdf5": 0}
    if category == "mixed":
        counts.update(not_found=2, connection_error=1, non_hdf5=1)
    else:
        counts[category] = 4
    for record in caplog.records:
        assert record.levelno == logging.WARNING
        assert isinstance(record.args, tuple)
        assert record.args[2:5] == ("exhausted", "gesdisc", counts)
        assert "outcome=exhausted last_source=gesdisc" in record.getMessage()


@pytest.mark.parametrize("earlier_failure", [True, False])
@pytest.mark.parametrize("failure_location", ["request", "status", "stream", "session"])
def test_download_file_preserves_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    earlier_failure: bool,
    failure_location: str,
) -> None:
    error = (
        requests.HTTPError("403 Forbidden")
        if failure_location == "status"
        else OSError("unexpected failure")
    )
    session = MagicMock()
    response = _fake_response(
        _HDF5_MAGIC, status=403 if failure_location == "status" else 200
    )
    if failure_location == "status":
        response.raise_for_status.side_effect = error
    if failure_location == "stream":
        response.iter_content.side_effect = error
    failure = error if failure_location == "request" else response
    session.get.side_effect = (
        [_fake_response(b"", status=404), failure] if earlier_failure else [failure]
    ) * 6
    _patch_sessions(monkeypatch, tmp_path, pps=session, earthdata=session)
    if failure_location == "session":
        session.get.side_effect = None
        session.get.return_value = _fake_response(b"", status=404)
        factory = MagicMock(
            side_effect=([session, error] if earlier_failure else [error]) * 6
        )
        monkeypatch.setattr(
            "reformatters.nasa.imerg.region_job.get_pps_session", factory
        )

    with pytest.raises(type(error)) as raised:
        _job().download_file(_recent_coord())

    assert raised.value is error
    requests_per_sweep = int(earlier_failure) + int(failure_location != "session")
    assert session.get.call_count == requests_per_sweep * 6
    assert len(caplog.records) == (6 if earlier_failure else 0)
    for record in caplog.records:
        assert isinstance(record.args, tuple)
        assert record.args[2] == f"unexpected_exception:{type(error).__name__}"
        assert (
            f"outcome=unexpected_exception:{type(error).__name__} last_source=jsimpson"
            in record.getMessage()
        )
        assert "'not_found': 1" in record.getMessage()


@pytest.mark.parametrize("candidate_count", [40, 4000])
def test_download_file_summary_state_stays_bounded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    candidate_count: int,
) -> None:
    candidates: list[tuple[DownloadSource, str]] = [
        ("gesdisc", f"https://example.org/{index}/" + "x" * 2000)
        for index in range(candidate_count)
    ]
    monkeypatch.setattr(
        NasaImergAnalysisSourceFileCoord, "candidate_urls", lambda self: candidates
    )
    session = MagicMock()
    session.get.return_value = _fake_response(b"", status=404, content_type="x" * 10000)
    _patch_sessions(monkeypatch, tmp_path, pps=session, earthdata=session)

    with pytest.raises(FileNotFoundError):
        _job().download_file(_recent_coord())

    assert session.get.call_count == candidate_count * 6
    assert len(caplog.records) == 6
    for record in caplog.records:
        assert isinstance(record.args, tuple)
        assert record.args[4] == {
            "not_found": candidate_count,
            "connection_error": 0,
            "non_hdf5": 0,
        }
        assert record.args[5] == {
            "not_found": f"source=gesdisc status=404 content_type={'x' * 128!r}",
            "connection_error": "",
            "non_hdf5": "",
        }
        assert record.args[6] == [candidates[0][1][:512], candidates[-1][1][:512]]
        assert len(record.getMessage()) < 2000


def test_download_file_rejects_non_hdf5_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A successful HTTP response can contain an error or authentication body;
    # a non-HDF5 body must never be handed to the reader as a granule.
    session = MagicMock()
    session.get.return_value = _fake_response(b"<html>not found</html>")
    _patch_sessions(monkeypatch, tmp_path, pps=session, earthdata=session)

    with pytest.raises(FileNotFoundError):
        _job().download_file(_recent_coord())


def test_download_file_all_sources_connection_error_raises_file_not_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # When every candidate fails to connect, the exhausted loop reports a missing
    # file, which the base job's recency gate treats as expected for recent granules.
    session = MagicMock()
    session.get.side_effect = requests.ConnectionError("Connection reset by peer")
    _patch_sessions(monkeypatch, tmp_path, pps=session, earthdata=session)

    with pytest.raises(FileNotFoundError):
        _job().download_file(_recent_coord())


def test_read_data_masks_exact_sentinel_and_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Source band is (lon, lat). One cell is the fill sentinel, the rest is 36 mm/hr.
    raw = np.full((GRID_LON_SIZE, GRID_LAT_SIZE), 36.0, dtype=np.float32)
    raw[0, 0] = np.float32(SOURCE_FILL_VALUE)

    reader = MagicMock()
    reader.read.return_value = raw
    reader.__enter__ = lambda self: self
    reader.__exit__ = lambda self, *args: None
    monkeypatch.setattr("rasterio.open", lambda _path: reader)

    job = _job()
    precip = next(v for v in job.data_vars if v.name == "precipitation_surface")
    coord = replace(
        NasaImergAnalysisSourceFileCoord(run="early", time=pd.Timestamp("2001-06-15")),
        downloaded_path=Path("granule.HDF5"),
    )
    data = job.read_data(coord, precip)

    assert data.shape == (GRID_LAT_SIZE, GRID_LON_SIZE)
    # The sentinel cell (source lon=0, lat=0 -> south pole after flip) becomes NaN.
    assert np.isnan(data[-1, 0])
    # A valid cell is scaled mm/hr -> kg m-2 s-1 and never spuriously masked.
    np.testing.assert_allclose(data[0, 0], 36.0 * MM_PER_HR_TO_KG_M2_S, rtol=1e-6)
    assert np.isfinite(data).mean() > 0.999
