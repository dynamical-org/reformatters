from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.config import Config, Env
from reformatters.common.pydantic import replace
from reformatters.common.types import ArrayFloat32
from reformatters.nasa.imerg.analysis_early_v7.region_job import (
    NasaImergAnalysisEarlyV7RegionJob,
)
from reformatters.nasa.imerg.analysis_early_v7.template_config import (
    NasaImergAnalysisEarlyV7TemplateConfig,
)
from reformatters.nasa.imerg.region_job import (
    NasaImergSourceFileCoord,
    _candidate_versions,
    _reorient_imerg_array,
)


def test_get_url_early() -> None:
    coord = NasaImergSourceFileCoord(time=pd.Timestamp("2026-05-26T00:00"), run="early")
    assert coord.get_url("V07C") == (
        "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHHE.07/"
        "2026/146/3B-HHR-E.MS.MRG.3IMERG.20260526-S000000-E002959.0000.V07C.HDF5"
    )


def test_get_url_uses_run_code_and_minutes_of_day() -> None:
    coord = NasaImergSourceFileCoord(time=pd.Timestamp("2024-01-15T00:30"), run="early")
    assert coord.get_url("V07B") == (
        "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHHE.07/"
        "2024/015/3B-HHR-E.MS.MRG.3IMERG.20240115-S003000-E005959.0030.V07B.HDF5"
    )


def test_candidate_versions() -> None:
    assert _candidate_versions(pd.Timestamp("2026-03-04")) == ("V07C", "V07B")
    assert _candidate_versions(pd.Timestamp("2026-03-03T23:30")) == ("V07B", "V07C")


def test_out_loc() -> None:
    coord = NasaImergSourceFileCoord(time=pd.Timestamp("2024-01-15T00:30"), run="early")
    assert coord.out_loc() == {"time": pd.Timestamp("2024-01-15T00:30")}


def test_reorient_imerg_array() -> None:
    # raw is (longitude ascending, latitude ascending south->north).
    # rows index longitude (lon0, lon1, lon2), columns index latitude (south, north).
    raw = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32
    )  # shape (lon=3, lat=2)
    result = _reorient_imerg_array(raw)
    # -> (latitude descending north->south, longitude ascending)
    expected = np.array([[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, 3)


def _region_job(tmp_path: Path) -> NasaImergAnalysisEarlyV7RegionJob:
    mock_ds = Mock()
    mock_ds.attrs = {"dataset_id": "nasa-imerg-analysis-early-v7"}
    return NasaImergAnalysisEarlyV7RegionJob.model_construct(
        tmp_store=tmp_path,
        template_ds=mock_ds,
        data_vars=NasaImergAnalysisEarlyV7TemplateConfig().data_vars,
        append_dim="time",
        region=slice(0, 1),
        reformat_job_name="test",
    )


def _mock_rasterio_open(monkeypatch: pytest.MonkeyPatch, raw: ArrayFloat32) -> None:
    mock_dataset = Mock()
    mock_dataset.read.return_value = raw
    mock_open = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_dataset
    monkeypatch.setattr("reformatters.nasa.imerg.region_job.rasterio.open", mock_open)


def test_read_data_precipitation_orientation_fill_and_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = NasaImergAnalysisEarlyV7TemplateConfig()
    precip = next(v for v in config.data_vars if v.name == "precipitation_surface")

    # Source array shape (lon=3600, lat=1800).
    raw = np.zeros((3600, 1800), dtype=np.float32)
    raw[100, 200] = 36.0  # 36 mm/hr -> 0.01 kg m-2 s-1 after scale
    raw[0, 1799] = -9999.9  # float fill -> NaN

    _mock_rasterio_open(monkeypatch, raw)

    coord = NasaImergSourceFileCoord(
        time=pd.Timestamp("2024-01-15T00:00"),
        run="early",
        downloaded_path=tmp_path / "fake.HDF5",
    )
    data = _region_job(tmp_path).read_data(coord, precip)

    assert data.shape == (1800, 3600)
    assert data.dtype == np.float32
    # raw[100, 200] -> data[1799 - 200, 100] = data[1599, 100], scaled by 1/3600.
    assert np.isclose(data[1599, 100], 36.0 / 3600.0)
    # raw[0, 1799] (lat north, lon west) -> data[0, 0]
    assert np.isnan(data[0, 0])
    # Everything else is zero (not fill, not scaled away to nonzero).
    assert np.nanmin(data) == 0.0


def test_read_data_probability_int_fill_no_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = NasaImergAnalysisEarlyV7TemplateConfig()
    prob = next(
        v
        for v in config.data_vars
        if v.name == "probability_of_liquid_precipitation_surface"
    )

    raw = np.zeros((3600, 1800), dtype=np.float32)
    raw[10, 20] = 55.0  # percent, no scaling
    raw[5, 5] = -9999.0  # int16 fill read as float -> NaN

    _mock_rasterio_open(monkeypatch, raw)

    coord = NasaImergSourceFileCoord(
        time=pd.Timestamp("2024-01-15T00:00"),
        run="early",
        downloaded_path=tmp_path / "fake.HDF5",
    )
    data = _region_job(tmp_path).read_data(coord, prob)

    assert data.shape == (1800, 3600)
    assert np.isclose(data[1799 - 20, 10], 55.0)  # unscaled
    assert np.isnan(data[1799 - 5, 5])


def test_generate_source_file_coords(tmp_path: Path) -> None:
    times = pd.date_range("2024-01-15T00:00", periods=3, freq="30min")
    processing_region_ds = xr.Dataset(coords={"time": times})
    region_job = _region_job(tmp_path)

    coords = region_job.generate_source_file_coords(
        processing_region_ds, region_job.data_vars
    )
    assert len(coords) == 3
    assert all(c.run == "early" for c in coords)
    assert [c.time for c in coords] == list(times)


def test_download_file_falls_back_to_other_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    region_job = _region_job(tmp_path)
    # >= 2026-03-04 so the primary candidate is V07C, fallback V07B.
    coord = NasaImergSourceFileCoord(time=pd.Timestamp("2026-05-26T00:00"), run="early")

    mock_404 = Mock(status_code=404)
    mock_ok = Mock(status_code=200)
    mock_ok.raise_for_status = Mock()
    mock_ok.iter_content = Mock(return_value=[b"imerg", b"-bytes"])
    mock_session = Mock()
    mock_session.get = Mock(side_effect=[mock_404, mock_ok])
    monkeypatch.setattr(
        "reformatters.nasa.imerg.region_job.get_authenticated_session",
        lambda: mock_session,
    )

    result = region_job.download_file(coord)

    assert mock_session.get.call_count == 2
    urls = [call.args[0] for call in mock_session.get.call_args_list]
    assert "V07C" in urls[0]
    assert "V07B" in urls[1]
    assert result.read_bytes() == b"imerg-bytes"


def test_operational_update_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template_config = NasaImergAnalysisEarlyV7TemplateConfig()

    existing_end = pd.Timestamp("2024-01-15T05:00")
    existing_ds = template_config.get_template(existing_end + pd.Timedelta("1s"))

    mock_store = Mock()
    original_open_zarr = xr.open_zarr

    def open_zarr(store: Store, decode_timedelta: bool = True) -> xr.Dataset:
        if store is mock_store:
            return existing_ds
        result = original_open_zarr(
            template_config.template_path(), decode_timedelta=decode_timedelta
        )
        assert isinstance(result, xr.Dataset)
        return result

    monkeypatch.setattr(xr, "open_zarr", open_zarr)
    monkeypatch.setattr(
        "pandas.Timestamp.now", lambda tz=None: pd.Timestamp("2024-01-15T06:34")
    )

    jobs, template_ds = NasaImergAnalysisEarlyV7RegionJob.operational_update_jobs(
        primary_store=mock_store,
        tmp_store=tmp_path,
        get_template_fn=template_config.get_template,
        append_dim=template_config.append_dim,
        all_data_vars=template_config.data_vars,
        reformat_job_name="test-update",
    )

    assert len(jobs) >= 1
    assert template_ds["time"].max().values >= np.datetime64("2024-01-15T06:00")
    for job in jobs:
        assert isinstance(job, NasaImergAnalysisEarlyV7RegionJob)
        assert job.run == "early"


@pytest.mark.slow
def test_download_and_read_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Download a real IMERG Early granule from GES DISC and read each variable.

    Requires NASA Earthdata credentials (the nasa-earthdata secret, available via a
    local kubectl context or a mounted secret in prod).
    """
    monkeypatch.setattr(Config, "env", Env.prod)

    config = NasaImergAnalysisEarlyV7TemplateConfig()
    region_job = _region_job(tmp_path)
    coord = NasaImergSourceFileCoord(time=pd.Timestamp("2024-01-15T00:00"), run="early")
    coord = replace(coord, downloaded_path=region_job.download_file(coord))

    by_name = {v.name: v for v in config.data_vars}

    precip = region_job.read_data(coord, by_name["precipitation_surface"])
    assert precip.shape == (1800, 3600)
    precip_valid = precip[~np.isnan(precip)]
    assert precip_valid.size > 0
    assert precip_valid.min() >= 0.0

    prob = region_job.read_data(
        coord, by_name["probability_of_liquid_precipitation_surface"]
    )
    prob_valid = prob[~np.isnan(prob)]
    assert prob_valid.size > 0
    assert prob_valid.min() >= 0.0
    assert prob_valid.max() <= 100.0

    quality = region_job.read_data(
        coord, by_name["precipitation_quality_index_surface"]
    )
    quality_valid = quality[~np.isnan(quality)]
    assert quality_valid.size > 0
    assert quality_valid.min() >= 0.0
    assert quality_valid.max() <= 1.0
