from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.pydantic import replace
from reformatters.ucsb_chc.chirps.analysis_final.region_job import (
    UcsbChcChirpsAnalysisFinalRegionJob,
)
from reformatters.ucsb_chc.chirps.analysis_final.template_config import (
    UcsbChcChirpsAnalysisFinalTemplateConfig,
)
from reformatters.ucsb_chc.chirps.analysis_preliminary.region_job import (
    UcsbChcChirpsAnalysisPreliminaryRegionJob,
)
from reformatters.ucsb_chc.chirps.analysis_preliminary.template_config import (
    UcsbChcChirpsAnalysisPreliminaryTemplateConfig,
)
from reformatters.ucsb_chc.chirps.chirps_config_models import ChirpsProduct
from reformatters.ucsb_chc.chirps.region_job import (
    UcsbChcChirpsAnalysisMaterializedRegionJob,
    UcsbChcChirpsAnalysisSourceFileCoord,
)
from reformatters.ucsb_chc.chirps.template_config import (
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    MM_PER_DAY_TO_KG_M2_S,
    SOURCE_FILL_VALUE,
)


def _job(
    product: ChirpsProduct = "final",
) -> UcsbChcChirpsAnalysisMaterializedRegionJob:
    job_class = (
        UcsbChcChirpsAnalysisFinalRegionJob
        if product == "final"
        else UcsbChcChirpsAnalysisPreliminaryRegionJob
    )
    config = (
        UcsbChcChirpsAnalysisFinalTemplateConfig()
        if product == "final"
        else UcsbChcChirpsAnalysisPreliminaryTemplateConfig()
    )
    return job_class(
        tmp_store=Path("unused.zarr"),
        template_ds=xr.DataTree(xr.Dataset(attrs={"dataset_id": config.dataset_id})),
        data_vars=list(config.data_vars),
        append_dim="time",
        region=slice(0, 1),
        reformat_job_name="test",
    )


def test_variant_region_jobs_carry_product() -> None:
    assert (
        UcsbChcChirpsAnalysisFinalRegionJob.model_fields["product"].default == "final"
    )
    assert (
        UcsbChcChirpsAnalysisPreliminaryRegionJob.model_fields["product"].default
        == "preliminary"
    )


def test_final_url() -> None:
    coord = UcsbChcChirpsAnalysisSourceFileCoord(
        product="final", time=pd.Timestamp("2020-06-15")
    )
    assert coord.get_url() == (
        "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/2020/"
        "chirps-v3.0.rnl.2020.06.15.tif"
    )


def test_preliminary_url() -> None:
    coord = UcsbChcChirpsAnalysisSourceFileCoord(
        product="preliminary", time=pd.Timestamp("2026-08-25")
    )
    assert coord.get_url() == (
        "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/prelim/sat/2026/"
        "chirps-v3.0.prelim.2026.08.25.tif"
    )


def test_download_file_never_falls_back_to_the_other_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []

    def fake_download(url: str, dataset_id: str) -> Path:
        requested.append(url)
        if "/prelim/sat/" in url:
            raise FileNotFoundError(url)
        return Path(url)

    monkeypatch.setattr(
        "reformatters.ucsb_chc.chirps.region_job.http_download_to_disk", fake_download
    )
    coord = UcsbChcChirpsAnalysisSourceFileCoord(
        product="preliminary", time=pd.Timestamp("2025-06-15")
    )

    with pytest.raises(FileNotFoundError):
        _job("preliminary").download_file(coord)

    assert requested == [coord.get_url()]
    assert "/prelim/sat/" in requested[0]


def test_download_file_propagates_missing_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download(url: str, dataset_id: str) -> Path:
        raise FileNotFoundError(url)

    monkeypatch.setattr(
        "reformatters.ucsb_chc.chirps.region_job.http_download_to_disk", fake_download
    )
    coord = UcsbChcChirpsAnalysisSourceFileCoord(
        product="final", time=pd.Timestamp("2030-01-01")
    )

    with pytest.raises(FileNotFoundError):
        _job().download_file(coord)


def test_generate_source_file_coords() -> None:
    times = pd.date_range("2025-01-01", "2025-01-03", freq="1D")
    processing_region_ds = xr.Dataset(coords={"time": times})

    coords = _job("preliminary").generate_source_file_coords(processing_region_ds, [])

    assert [c.time for c in coords] == list(times)
    assert {c.product for c in coords} == {"preliminary"}
    assert coords[0].out_loc() == {"time": pd.Timestamp("2025-01-01")}


def test_read_data_masks_fill_value_and_converts_to_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = np.full((GRID_LAT_SIZE, GRID_LON_SIZE), 36.0, dtype=np.float32)
    raw[0, 0] = np.float32(SOURCE_FILL_VALUE)

    reader = MagicMock()
    reader.read.return_value = raw
    reader.__enter__ = lambda self: self
    reader.__exit__ = lambda self, *args: None
    monkeypatch.setattr("rasterio.open", lambda _path: reader)

    job = _job()
    (precip,) = job.data_vars
    coord = replace(
        UcsbChcChirpsAnalysisSourceFileCoord(
            product="final", time=pd.Timestamp("1981-01-01")
        ),
        downloaded_path=Path("chirps.tif"),
    )

    data = job.read_data(coord, precip)

    assert data.shape == (GRID_LAT_SIZE, GRID_LON_SIZE)
    assert np.isnan(data[0, 0])
    np.testing.assert_allclose(data[0, 1], 36.0 * MM_PER_DAY_TO_KG_M2_S, rtol=1e-6)
    assert np.isfinite(data).mean() > 0.999
