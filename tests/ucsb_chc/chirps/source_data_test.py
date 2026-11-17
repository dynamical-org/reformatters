from pathlib import Path
from unittest.mock import Mock

import httpx
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr

from reformatters.common.download import RateLimiter
from reformatters.ucsb_chc.chirps import region_job
from reformatters.ucsb_chc.chirps.analysis_final.region_job import (
    UcsbChcChirpsAnalysisFinalRegionJob,
)
from reformatters.ucsb_chc.chirps.analysis_final.template_config import (
    UcsbChcChirpsAnalysisFinalTemplateConfig,
)
from reformatters.ucsb_chc.chirps.analysis_preliminary.template_config import (
    UcsbChcChirpsAnalysisPreliminaryTemplateConfig,
)
from reformatters.ucsb_chc.chirps.chirps_config_models import ChirpsProduct
from reformatters.ucsb_chc.chirps.region_job import (
    UcsbChcChirpsAnalysisSourceFileCoord,
)
from reformatters.ucsb_chc.chirps.template_config import (
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    SOURCE_FILL_VALUE,
)
from tests.ucsb_chc.chirps.refresh_fixtures import fixture_path
from tests.ucsb_chc.chirps.region_job_test import _job

# These tests run against real CHIRPS responses checked in under fixtures/, fetched
# on 2026-11-17 from data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/ at the paths in
# refresh_fixtures.FIXTURE_PATHS; that module re-fetches them and documents how the
# GeoTIFFs are reduced.


@pytest.fixture(autouse=True)
def serve_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    def get(url: str, *, rate_limiter: RateLimiter) -> httpx.Response:
        return httpx.Response(
            200,
            content=fixture_path(url).read_bytes(),
            request=httpx.Request("GET", url),
        )

    def download(url: str, dataset_id: str, *, rate_limiter: RateLimiter) -> Path:
        return fixture_path(url)

    monkeypatch.setattr(region_job, "httpx_get", get)
    monkeypatch.setattr(region_job, "httpx_download_to_disk", download)


@pytest.mark.parametrize("product", ["final", "preliminary"])
def test_directory_listing(product: ChirpsProduct) -> None:
    times = pd.date_range("2025-01-01", periods=3)
    coords = _job(product).generate_source_file_coords(
        xr.Dataset(coords={"time": times}), []
    )
    assert [coord.time for coord in coords] == list(times)


@pytest.fixture
def example_source_file_paths() -> dict[ChirpsProduct, Path]:
    config = UcsbChcChirpsAnalysisFinalTemplateConfig()
    job = UcsbChcChirpsAnalysisFinalRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=config.get_template(config.append_dim_start),
        data_vars=list(config.data_vars),
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )
    return {
        product: job.download_file(
            UcsbChcChirpsAnalysisSourceFileCoord(product=product, time=time)
        )
        for product, time in (
            ("final", UcsbChcChirpsAnalysisFinalTemplateConfig().append_dim_start),
            (
                "preliminary",
                UcsbChcChirpsAnalysisPreliminaryTemplateConfig().append_dim_start,
            ),
        )
    }


def test_grid_matches_source_file(
    example_source_file_paths: dict[ChirpsProduct, Path],
) -> None:
    coords = UcsbChcChirpsAnalysisFinalTemplateConfig().dimension_coordinates()

    with rasterio.open(example_source_file_paths["final"]) as reader:
        bounds = reader.bounds
        pixel_size_x = reader.transform.a
        pixel_size_y = abs(reader.transform.e)
        assert reader.shape == (GRID_LAT_SIZE, GRID_LON_SIZE)
        assert reader.crs.to_epsg() == 4326
        # Columns run west -> east and rows north -> south, matching the ascending
        # longitude and descending latitude coordinates.
        assert reader.transform.a == pytest.approx(0.05, abs=1e-6)
        assert reader.transform.e == pytest.approx(-0.05, abs=1e-6)

    lat = coords["latitude"]
    lon = coords["longitude"]
    atol = 1e-5
    rtol = 0.0
    assert np.isclose(bounds.left + pixel_size_x / 2, lon.min(), atol=atol, rtol=rtol)
    assert np.isclose(bounds.right - pixel_size_x / 2, lon.max(), atol=atol, rtol=rtol)
    assert np.isclose(bounds.top - pixel_size_y / 2, lat.max(), atol=atol, rtol=rtol)
    assert np.isclose(bounds.bottom + pixel_size_y / 2, lat.min(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("product", ["final", "preliminary"])
def test_source_fill_value_is_the_only_missing_value_marker(
    example_source_file_paths: dict[ChirpsProduct, Path], product: ChirpsProduct
) -> None:
    with rasterio.open(example_source_file_paths[product]) as reader:
        assert reader.nodata is None
        data = reader.read(1, out_dtype=np.float32)

    np.testing.assert_array_equal(
        np.unique(data[data < 0]), np.array([SOURCE_FILL_VALUE], dtype=np.float32)
    )
    # The ocean and marginal seas CHIRPS does not estimate over.
    assert 0.7 < float((data == SOURCE_FILL_VALUE).mean()) < 0.75
