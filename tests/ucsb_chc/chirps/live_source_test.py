from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr

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
from tests.ucsb_chc.chirps.region_job_test import _job

# Every test that contacts the CHIRPS host lives in this file, so pytest-xdist's
# loadfile scheduling runs them in one process under the one CHIRPS rate limiter.


@pytest.mark.slow
@pytest.mark.parametrize("product", ["final", "preliminary"])
def test_live_directory_listing(product: ChirpsProduct) -> None:
    times = pd.date_range("2025-01-01", periods=3)
    coords = _job(product).generate_source_file_coords(
        xr.Dataset(coords={"time": times}), []
    )
    assert [coord.time for coord in coords] == list(times)


@pytest.fixture(scope="session")
def example_source_file_paths(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[ChirpsProduct, Path]:
    config = UcsbChcChirpsAnalysisFinalTemplateConfig()
    region_job = UcsbChcChirpsAnalysisFinalRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=config.get_template(config.append_dim_start),
        data_vars=list(config.data_vars),
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "reformatters.common.download.DOWNLOAD_DIR",
            tmp_path_factory.mktemp("chirps-source"),
        )
        return {
            product: region_job.download_file(
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


@pytest.mark.slow
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


@pytest.mark.slow
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
