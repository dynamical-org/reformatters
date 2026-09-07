from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import rasterio

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


def test_dataset_ids_and_names() -> None:
    final = UcsbChcChirpsAnalysisFinalTemplateConfig()
    preliminary = UcsbChcChirpsAnalysisPreliminaryTemplateConfig()
    assert final.dataset_attributes.dataset_id == "ucsb-chc-chirps-analysis-final"
    assert final.dataset_attributes.name == "UCSB CHC CHIRPS analysis final"
    assert (
        preliminary.dataset_attributes.dataset_id
        == "ucsb-chc-chirps-analysis-preliminary"
    )
    assert preliminary.dataset_attributes.name == "UCSB CHC CHIRPS analysis preliminary"


def test_append_dim_starts() -> None:
    final = UcsbChcChirpsAnalysisFinalTemplateConfig()
    preliminary = UcsbChcChirpsAnalysisPreliminaryTemplateConfig()
    # The time coordinate labels each daily total at the start of the day it covers.
    assert str(final.append_dim_start) == "1981-01-01 00:00:00"
    assert str(preliminary.append_dim_start) == "2025-01-01 00:00:00"
    assert final.append_dim_frequency == preliminary.append_dim_frequency
    assert str(final.append_dim_frequency) == "1 days 00:00:00"

    first_time = final.dimension_coordinates()["time"][0]
    assert str(first_time) == "1981-01-01 00:00:00"


def test_grid_orientation_and_endpoints() -> None:
    coords = UcsbChcChirpsAnalysisFinalTemplateConfig().dimension_coordinates()
    lat = coords["latitude"]
    lon = coords["longitude"]
    assert lat.shape == (GRID_LAT_SIZE,)
    assert lon.shape == (GRID_LON_SIZE,)
    # Pixel centers of the 60N-60S, 180W-180E grid, latitude descending to match
    # the source files' north -> south row order.
    assert lat[0] == 59.975
    assert lat[-1] == -59.975
    assert np.all(np.diff(lat) < 0)
    assert lon[0] == -179.975
    assert lon[-1] == 179.975
    assert np.all(np.diff(lon) > 0)


def test_precipitation_surface_metadata() -> None:
    (precip,) = UcsbChcChirpsAnalysisFinalTemplateConfig().data_vars
    assert precip.name == "precipitation_surface"
    assert precip.attrs.short_name == "prate"
    assert precip.attrs.long_name == "Precipitation rate"
    assert precip.attrs.standard_name == "precipitation_flux"
    assert precip.attrs.units == "kg m-2 s-1"
    assert precip.attrs.step_type == "avg"
    assert precip.attrs.comment is not None
    assert "24 hours starting at the time coordinate" in precip.attrs.comment
    assert "the ocean and marginal seas" in precip.attrs.comment
    assert precip.internal_attrs.keep_mantissa_bits == 8
    assert precip.internal_attrs.source_fill_value == SOURCE_FILL_VALUE
    assert np.isnan(precip.encoding.fill_value)


@pytest.fixture(scope="session")
def example_source_file_paths() -> dict[ChirpsProduct, Path]:
    config = UcsbChcChirpsAnalysisFinalTemplateConfig()
    region_job = UcsbChcChirpsAnalysisFinalRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=config.get_template(config.append_dim_start),
        data_vars=list(config.data_vars),
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
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
