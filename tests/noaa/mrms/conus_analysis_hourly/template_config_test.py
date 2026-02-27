from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.pydantic import replace
from reformatters.noaa.mrms.conus_analysis_hourly.region_job import (
    NoaaMrmsRegionJob,
    NoaaMrmsSourceFileCoord,
)
from reformatters.noaa.mrms.conus_analysis_hourly.template_config import (
    MRMS_V12_START,
    NoaaMrmsConusAnalysisHourlyTemplateConfig,
)


def test_template_config_attrs() -> None:
    config = NoaaMrmsConusAnalysisHourlyTemplateConfig()

    assert config.dims == ("time", "latitude", "longitude")
    assert config.append_dim == "time"
    assert config.append_dim_start == pd.Timestamp("2014-10-01T00:00")
    assert config.append_dim_frequency == pd.Timedelta("1h")


def test_dimension_coordinates() -> None:
    config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    dim_coords = config.dimension_coordinates()

    assert "time" in dim_coords
    assert "latitude" in dim_coords
    assert "longitude" in dim_coords

    lat = dim_coords["latitude"]
    lon = dim_coords["longitude"]

    assert len(lat) == 3500
    assert len(lon) == 7000

    assert np.isclose(lat[0], 54.995)
    assert np.isclose(lat[-1], 20.005)
    assert np.isclose(lon[0], -129.995)
    assert np.isclose(lon[-1], -60.005)

    # Verify 0.01 degree spacing
    assert np.allclose(np.diff(lat), -0.01)
    assert np.allclose(np.diff(lon), 0.01)


def test_coordinate_configs() -> None:
    config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    coords = config.coords

    coord_names = {coord.name for coord in coords}
    expected = {"time", "latitude", "longitude", "spatial_ref"}
    assert coord_names == expected


def test_spatial_coordinates() -> None:
    config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    ds = config.get_template(config.append_dim_start + pd.Timedelta(days=1))

    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert ds.latitude.dims == ("latitude",)
    assert ds.longitude.dims == ("longitude",)

    assert len(ds.latitude) == 3500
    assert len(ds.longitude) == 7000

    assert np.isclose(float(ds.latitude.min()), 20.005)
    assert np.isclose(float(ds.latitude.max()), 54.995)
    assert np.isclose(float(ds.longitude.min()), -129.995)
    assert np.isclose(float(ds.longitude.max()), -60.005)


def test_data_vars() -> None:
    config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    data_vars = config.data_vars

    assert len(data_vars) == 4

    names = [v.name for v in data_vars]
    assert "precipitation_surface" in names
    assert "precipitation_pass_1_surface" in names
    assert "precipitation_radar_only_surface" in names
    assert "categorical_precipitation_type_surface" in names


def test_precipitation_vars_configured_for_deaccumulation() -> None:
    config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    precip_vars = [v for v in config.data_vars if v.internal_attrs.deaccumulate_to_rate]

    assert len(precip_vars) == 3
    for v in precip_vars:
        assert v.internal_attrs.window_reset_frequency == pd.Timedelta("1h")
        assert v.attrs.units == "kg m-2 s-1"
        assert v.attrs.step_type == "avg"


def test_precipitation_pass_1_available_from_v12() -> None:
    config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    pass_1_var = next(
        v for v in config.data_vars if v.name == "precipitation_pass_1_surface"
    )
    assert pass_1_var.internal_attrs.available_from == MRMS_V12_START


def test_precipitation_surface_has_pre_v12_product() -> None:
    config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    precip_var = next(v for v in config.data_vars if v.name == "precipitation_surface")
    assert precip_var.internal_attrs.mrms_product == "MultiSensor_QPE_01H_Pass2"
    assert precip_var.internal_attrs.mrms_product_pre_v12 == "GaugeCorr_QPE_01H"


def test_categorical_precipitation_type_is_instant() -> None:
    config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    ptype_var = next(
        v
        for v in config.data_vars
        if v.name == "categorical_precipitation_type_surface"
    )
    assert ptype_var.attrs.step_type == "instant"
    assert not ptype_var.internal_attrs.deaccumulate_to_rate
    assert ptype_var.internal_attrs.keep_mantissa_bits == "no-rounding"


def test_derive_coordinates_integration() -> None:
    config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    template_ds = config.get_template(pd.Timestamp("2014-10-01T12:00"))

    assert (
        template_ds.coords["time"]
        == pd.date_range("2014-10-01T00:00", "2014-10-01T11:00", freq="1h")
    ).all()
    assert "spatial_ref" in template_ds.coords


@pytest.mark.slow
def test_source_file_coords_and_crs_match_template(tmp_path: Path) -> None:
    """Download a real MRMS file and verify that its lat/lon coordinates and
    CRS/spatial_ref attributes match the template."""
    config = NoaaMrmsConusAnalysisHourlyTemplateConfig()
    time = pd.Timestamp("2024-01-15T12:00")

    precip_var = next(v for v in config.data_vars if v.name == "precipitation_surface")

    mock_ds = Mock()
    mock_ds.attrs = {"dataset_id": "noaa-mrms-conus-analysis-hourly"}
    region_job = NoaaMrmsRegionJob.model_construct(
        tmp_store=tmp_path,
        template_ds=mock_ds,
        data_vars=[precip_var],
        append_dim=config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    coord = NoaaMrmsSourceFileCoord(
        time=time,
        product=precip_var.internal_attrs.mrms_product,
        level=precip_var.internal_attrs.mrms_level,
    )
    coord = replace(coord, downloaded_path=region_job.download_file(coord))
    assert coord.downloaded_path is not None

    ds = xr.open_dataset(coord.downloaded_path, engine="rasterio")

    # Verify lat/lon coordinates match template dimension_coordinates
    dim_coords = config.dimension_coordinates()
    np.testing.assert_allclose(
        ds.y.values,
        dim_coords["latitude"],
        atol=1e-6,
        err_msg="GRIB latitudes do not match template dimension_coordinates",
    )
    np.testing.assert_allclose(
        ds.x.values,
        dim_coords["longitude"],
        atol=1e-6,
        err_msg="GRIB longitudes do not match template dimension_coordinates",
    )

    # Verify CRS/spatial_ref attributes match template
    spatial_ref_coord = next(c for c in config.coords if c.name == "spatial_ref")
    template_attrs = spatial_ref_coord.attrs.model_dump(exclude_none=True)
    file_attrs = dict(ds.spatial_ref.attrs)
    common_keys = set(template_attrs) & set(file_attrs)
    assert "spatial_ref" in common_keys
    assert "crs_wkt" in common_keys
    for key in common_keys:
        assert file_attrs[key] == template_attrs[key], (
            f"spatial_ref.{key}: file={file_attrs[key]!r} != template={template_attrs[key]!r}"
        )
