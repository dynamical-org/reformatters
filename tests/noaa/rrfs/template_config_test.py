from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio

from reformatters.noaa.rrfs.forecast_18_hour_virtual.template_config import (
    NoaaRrfsForecast18HourVirtualTemplateConfig,
)
from reformatters.noaa.rrfs.forecast_84_hour_virtual.template_config import (
    NoaaRrfsForecast84HourVirtualTemplateConfig,
)
from reformatters.noaa.rrfs.forecast_sub_hourly_virtual.template_config import (
    NoaaRrfsForecastSubHourlyVirtualTemplateConfig,
)
from reformatters.noaa.rrfs.template_config import NoaaRrfsForecastTemplateConfig
from reformatters.noaa.rrfs_ens.forecast_virtual.template_config import (
    NoaaRrfsEnsForecastVirtualTemplateConfig,
)


@pytest.mark.parametrize(
    ("config", "count"),
    [
        (NoaaRrfsForecast84HourVirtualTemplateConfig(), 323),
        (NoaaRrfsForecast18HourVirtualTemplateConfig(), 323),
        (NoaaRrfsForecastSubHourlyVirtualTemplateConfig(), 39),
        (NoaaRrfsEnsForecastVirtualTemplateConfig(), 64),
    ],
    ids=lambda value: (
        value.dataset_id
        if isinstance(value, NoaaRrfsForecastTemplateConfig)
        else str(value)
    ),
)
def test_configured_field_inventory(
    config: NoaaRrfsForecastTemplateConfig, count: int
) -> None:
    removed_fields = {
        "minimum_vegetation_surface",
        "maximum_vegetation_surface",
        "specific_humidity_surface",
        "potential_evaporation_rate_surface",
        "potential_evaporation_surface",
    }
    deterministic_fields = {
        "aerosol_optical_thickness_atmosphere",
        "wildfire_potential_surface",
    }
    configured_paths = {v.path for v in config.data_vars}
    assert len(config.data_vars) == len(configured_paths) == count
    assert configured_paths.isdisjoint(removed_fields)
    assert config.append_dim_start == config.append_dim_start.normalize()
    if config.sub_hourly or config.members:
        assert configured_paths.isdisjoint(deterministic_fields)
    else:
        assert deterministic_fields <= configured_paths
    if not config.members:
        assert "specific_humidity_2m" in configured_paths
    if not config.sub_hourly:
        assert "pressure_level/specific_humidity" in configured_paths


@pytest.mark.parametrize(
    "config",
    [
        NoaaRrfsForecast84HourVirtualTemplateConfig(),
        NoaaRrfsForecast18HourVirtualTemplateConfig(),
        NoaaRrfsEnsForecastVirtualTemplateConfig(),
    ],
    ids=lambda c: c.dataset_id,
)
def test_source_lead_availability(
    config: NoaaRrfsForecastTemplateConfig,
) -> None:
    hours = int(config.forecast_length / pd.Timedelta("1h")) + 1
    expected = {"total_precipitation_surface": range(1, hours)}
    if not config.members:
        expected |= {
            "aerosol_optical_thickness_atmosphere": range(hours),
            "wildfire_potential_surface": range(hours),
        }
    variables = {v.path: v for v in config.data_vars}
    for name, available_hours in expected.items():
        var = variables[name]
        assert var.internal_attrs.source_family == "2dfld"
        assert [
            h for h in range(hours) if var.available_at(pd.Timedelta(hours=h))
        ] == list(available_hours)
        assert np.isnan(var.encoding.fill_value)


@pytest.mark.parametrize(
    ("fixture_name", "config"),
    [
        ("spfh", NoaaRrfsForecast84HourVirtualTemplateConfig()),
        ("aotk", NoaaRrfsEnsForecastVirtualTemplateConfig()),
    ],
)
def test_conus_grid_matches_source_header(
    fixture_name: str, config: NoaaRrfsForecastTemplateConfig
) -> None:
    shape, bounds, resolution, crs = config._spatial_info()
    spatial_ref = next(c for c in config.coords if c.name == "spatial_ref")
    assert spatial_ref.attrs.GeoTransform is not None
    expected_transform = tuple(float(v) for v in spatial_ref.attrs.GeoTransform.split())
    path = Path(__file__).parent / "fixtures" / f"{fixture_name}-all-missing.grib2"
    with rasterio.open(path) as source:
        assert source.shape == shape
        np.testing.assert_allclose(source.bounds, bounds, rtol=0, atol=1e-8)
        assert source.res == (resolution[0], -resolution[1])
        assert source.crs == rasterio.crs.CRS.from_string(crs)
        np.testing.assert_allclose(
            source.transform.to_gdal(), expected_transform, rtol=0, atol=1e-8
        )
