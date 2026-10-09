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
        (NoaaRrfsForecast84HourVirtualTemplateConfig(), 318),
        (NoaaRrfsForecast18HourVirtualTemplateConfig(), 318),
        (NoaaRrfsForecastSubHourlyVirtualTemplateConfig(), 38),
        (NoaaRrfsEnsForecastVirtualTemplateConfig(), 62),
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
    # These surface fields are effectively empty in NOAA's source as of 2026-10-09.
    # RRFS is not yet operational, so this may change. specific_humidity_2m is
    # present and remains in the dataset.
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
    # These fields are not usable as of 2026-10-09; RRFS is not yet operational,
    # so this may change.
    assert configured_paths.isdisjoint(
        {
            "instantaneous_upward_long_wave_radiation_flux_top_of_atmosphere",
            "vertical_u_component_shear_0_1000m",
            "vertical_v_component_shear_0_1000m",
            "categorical_precipitation_exceeding_flash_flood_guidance_surface",
            "categorical_precipitation_exceeding_flash_flood_guidance_run_total_surface",
        }
    )
    if not config.sub_hourly:
        assert {
            "vertical_u_component_shear_0_6000m",
            "vertical_v_component_shear_0_6000m",
        } <= configured_paths
    if not config.sub_hourly and not config.members:
        assert "upward_long_wave_radiation_flux_top_of_atmosphere" in configured_paths
    assert config.append_dim_start == config.append_dim_start.normalize()
    if config.sub_hourly or config.members:
        # The ensemble fields are effectively empty in NOAA's source as of
        # 2026-10-09; RRFS is not yet operational, so this may change.
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
    ],
    ids=lambda c: c.dataset_id,
)
@pytest.mark.parametrize(("component", "element"), [("u", "UEID"), ("v", "VEID")])
def test_effective_layer_storm_motion_metadata(
    config: NoaaRrfsForecastTemplateConfig, component: str, element: str
) -> None:
    variables = {v.path: v for v in config.data_vars}
    assert (
        f"effective_inflow_layer_wind_{component}_level_of_free_convection"
        not in variables
    )
    variable = variables[
        f"effective_layer_storm_motion_{component}_level_of_free_convection"
    ]
    assert (
        variable.attrs.long_name == f"Effective layer {component.upper()} storm motion"
    )
    assert variable.attrs.standard_name is None
    assert variable.attrs.short_name == element.lower()
    assert variable.attrs.units == "m s-1"
    assert variable.internal_attrs.grib_element == element
    assert variable.internal_attrs.grib_index_level == "level of free convection"


@pytest.mark.parametrize(
    "config",
    [
        NoaaRrfsForecast84HourVirtualTemplateConfig(),
        NoaaRrfsForecast18HourVirtualTemplateConfig(),
    ],
    ids=lambda c: c.dataset_id,
)
def test_fractional_ice_and_inapplicable_count_metadata(
    config: NoaaRrfsForecastTemplateConfig,
) -> None:
    variables = {v.name: v for v in config.data_vars}
    ice = variables["ice_cover_surface"]
    assert ice.attrs.long_name == "Ice concentration"
    assert ice.attrs.standard_name == "sea_ice_area_fraction"
    assert ice.attrs.units == "1"
    assert ice.attrs.flag_values is None
    assert ice.attrs.flag_meanings is None
    assert np.isnan(
        variables["number_of_soil_layers_in_root_zone_surface"].encoding.fill_value
    )
    assert np.isnan(
        variables[
            "effective_layer_shear_u_level_of_free_convection"
        ].encoding.fill_value
    )
    assert np.isnan(
        variables[
            "effective_layer_shear_v_level_of_free_convection"
        ].encoding.fill_value
    )


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
    # These two fields are effectively empty in NOAA's ensemble source as of
    # 2026-10-09; RRFS is not yet operational, so this may change.
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
