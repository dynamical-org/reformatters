from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr

from reformatters.common.iterating import flatten_groups
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
def test_decoder_dependent_fields_are_deferred_in_config_and_stored_template(
    config: NoaaRrfsForecastTemplateConfig, count: int
) -> None:
    deterministic_deferred = {
        "minimum_vegetation_surface",
        "maximum_vegetation_surface",
        "specific_humidity_surface",
        "potential_evaporation_rate_surface",
        "potential_evaporation_surface",
    }
    member_deferred = {
        "aerosol_optical_thickness_atmosphere",
        "wildfire_potential_surface",
    }
    deferred = deterministic_deferred | (
        member_deferred if config.members or config.sub_hourly else set()
    )
    configured_paths = {v.path for v in config.data_vars}
    with xr.open_datatree(
        config.template_path(), engine="zarr", chunks=None, consolidated=False
    ) as tree:
        stored_paths = set(flatten_groups(tree).data_vars)
    assert len(config.data_vars) == len(configured_paths) == count
    assert stored_paths == configured_paths
    assert configured_paths.isdisjoint(deferred)
    if not config.members and not config.sub_hourly:
        assert member_deferred <= configured_paths


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
