import numpy as np
import pandas as pd

from reformatters.noaa.gefs.forecast_10_day_spatial.template_config import (
    GefsForecast10DaySpatialTemplateConfig,
)

TEMPLATE_CONFIG = GefsForecast10DaySpatialTemplateConfig()


def test_dimension_coordinates_native_quarter_degree_grid() -> None:
    dim_coords = TEMPLATE_CONFIG.dimension_coordinates()

    latitude = dim_coords["latitude"]
    assert len(latitude) == 721
    assert latitude[0] == 90.0
    assert latitude[-1] == -90.0

    longitude = dim_coords["longitude"]
    assert len(longitude) == 1440
    assert longitude[0] == 0.0
    assert longitude[-1] == 359.75

    lead_time = dim_coords["lead_time"]
    assert len(lead_time) == 81
    assert lead_time[0] == pd.Timedelta("0h")
    assert lead_time[-1] == pd.Timedelta("240h")

    assert len(dim_coords["ensemble_member"]) == 31


def test_data_vars_are_s_file_vars_with_virtual_encoding() -> None:
    data_vars = TEMPLATE_CONFIG.data_vars
    # The 35-day forecast's vars minus the three only available in the 0.5
    # degree a/b files (geopotential_height_500hpa, wind_u_100m, wind_v_100m).
    assert len(data_vars) == 19
    var_names = {v.name for v in data_vars}
    assert "temperature_2m" in var_names
    assert var_names.isdisjoint(
        {"geopotential_height_500hpa", "wind_u_100m", "wind_v_100m"}
    )

    for var in data_vars:
        encoding = var.encoding
        # One chunk per GRIB message; no shards, no compressors, GribberishCodec
        # serializer (see docs/virtual_datasets.md "Encoding rules").
        assert encoding.chunks == (1, 1, 1, 721, 1440)
        assert encoding.shards is None
        # Empty (not None, which serialization would drop -> zarr defaults)
        assert encoding.compressors is not None
        assert len(encoding.compressors) == 0
        assert encoding.filters is not None
        assert len(encoding.filters) == 0
        assert encoding.dtype == "float64"
        assert np.isnan(encoding.fill_value)
        assert encoding.serializer == {
            "name": "gribberish",
            "configuration": {"var": var.internal_attrs.grib_element},
        }


def test_raw_grib_value_attrs() -> None:
    # GribberishCodec serves raw GRIB values: Kelvin temperatures (no GDAL K->C
    # conversion) and accumulated (not deaccumulated-to-rate) precipitation.
    by_name = {v.name: v for v in TEMPLATE_CONFIG.data_vars}
    for name in ("temperature_2m", "maximum_temperature_2m", "minimum_temperature_2m"):
        assert by_name[name].attrs.units == "K"
    precipitation = by_name["precipitation_surface"]
    assert precipitation.attrs.units == "kg m-2"
    assert precipitation.attrs.standard_name == "precipitation_amount"
    assert precipitation.attrs.step_type == "accum"


def test_get_template_expected_forecast_length() -> None:
    ds = TEMPLATE_CONFIG.get_template(pd.Timestamp("2020-10-02T00:00"))
    assert ds.sizes["init_time"] == 4
    np.testing.assert_array_equal(
        ds["expected_forecast_length"].values,
        np.full(4, pd.Timedelta("240h").to_timedelta64()),
    )
