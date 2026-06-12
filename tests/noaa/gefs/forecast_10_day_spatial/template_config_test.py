import numpy as np
import pandas as pd

from reformatters.noaa.gefs.forecast_10_day_spatial.template_config import (
    GefsForecast10DaySpatialTemplateConfig,
)

TEMPLATE_CONFIG = GefsForecast10DaySpatialTemplateConfig()


def test_dataset_attributes() -> None:
    attrs = TEMPLATE_CONFIG.dataset_attributes
    assert attrs.dataset_id == "noaa-gefs-forecast-10-day-spatial-dev"
    assert attrs.dataset_version == "0.1.0"
    assert attrs.spatial_domain == "Global"
    assert attrs.spatial_resolution == "0.25 degrees (~20km)"
    assert attrs.forecast_domain == "Forecast lead time 0-240 hours (0-10 days) ahead"
    assert (
        attrs.time_domain == "Forecasts initialized 2020-10-01 00:00:00 UTC to Present"
    )


def test_dims_and_append_dim() -> None:
    assert TEMPLATE_CONFIG.dims == (
        "init_time",
        "ensemble_member",
        "lead_time",
        "latitude",
        "longitude",
    )
    assert TEMPLATE_CONFIG.append_dim == "init_time"
    assert TEMPLATE_CONFIG.append_dim_start == pd.Timestamp("2020-10-01T00:00")
    assert TEMPLATE_CONFIG.append_dim_frequency == pd.Timedelta("6h")


def test_coords_names_and_encodings() -> None:
    coords = TEMPLATE_CONFIG.coords
    assert [c.name for c in coords] == [
        "latitude",
        "longitude",
        "spatial_ref",
        "init_time",
        "ensemble_member",
        "lead_time",
        "valid_time",
        "expected_forecast_length",
    ]
    by_name = {c.name: c for c in coords}
    # The archive is continuously appended to, never pre-sized toward a fixed end.
    assert by_name["init_time"].attrs.statistics_approximate is not None
    assert by_name["init_time"].attrs.statistics_approximate.max == "Present"
    assert by_name["valid_time"].attrs.statistics_approximate is not None
    assert by_name["valid_time"].attrs.statistics_approximate.max == "Present"
    for coord in coords:
        assert coord.encoding.shards is None


def test_append_dim_coordinates_range() -> None:
    init_times = TEMPLATE_CONFIG.append_dim_coordinates(
        pd.Timestamp("2020-10-02T00:00")
    )
    assert init_times[0] == pd.Timestamp("2020-10-01T00:00")
    # End is exclusive.
    assert init_times[-1] == pd.Timestamp("2020-10-01T18:00")
    assert len(init_times) == 4


def test_derive_coordinates() -> None:
    ds = TEMPLATE_CONFIG.get_template(pd.Timestamp("2020-10-02T00:00"))
    assert "spatial_ref" in ds.coords
    np.testing.assert_array_equal(
        ds["valid_time"].values,
        ds["init_time"].values[:, None] + ds["lead_time"].values[None, :],
    )


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
        # Empty (not None, which serialization would drop -> zarr defaults).
        # filters' BeforeValidator normalizes the input tuple to a list.
        assert encoding.compressors == ()
        assert encoding.filters == []
        assert encoding.dtype == "float64"
        assert np.isnan(encoding.fill_value)
        assert encoding.serializer == {
            "name": "gribberish",
            "configuration": {"var": var.internal_attrs.grib_element},
        }


def test_raw_grib_value_attrs() -> None:
    # GribberishCodec serves raw GRIB values: Kelvin temperatures (no GDAL K->C
    # conversion) and accumulated (not deaccumulated-to-rate) precipitation,
    # which is a different quantity and so gets a distinct variable name.
    by_name = {v.name: v for v in TEMPLATE_CONFIG.data_vars}
    for name in ("temperature_2m", "maximum_temperature_2m", "minimum_temperature_2m"):
        assert by_name[name].attrs.units == "K"
    assert "precipitation_surface" not in by_name
    precipitation = by_name["total_precipitation_surface"]
    assert precipitation.attrs.short_name == "tp"
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
