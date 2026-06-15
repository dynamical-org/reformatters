import numpy as np
import pandas as pd

from reformatters.ecmwf.ifs_ens.forecast_15_day_spatial.template_config import (
    EcmwfIfsEnsForecast15DaySpatialTemplateConfig,
)

TEMPLATE_CONFIG = EcmwfIfsEnsForecast15DaySpatialTemplateConfig()


def test_dataset_attributes() -> None:
    attrs = TEMPLATE_CONFIG.dataset_attributes
    assert attrs.dataset_id == "ecmwf-ifs-ens-forecast-15-day-spatial-dev"
    assert attrs.dataset_version == "0.1.0"
    assert attrs.spatial_domain == "Global"
    assert attrs.spatial_resolution == "0.25 degrees (~20km)"
    assert attrs.forecast_domain == "Forecast lead time 0-360 hours (0-15 days) ahead"
    assert (
        attrs.time_domain == "Forecasts initialized 2026-05-13 00:00:00 UTC to Present"
    )


def test_dims_and_append_dim() -> None:
    assert TEMPLATE_CONFIG.dims == (
        "init_time",
        "lead_time",
        "ensemble_member",
        "latitude",
        "longitude",
    )
    assert TEMPLATE_CONFIG.append_dim == "init_time"
    assert TEMPLATE_CONFIG.append_dim_start == pd.Timestamp("2026-05-13T00:00")
    assert TEMPLATE_CONFIG.append_dim_frequency == pd.Timedelta("24h")


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
        pd.Timestamp("2026-05-15T00:00")
    )
    assert init_times[0] == pd.Timestamp("2026-05-13T00:00")
    # End is exclusive; 24h frequency.
    assert init_times[-1] == pd.Timestamp("2026-05-14T00:00")
    assert len(init_times) == 2


def test_derive_coordinates() -> None:
    ds = TEMPLATE_CONFIG.get_template(pd.Timestamp("2026-05-15T00:00"))
    assert "spatial_ref" in ds.coords
    np.testing.assert_array_equal(
        ds["valid_time"].values,
        ds["init_time"].values[:, None] + ds["lead_time"].values[None, :],
    )


def test_dimension_coordinates_native_grid() -> None:
    dim_coords = TEMPLATE_CONFIG.dimension_coordinates()

    latitude = dim_coords["latitude"]
    assert len(latitude) == 721
    assert latitude[0] == 90.0
    assert latitude[-1] == -90.0

    # Native ECMWF open-data longitude wraps: 180..359.75, then 0..179.75.
    longitude = dim_coords["longitude"]
    assert len(longitude) == 1440
    assert longitude[0] == 180.0
    assert longitude[719] == 359.75
    assert longitude[720] == 0.0
    assert longitude[-1] == 179.75

    lead_time = dim_coords["lead_time"]
    assert len(lead_time) == 85
    assert lead_time[0] == pd.Timedelta("0h")
    assert lead_time[-1] == pd.Timedelta("360h")

    # control (0) + 50 perturbed
    assert len(dim_coords["ensemble_member"]) == 51
    assert dim_coords["ensemble_member"][0] == 0
    assert dim_coords["ensemble_member"][-1] == 50


def test_data_vars_are_virtual_with_radiation_dropped() -> None:
    data_vars = TEMPLATE_CONFIG.data_vars
    # The materialized dataset's 19 vars minus the two accumulated-radiation vars.
    assert len(data_vars) == 17
    var_names = {v.name for v in data_vars}
    assert "temperature_2m" in var_names
    assert var_names.isdisjoint(
        {
            "downward_long_wave_radiation_flux_surface",
            "downward_short_wave_radiation_flux_surface",
        }
    )

    for var in data_vars:
        encoding = var.encoding
        # One chunk per GRIB message; no shards, no compressors, GribberishCodec
        # serializer (see docs/virtual_datasets.md "Encoding rules").
        assert encoding.chunks == (1, 1, 1, 721, 1440)
        assert encoding.shards is None
        # Empty (not None, which serialization would drop -> zarr defaults).
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
    # conversion), a 0-1 cloud fraction (not scaled to percent), and accumulated
    # (not deaccumulated-to-rate) precipitation, which is a different quantity and
    # so gets a distinct variable name.
    by_name = {v.name: v for v in TEMPLATE_CONFIG.data_vars}
    for name in (
        "temperature_2m",
        "temperature_850hpa",
        "temperature_925hpa",
        "dew_point_temperature_2m",
    ):
        assert by_name[name].attrs.units == "K"
    assert by_name["total_cloud_cover_atmosphere"].attrs.units == "1"
    assert "precipitation_surface" not in by_name
    precipitation = by_name["total_precipitation_surface"]
    assert precipitation.attrs.short_name == "tp"
    assert precipitation.attrs.units == "m"
    assert precipitation.attrs.standard_name == "lwe_thickness_of_precipitation_amount"
    assert precipitation.attrs.step_type == "accum"


def test_get_template_expected_forecast_length() -> None:
    ds = TEMPLATE_CONFIG.get_template(pd.Timestamp("2026-05-15T00:00"))
    assert ds.sizes["init_time"] == 2
    np.testing.assert_array_equal(
        ds["expected_forecast_length"].values,
        np.full(2, pd.Timedelta("360h").to_timedelta64()),
    )
