import numpy as np
import pandas as pd

from reformatters.noaa.gefs.forecast_16_day_spatial.template_config import (
    GefsForecast16DaySpatialTemplateConfig,
)

TEMPLATE_CONFIG = GefsForecast16DaySpatialTemplateConfig()


def test_dimension_coordinates_native_half_degree_grid() -> None:
    dim_coords = TEMPLATE_CONFIG.dimension_coordinates()

    latitude = dim_coords["latitude"]
    assert len(latitude) == 361
    assert latitude[0] == 90.0
    assert latitude[-1] == -90.0

    longitude = dim_coords["longitude"]
    assert len(longitude) == 720
    assert longitude[0] == 0.0
    assert longitude[-1] == 359.5

    lead_time = dim_coords["lead_time"]
    assert len(lead_time) == 105
    assert lead_time[0] == pd.Timedelta("0h")
    assert lead_time[-1] == pd.Timedelta("384h")

    assert len(dim_coords["ensemble_member"]) == 31


def test_data_vars_use_virtual_encoding() -> None:
    data_vars = TEMPLATE_CONFIG.data_vars
    assert len(data_vars) == 22
    for var in data_vars:
        encoding = var.encoding
        # One chunk per GRIB message; no shards, no compressors, GribberishCodec
        # serializer (see docs/virtual_datasets.md "Encoding rules").
        assert encoding.chunks == (1, 1, 1, 361, 720)
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


def test_get_template_expected_forecast_length() -> None:
    ds = TEMPLATE_CONFIG.get_template(pd.Timestamp("2020-10-02T00:00"))
    assert ds.sizes["init_time"] == 4
    np.testing.assert_array_equal(
        ds["expected_forecast_length"].values,
        np.full(4, pd.Timedelta("384h").to_timedelta64()),
    )
