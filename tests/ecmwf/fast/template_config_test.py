import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from reformatters.common.config_models import DatasetAttributes
from reformatters.ecmwf.aifs_ens.forecast_fast.template_config import (
    EcmwfAifsEnsForecastFastTemplateConfig,
)
from reformatters.ecmwf.fast.template_config import (
    EcmwfFastTemplateConfig,
    FastModel,
    lead_hours,
)
from reformatters.ecmwf.ifs_ens.forecast_0_25_degree_fast.template_config import (
    EcmwfIfsEnsForecast025DegreeFastTemplateConfig,
)


@pytest.mark.parametrize("model", ["ifs", "aifs"])
def test_native_grid_and_encoding(model: FastModel) -> None:
    config = EcmwfFastTemplateConfig(model=model)
    coords = config.dimension_coordinates()
    np.testing.assert_array_equal(coords["ensemble_member"], np.arange(51))
    np.testing.assert_array_equal(coords["latitude"], np.arange(90, -90.25, -0.25))
    np.testing.assert_array_equal(coords["longitude"], np.arange(-180, 180, 0.25))
    assert len(config.data_vars) == 7
    for var in config.data_vars:
        assert var.encoding.dtype == "float64"
        assert var.encoding.chunks == (1, 1, 1, 721, 1440)
        assert var.encoding.shards is None
        assert var.encoding.compressors == ()
        assert var.internal_attrs.keep_mantissa_bits == "no-rounding"
        assert np.isnan(var.encoding.fill_value)
    spatial_ref = next(c for c in config.coords if c.name == "spatial_ref")
    radius = 6367470 if model == "ifs" else 6371229
    assert spatial_ref.attrs.semi_major_axis == radius
    assert spatial_ref.attrs.semi_minor_axis == radius
    assert str(radius) in str(spatial_ref.attrs.crs_wkt)


@pytest.mark.parametrize("model", ["ifs", "aifs"])
def test_units_and_scaling(model: FastModel) -> None:
    config = EcmwfFastTemplateConfig(model=model)
    variables = {v.name: v for v in config.data_vars}
    temperature = variables["temperature_2m"]
    assert temperature.encoding.filters == [
        {"name": "scale_offset", "configuration": {"offset": -273.15}}
    ]
    rain = variables["total_precipitation_run_total_surface"]
    radiation = variables["downward_short_wave_radiation_run_total_surface"]
    assert rain.attrs.units == "kg m-2"
    assert radiation.attrs.units == "W s m-2"
    assert rain.encoding.filters == (
        [{"name": "scale_offset", "configuration": {"scale": 0.001}}]
        if model == "ifs"
        else []
    )
    for var in (rain, radiation):
        assert var.attrs.step_type == "accum"
        assert (
            var.attrs.comment
            == "Accumulated from the forecast initialization time to this step."
        )
    assert all(v.attrs.comment is None for v in config.data_vars[:5])


def test_model_specific_forecast_axes() -> None:
    assert len(lead_hours("ifs", 0)) == len(lead_hours("ifs", 12)) == 145
    assert len(lead_hours("ifs", 6)) == len(lead_hours("ifs", 18)) == 109
    assert lead_hours("ifs", 6)[-1] == 144
    assert all(len(lead_hours("aifs", hour)) == 61 for hour in (0, 6, 12, 18))
    config = EcmwfFastTemplateConfig(model="ifs")
    assert config.append_dim_start == pd.Timestamp("2026-11-01T00:00")
    inits = config.append_dim_coordinates("2026-11-02")
    assert pd.Timestamp("2026-11-01T06:00") in inits
    coords = config.derive_coordinates(
        xr.Dataset(
            coords={
                "init_time": inits,
                "lead_time": pd.to_timedelta([0, 360], unit="h"),
            }
        )
    )
    np.testing.assert_array_equal(
        coords["expected_forecast_length"][1],
        np.array([360, 144, 360, 144], dtype="timedelta64[h]"),
    )


@pytest.mark.parametrize(
    "config",
    [
        EcmwfIfsEnsForecast025DegreeFastTemplateConfig(),
        EcmwfAifsEnsForecastFastTemplateConfig(),
    ],
)
def test_commercial_license_is_exact_and_confined_to_license(
    config: EcmwfFastTemplateConfig,
) -> None:
    attrs = config.dataset_attributes.model_dump()
    assert attrs.pop("license") == "LicenseRef-Commercial"
    assert "licen" not in json.dumps(attrs).lower()
    with pytest.raises(ValidationError):
        DatasetAttributes(**attrs, license="LicenseRef-Unapproved")
