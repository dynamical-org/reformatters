import numpy as np
import pandas as pd

from reformatters.common.iterating import item
from reformatters.ecmwf.aifs_deterministic.forecast.template_config import (
    EcmwfAifsForecastTemplateConfig,
)


def test_dimension_coordinates() -> None:
    config = EcmwfAifsForecastTemplateConfig()
    coords = config.dimension_coordinates()

    assert config.append_dim == "init_time"
    assert config.append_dim_frequency == pd.Timedelta("6h")

    lead_times = coords["lead_time"]
    assert len(lead_times) == 61  # 0h to 360h every 6h
    assert lead_times[0] == pd.Timedelta("0h")
    assert lead_times[-1] == pd.Timedelta("360h")

    lat = coords["latitude"]
    assert len(lat) == 721
    assert float(lat.max()) == 90.0
    assert float(lat.min()) == -90.0

    lon = coords["longitude"]
    assert len(lon) == 1440
    assert float(lon.min()) == -180.0
    assert float(lon.max()) == 179.75
    assert float(item(np.unique(np.diff(lon)))) == 0.25
    assert float(item(np.unique(np.diff(lat)))) == -0.25


def test_data_vars_date_available() -> None:
    config = EcmwfAifsForecastTemplateConfig()
    expanded_date = pd.Timestamp("2025-02-26T00:00")

    vars_with_date = [
        v for v in config.data_vars if v.internal_attrs.date_available is not None
    ]
    vars_without_date = [
        v for v in config.data_vars if v.internal_attrs.date_available is None
    ]

    assert len(vars_with_date) > 0
    assert len(vars_without_date) > 0

    for v in vars_with_date:
        assert v.internal_attrs.date_available == expanded_date


def test_derive_coordinates() -> None:
    config = EcmwfAifsForecastTemplateConfig()
    ds = config.get_template(config.append_dim_start + config.append_dim_frequency)

    assert "valid_time" in ds.coords
    assert "ingested_forecast_length" in ds.coords
    assert "expected_forecast_length" in ds.coords
    assert "spatial_ref" in ds.coords

    # valid_time = init_time + lead_time
    expected_valid_time = ds["init_time"] + ds["lead_time"]
    np.testing.assert_array_equal(ds["valid_time"].values, expected_valid_time.values)
