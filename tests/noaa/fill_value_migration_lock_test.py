import numpy as np
import pytest
import xarray as xr

from reformatters.noaa.gefs.analysis.template_config import GefsAnalysisTemplateConfig
from reformatters.noaa.gefs.forecast_35_day.template_config import (
    GefsForecast35DayTemplateConfig,
)
from reformatters.noaa.gfs.forecast.template_config import NoaaGfsForecastTemplateConfig
from reformatters.noaa.hrrr.analysis.template_config import (
    NoaaHrrrAnalysisTemplateConfig,
)
from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    NoaaHrrrForecast48HourTemplateConfig,
)

type MaterializedNoaaTemplateConfig = (
    NoaaGfsForecastTemplateConfig
    | GefsAnalysisTemplateConfig
    | GefsForecast35DayTemplateConfig
    | NoaaHrrrForecast48HourTemplateConfig
    | NoaaHrrrAnalysisTemplateConfig
)


def _same_fill(actual: float, expected: float) -> bool:
    return bool((np.isnan(actual) and np.isnan(expected)) or actual == expected)


@pytest.mark.parametrize(
    ("template_config", "default_fill", "fill_overrides"),
    [
        (NoaaGfsForecastTemplateConfig(), np.nan, {}),
        (GefsAnalysisTemplateConfig(), np.nan, {}),
        (GefsForecast35DayTemplateConfig(), np.nan, {}),
        (
            NoaaHrrrForecast48HourTemplateConfig(),
            np.nan,
            {"percent_frozen_precipitation_surface": -50.0},
        ),
        (
            NoaaHrrrAnalysisTemplateConfig(),
            np.nan,
            {"percent_frozen_precipitation_surface": -50.0},
        ),
    ],
)
def test_materialized_fill_value_migration_lock(
    template_config: MaterializedNoaaTemplateConfig,
    default_fill: float,
    fill_overrides: dict[str, float],
) -> None:
    float_vars = [
        var
        for var in template_config.data_vars
        if np.issubdtype(np.dtype(var.encoding.dtype), np.floating)
    ]
    raw_template = xr.open_zarr(
        template_config.template_path(), chunks=None, decode_cf=False
    )

    for var in float_vars:
        expected = fill_overrides.get(var.name, default_fill)
        assert _same_fill(var.encoding.fill_value, expected), var.name
        assert _same_fill(raw_template[var.name].encoding["fill_value"], expected), (
            var.name
        )
        expected_cf_fill = -50.0 if expected == -50.0 else np.nan
        assert _same_fill(
            raw_template[var.name].attrs["_FillValue"], expected_cf_fill
        ), var.name
        assert "missing_value" not in raw_template[var.name].attrs

    raw_template.close()
