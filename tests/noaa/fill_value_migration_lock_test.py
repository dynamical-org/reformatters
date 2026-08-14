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


@pytest.mark.parametrize(
    "template_config",
    [
        NoaaGfsForecastTemplateConfig(),
        GefsAnalysisTemplateConfig(),
        GefsForecast35DayTemplateConfig(),
        NoaaHrrrForecast48HourTemplateConfig(),
        NoaaHrrrAnalysisTemplateConfig(),
    ],
)
def test_materialized_fill_value_migration_lock(
    template_config: MaterializedNoaaTemplateConfig,
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
        assert np.isnan(var.encoding.fill_value), var.name
        assert np.isnan(raw_template[var.name].encoding["fill_value"]), var.name
        assert np.isnan(raw_template[var.name].attrs["_FillValue"]), var.name
        assert "missing_value" not in raw_template[var.name].attrs

    raw_template.close()
