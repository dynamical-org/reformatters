import numpy as np
import pytest

from reformatters.noaa.gefs.analysis.template_config import GefsAnalysisTemplateConfig
from reformatters.noaa.gefs.forecast_35_day.template_config import (
    GefsForecast35DayTemplateConfig,
)
from reformatters.noaa.gfs.analysis.template_config import NoaaGfsAnalysisTemplateConfig
from reformatters.noaa.gfs.forecast.template_config import NoaaGfsForecastTemplateConfig
from reformatters.noaa.hrrr.analysis.template_config import (
    NoaaHrrrAnalysisTemplateConfig,
)
from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    NoaaHrrrForecast48HourTemplateConfig,
)
from reformatters.noaa.models import (
    mask_source_missing_values_inplace,
    source_missing_value_var_names,
)

type MaterializedNoaaTemplateConfig = (
    NoaaGfsForecastTemplateConfig
    | NoaaGfsAnalysisTemplateConfig
    | GefsAnalysisTemplateConfig
    | GefsForecast35DayTemplateConfig
    | NoaaHrrrForecast48HourTemplateConfig
    | NoaaHrrrAnalysisTemplateConfig
)


@pytest.mark.parametrize(
    ("template_config", "expected_cloud_ceiling"),
    [
        (NoaaGfsForecastTemplateConfig(), (20_000.0, 1.0)),
        (NoaaGfsAnalysisTemplateConfig(), (20_000.0, 1.0)),
        (GefsAnalysisTemplateConfig(), (20_000.0, 1.0)),
        (GefsForecast35DayTemplateConfig(), (20_000.0, 1.0)),
        (NoaaHrrrForecast48HourTemplateConfig(), (9_999.0, 0.01)),
        (NoaaHrrrAnalysisTemplateConfig(), (9_999.0, 0.01)),
    ],
)
def test_materialized_source_missing_values(
    template_config: MaterializedNoaaTemplateConfig,
    expected_cloud_ceiling: tuple[float, float],
) -> None:
    data_vars = template_config.data_vars
    by_name = {var.name: var for var in data_vars}
    frozen = by_name["percent_frozen_precipitation_surface"].internal_attrs
    ceiling = by_name["geopotential_height_cloud_ceiling"].internal_attrs

    assert (frozen.source_missing_value, frozen.source_missing_value_atol) == (
        -50.0,
        0.01,
    )
    assert (
        ceiling.source_missing_value,
        ceiling.source_missing_value_atol,
    ) == expected_cloud_ceiling
    assert source_missing_value_var_names(data_vars) == (
        "percent_frozen_precipitation_surface",
        "geopotential_height_cloud_ceiling",
    )


def test_mask_source_missing_values_uses_packing_tolerance() -> None:
    var = next(
        var
        for var in NoaaGfsForecastTemplateConfig().data_vars
        if var.name == "percent_frozen_precipitation_surface"
    )
    values = np.array([-50.000008, -49.98, 0.0, 50.0], dtype=np.float32)

    mask_source_missing_values_inplace(values, var.internal_attrs)

    np.testing.assert_array_equal(
        values,
        np.array([np.nan, -49.98, 0.0, 50.0], dtype=np.float32),
    )
