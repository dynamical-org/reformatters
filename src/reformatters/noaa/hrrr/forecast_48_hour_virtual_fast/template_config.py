from pydantic import computed_field

from reformatters.common.config_models import ROOT, DatasetAttributes, Group
from reformatters.common.pydantic import replace
from reformatters.common.types import Dim
from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    NoaaHrrrForecast48HourTemplateConfig,
)
from reformatters.noaa.hrrr.forecast_48_hour_virtual.template_config import (
    NoaaHrrrForecast48HourVirtualTemplateConfig,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar

# The materialized dataset's deaccumulated rates cannot be produced by a read-time
# codec (deaccumulation spans lead times), so the raw source quantities stand in.
_DEACCUMULATED_VAR_SUBSTITUTES: dict[str, tuple[str, ...]] = {
    "precipitation_surface": (
        "total_precipitation_surface",
        "precipitation_rate_surface",
    ),
    "snowfall_surface": ("total_snowfall_run_total_surface",),
}


class NoaaHrrrForecast48HourVirtualFastTemplateConfig(
    NoaaHrrrForecast48HourVirtualTemplateConfig
):
    """Virtual HRRR 48-hour forecast trimmed to the materialized
    noaa-hrrr-forecast-48-hour variable set. Every variable is single-level, so
    only wrfsfc source files are read."""

    dims: dict[Group, tuple[Dim, ...]] = {ROOT: ("init_time", "lead_time", "y", "x")}

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return replace(
            super().dataset_attributes,
            dataset_id="noaa-hrrr-forecast-48-hour-virtual-fast",
            dataset_version="0.1.0",
            name="NOAA HRRR forecast, 48 hour, virtual, fast",
        )

    def _catalog_data_vars(self) -> list[NoaaHrrrDataVar]:
        materialized_vars = NoaaHrrrForecast48HourTemplateConfig().data_vars
        deaccumulated_names = {
            v.name for v in materialized_vars if v.internal_attrs.deaccumulate_to_rate
        }
        assert deaccumulated_names == _DEACCUMULATED_VAR_SUBSTITUTES.keys(), (
            f"_DEACCUMULATED_VAR_SUBSTITUTES keys must be exactly the materialized "
            f"deaccumulated variables {sorted(deaccumulated_names)}"
        )
        target_names = {
            name
            for var in materialized_vars
            for name in _DEACCUMULATED_VAR_SUBSTITUTES.get(var.name, (var.name,))
        }
        data_vars = [
            var
            for var in super()._catalog_data_vars()
            if var.group is ROOT and var.name in target_names
        ]
        missing = target_names - {var.name for var in data_vars}
        assert not missing, (
            f"No virtual catalog equivalent for materialized variables {sorted(missing)}; "
            "add them to the shared catalog or map them in _DEACCUMULATED_VAR_SUBSTITUTES"
        )
        return data_vars
