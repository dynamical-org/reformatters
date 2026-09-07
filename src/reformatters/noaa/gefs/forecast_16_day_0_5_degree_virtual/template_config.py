import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import ROOT, DatasetAttributes
from reformatters.common.types import Dims, Timedelta
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_PRE_EXTENSION_MAX,
    GEFSSourceFileType,
)
from reformatters.noaa.gefs.virtual_template_config import (
    NoaaGefsForecastVirtualTemplateConfig,
)

# The a and b files publish every 3 hours through 240 hours of lead time and every
# 6 hours from there to GEFS_PRE_EXTENSION_MAX.
_FINE_LEAD_FREQUENCY = pd.Timedelta("3h")
_FINE_LEAD_MAX = pd.Timedelta("240h")
_COARSE_LEAD_FREQUENCY = pd.Timedelta("6h")


class NoaaGefsForecast16Day05DegreeVirtualTemplateConfig(
    NoaaGefsForecastVirtualTemplateConfig
):
    """Virtual GEFS 16 day forecast: every 0.5 degree pgrb2a and pgrb2b message of all
    31 ensemble members, out to the 384 hour lead where those files end."""

    source_file_types: frozenset[GEFSSourceFileType] = frozenset({"a", "b"})
    forecast_length: Timedelta = GEFS_PRE_EXTENSION_MAX

    dims: Dims = {
        ROOT: ("init_time", "ensemble_member", "lead_time", "latitude", "longitude"),
        "pressure_level": (
            "init_time",
            "ensemble_member",
            "lead_time",
            "latitude",
            "longitude",
            "pressure_level",
        ),
        "model_level": (
            "init_time",
            "ensemble_member",
            "lead_time",
            "latitude",
            "longitude",
            "model_level",
        ),
        "height_above_mean_sea_level": (
            "init_time",
            "ensemble_member",
            "lead_time",
            "latitude",
            "longitude",
            "height_above_mean_sea_level",
        ),
    }

    def lead_times(self) -> pd.TimedeltaIndex:
        return pd.timedelta_range(
            "0h", _FINE_LEAD_MAX, freq=_FINE_LEAD_FREQUENCY
        ).union(
            pd.timedelta_range(
                _FINE_LEAD_MAX, self.forecast_length, freq=_COARSE_LEAD_FREQUENCY
            )
        )

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return self._dataset_attributes(
            dataset_id="noaa-gefs-forecast-16-day-0-5-degree-virtual",
            dataset_version="0.1.0",
            name="NOAA GEFS forecast, 16 day, 0.5 degree, virtual",
            description=(
                "Weather forecasts from the Global Ensemble Forecast System (GEFS) "
                "operated by NOAA NWS NCEP."
            ),
        )
