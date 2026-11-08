from typing import Any

import pandas as pd

from reformatters.common.types import Timestamp
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar
from reformatters.ecmwf.ecmwf_template_config import (
    Ecmwf025DegreeForecastTemplateConfig,
)

# ECMWF's open data cutover from the aifs/ stream to aifs-single/. At this init the
# source path changes, the added variables start, and tp/cp switch from metres to
# kg m-2 -- one boundary, so both variants key all three off this constant.
AIFS_SINGLE_FORMAT_CHANGE_DATE = pd.Timestamp("2025-02-24T06:00")
# The aifs-single stream spends its first 36 hours under an experimental/ path segment
# before moving to the operational one.
AIFS_SINGLE_OPERATIONAL_PATH_DATE = pd.Timestamp("2025-02-25T06:00")


def aifs_single_stream_path(init_time: Timestamp) -> str:
    """The source path segment between the init hour directory and the file name."""
    if init_time < AIFS_SINGLE_FORMAT_CHANGE_DATE:
        return "aifs/0p25/oper"
    if init_time < AIFS_SINGLE_OPERATIONAL_PATH_DATE:
        return "aifs-single/0p25/experimental/oper"
    return "aifs-single/0p25/oper"


class EcmwfAifsSingleCommonTemplateConfig[DATA_VAR: EcmwfDataVar](
    Ecmwf025DegreeForecastTemplateConfig[DATA_VAR]
):
    # Earlier data (2024-02-29 to 2024-03-31) uses a reduced Gaussian grid.
    append_dim_start: Timestamp = pd.Timestamp("2024-04-01T00:00")

    def dimension_coordinates(self) -> dict[str, Any]:
        return super().dimension_coordinates() | {
            "lead_time": pd.timedelta_range("0h", "360h", freq="6h"),
        }
