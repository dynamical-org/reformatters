from typing import ClassVar

import pandas as pd

from reformatters.common.types import Timedelta
from reformatters.noaa.rtma.analysis_virtual.region_job import (
    NoaaRtmaAnalysisVirtualRegionJob,
    NoaaRtmaAnalysisVirtualSourceFileCoord,
)

SOURCE_PREFIX = "s3://noaa-urma-pds/"


class NoaaUrmaAnalysisVirtualSourceFileCoord(NoaaRtmaAnalysisVirtualSourceFileCoord):
    def get_url(self) -> str:
        return f"{SOURCE_PREFIX}urma2p5.{self.time:%Y%m%d}/urma2p5.t{self.time:%H}z.2dvaranl_ndfd.grb2_wexp"


class NoaaUrmaAnalysisVirtualRegionJob(NoaaRtmaAnalysisVirtualRegionJob):
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("48h")
    source_prefix: ClassVar[str] = SOURCE_PREFIX
    source_file_coord_class: ClassVar[type[NoaaUrmaAnalysisVirtualSourceFileCoord]] = (
        NoaaUrmaAnalysisVirtualSourceFileCoord
    )
