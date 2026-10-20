from typing import ClassVar

from reformatters.noaa.rtma.analysis_virtual.region_job import (
    SOURCE_PREFIX,
    NoaaRtmaAnalysisVirtualRegionJob,
    NoaaRtmaAnalysisVirtualSourceFileCoord,
)


class NoaaRtmaRuAnalysisVirtualSourceFileCoord(NoaaRtmaAnalysisVirtualSourceFileCoord):
    def get_url(self) -> str:
        return f"{SOURCE_PREFIX}rtma2p5_ru.{self.time:%Y%m%d}/rtma2p5_ru.t{self.time:%H%M}z.2dvaranl_ndfd.grb2"


class NoaaRtmaRuAnalysisVirtualRegionJob(NoaaRtmaAnalysisVirtualRegionJob):
    source_file_coord_class: ClassVar[
        type[NoaaRtmaRuAnalysisVirtualSourceFileCoord]
    ] = NoaaRtmaRuAnalysisVirtualSourceFileCoord
