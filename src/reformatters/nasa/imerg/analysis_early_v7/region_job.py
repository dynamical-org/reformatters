from typing import ClassVar

from reformatters.nasa.imerg.region_job import (
    NasaImergRegionJob,
    NasaImergSourceFileCoord,
)


class NasaImergAnalysisEarlyV7SourceFileCoord(NasaImergSourceFileCoord):
    """Early Run granule: GES DISC product GPM_3IMERGHHE.07, filename code "E"."""

    gesdisc_product_id: ClassVar[str] = "GPM_3IMERGHHE.07"
    run_code: ClassVar[str] = "E"


class NasaImergAnalysisEarlyV7RegionJob(NasaImergRegionJob):
    source_file_coord_class: ClassVar[type[NasaImergSourceFileCoord]] = (
        NasaImergAnalysisEarlyV7SourceFileCoord
    )
