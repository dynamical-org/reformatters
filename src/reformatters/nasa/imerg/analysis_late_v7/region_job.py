from typing import ClassVar

from reformatters.nasa.imerg.region_job import (
    NasaImergRegionJob,
    NasaImergSourceFileCoord,
)


class NasaImergAnalysisLateV7SourceFileCoord(NasaImergSourceFileCoord):
    """Late Run granule: GES DISC product GPM_3IMERGHHL.07, filename code "L"."""

    gesdisc_product_id: ClassVar[str] = "GPM_3IMERGHHL.07"
    run_code: ClassVar[str] = "L"


class NasaImergAnalysisLateV7RegionJob(NasaImergRegionJob):
    source_file_coord_class: ClassVar[type[NasaImergSourceFileCoord]] = (
        NasaImergAnalysisLateV7SourceFileCoord
    )
