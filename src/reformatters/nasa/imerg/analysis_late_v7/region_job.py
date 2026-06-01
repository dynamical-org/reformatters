from typing import ClassVar

from reformatters.nasa.imerg.region_job import ImergRun, NasaImergRegionJob


class NasaImergAnalysisLateV7RegionJob(NasaImergRegionJob):
    run: ClassVar[ImergRun] = "late"
