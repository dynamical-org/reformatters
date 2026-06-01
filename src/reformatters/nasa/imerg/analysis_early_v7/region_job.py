from typing import ClassVar

from reformatters.nasa.imerg.region_job import ImergRun, NasaImergRegionJob


class NasaImergAnalysisEarlyV7RegionJob(NasaImergRegionJob):
    run: ClassVar[ImergRun] = "early"
