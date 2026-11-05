from typing import ClassVar

from reformatters.ecmwf.fast.region_job import EcmwfFastRegionJob
from reformatters.ecmwf.fast.template_config import FastModel


class EcmwfAifsEnsForecastFastRegionJob(EcmwfFastRegionJob):
    model: ClassVar[FastModel] = "aifs"
