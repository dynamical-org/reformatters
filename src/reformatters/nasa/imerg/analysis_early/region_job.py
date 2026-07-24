import pandas as pd

from reformatters.nasa.imerg.imerg_config_models import ImergRun
from reformatters.nasa.imerg.region_job import NasaImergAnalysisMaterializedRegionJob


class NasaImergAnalysisEarlyRegionJob(NasaImergAnalysisMaterializedRegionJob):
    run: ImergRun = "early"
    publish_latency = pd.Timedelta(hours=4.6)
