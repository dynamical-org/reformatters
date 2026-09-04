from reformatters.ucsb_chc.chirps.chirps_config_models import ChirpsProduct
from reformatters.ucsb_chc.chirps.region_job import (
    UcsbChcChirpsAnalysisMaterializedRegionJob,
)


class UcsbChcChirpsAnalysisPreliminaryRegionJob(
    UcsbChcChirpsAnalysisMaterializedRegionJob
):
    product: ChirpsProduct = "preliminary"
