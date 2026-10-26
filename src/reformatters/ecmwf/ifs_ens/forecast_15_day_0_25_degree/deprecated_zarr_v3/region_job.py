from typing import ClassVar

from reformatters.ecmwf.ecmwf_utils import EcmwfOpenDataSource
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.region_job import (
    EcmwfIfsEnsForecast15Day025DegreeRegionJob,
)


class EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3RegionJob(
    EcmwfIfsEnsForecast15Day025DegreeRegionJob
):
    # Stays off the throttled s3://ecmwf-forecasts bucket.
    open_data_sources: ClassVar[tuple[EcmwfOpenDataSource, ...]] = ("gcs",)
