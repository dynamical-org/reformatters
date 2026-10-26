from collections.abc import Sequence
from datetime import timedelta

from pydantic import computed_field

from reformatters.common import validation
from reformatters.common.kubernetes import CronJob
from reformatters.common.pydantic import replace
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day025DegreeDataset,
)

from .region_job import EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3RegionJob
from .template_config import (
    EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3TemplateConfig,
)


class EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3Dataset(
    EcmwfIfsEnsForecast15Day025DegreeDataset
):
    """The plain Zarr v3 store of this dataset, processed independently of the
    Icechunk store so it can keep its own append_dim_start."""

    template_config: EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3TemplateConfig = (
        EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3TemplateConfig()
    )
    region_job_class: type[
        EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3RegionJob
    ] = EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3RegionJob

    @computed_field
    @property
    def dataset_id(self) -> str:
        return "ecmwf-ifs-ens-15-day-0-25-degree-zarr"

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        (update_cron_job,) = super().operational_kubernetes_resources(image_tag)
        return [
            replace(
                update_cron_job,
                # f360 lands on GCS ~08:20-08:35 UTC
                schedule="45 8 * * *",
                suspend=True,
            )
        ]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=9, minutes=20)),
            validation.CheckRecentNans(append_dim_window=3),
        )
