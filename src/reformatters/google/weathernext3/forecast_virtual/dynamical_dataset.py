from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.google.weathernext_virtual.validation import CheckNoRefsInsideHoldback

from .region_job import (
    GoogleWeathernext3ForecastVirtualRegionJob,
    GoogleWeathernext3ForecastVirtualSourceFileCoord,
)
from .template_config import (
    GoogleWeathernext3DataVar,
    GoogleWeathernext3ForecastVirtualTemplateConfig,
)


class GoogleWeathernext3ForecastVirtualDataset(
    DynamicalDataset[
        GoogleWeathernext3DataVar, GoogleWeathernext3ForecastVirtualSourceFileCoord
    ]
):
    template_config: GoogleWeathernext3ForecastVirtualTemplateConfig
    region_job_class: type[GoogleWeathernext3ForecastVirtualRegionJob]
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=15)

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        prefix = (
            self.dataset_id.replace("weathernext3-forecast", "wn3")
            .replace("15-day", "15d")
            .replace("48-hour", "48h")
        )
        return [
            ReformatCronJob(
                name=f"{prefix}-update",
                schedule="10 * * * *",
                pod_active_deadline=timedelta(minutes=40),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1.7",
                memory="7G",
                secret_names=self.store_factory.k8s_secret_names(),
                suspend=True,
            ),
            ValidationCronJob(
                name=f"{prefix}-validate",
                schedule="55 * * * *",
                pod_active_deadline=timedelta(minutes=30),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1.3",
                memory="7G",
                secret_names=self.store_factory.k8s_secret_names(),
                suspend=True,
            ),
        ]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=12)),
            validation.CheckVirtualManifestCompleteness(
                min_present_fraction=(0.05, 1.0)
            ),
            CheckNoRefsInsideHoldback(),
            validation.CheckVirtualDecodeHealth(
                positions="all",
                max_positions=2,
                sampled_leads=1,
                sample_all_dims=("statistic",),
                max_workers=2,
            ),
        )
