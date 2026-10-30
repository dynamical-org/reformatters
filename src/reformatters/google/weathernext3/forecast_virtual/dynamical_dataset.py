from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

import pandas as pd

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob

from .region_job import (
    GoogleWeathernext3ForecastVirtualRegionJob,
    GoogleWeathernext3ForecastVirtualSourceFileCoord,
)
from .template_config import (
    STATISTICS,
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
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(minutes=3, seconds=7)

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        prefix = (
            self.dataset_id.replace("weathernext3-forecast", "wn3")
            .replace("15-day", "15d")
            .replace("48-hour", "48h")
        )
        schedule = (
            "16 * * * *"
            if self.region_job_class.init_frequency == pd.Timedelta("1h")
            else "33 0,6,12,18 * * *"
        )
        deadline = self.region_job_class.init_frequency + timedelta(
            minutes=4, seconds=23
        )
        return [
            ReformatCronJob(
                name=f"{prefix}-update",
                schedule=schedule,
                concurrency_policy="Allow",
                pod_active_deadline=deadline,
                active_deadline=deadline,
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1.7",
                memory="7G",
                secret_names=self.store_factory.k8s_secret_names(),
                suspend=self.region_job_class.init_frequency == pd.Timedelta("1h"),
            ),
        ]

    def _min_present_fraction(self) -> tuple[float, ...]:
        # Hourly inits keep landing while the next one starts, and a synoptic init can
        # finish after the off-synoptic init after it, so two positions may be partial.
        if self.region_job_class.init_frequency == pd.Timedelta("1h"):
            return (0.0, 0.0, 1.0)
        return (1.0,)

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=9)),
            validation.CheckVirtualManifestCompleteness(
                min_present_fraction=self._min_present_fraction()
            ),
            validation.CheckVirtualDecodeHealth(
                positions="all",
                max_positions=2,
                sampled_leads=1,
                sampled_levels=len(STATISTICS),
                max_workers=2,
            ),
        )
