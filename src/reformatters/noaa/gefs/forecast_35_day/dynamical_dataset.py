from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.noaa.gefs.gefs_config_models import NoaaGefsDataVar

from .region_job import GefsForecast35DayRegionJob, GefsForecast35DaySourceFileCoord
from .template_config import GefsForecast35DayTemplateConfig


class GefsForecast35DayDataset(
    DynamicalDataset[NoaaGefsDataVar, GefsForecast35DaySourceFileCoord]
):
    """GEFS 35-day forecast dataset implementation."""

    template_config: GefsForecast35DayTemplateConfig = GefsForecast35DayTemplateConfig()
    region_job_class: type[GefsForecast35DayRegionJob] = GefsForecast35DayRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update this dataset."""
        workers = 2 * self.num_variable_groups()
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            # A 00z init's lead times through GEFS_PRE_EXTENSION_MAX are all on the
            # source by ~init+6h40m, occasionally as late as ~init+6h48m; starting at
            # init+6h45m catches the slowest members instead of racing them. The prior
            # init (reprocessed each run, see operational_update_jobs) is by now
            # complete out to f840 (lands ~init+28h).
            schedule="45 6 * * *",
            pod_active_deadline=timedelta(minutes=20),  # runs take ~8 min
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="6",  # fit on 8 vCPU node
            memory="120G",  # fit on 128GB node (more than needed)
            shared_memory="24G",
            ephemeral_storage="150G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=workers,
            parallelism=workers,
        )

        return [operational_update_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=7, minutes=5)),
            # The newest init_time stops at GEFS_PRE_EXTENSION_MAX, leaving 76 of
            # 181 lead times NaN at any spatial point: 0.42, or 0.422 for a variable
            # with no hour-0 value, whose lead_time=0 slice is dropped before the
            # fraction is computed. Older init_times are fully populated.
            validation.CheckRecentNans(max_nan_fraction=(0.45, 0.0)),
        )
