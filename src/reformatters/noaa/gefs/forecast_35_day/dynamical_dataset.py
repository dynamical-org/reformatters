from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar

from .region_job import GefsForecast35DayRegionJob, GefsForecast35DaySourceFileCoord
from .template_config import GefsForecast35DayTemplateConfig


class GefsForecast35DayDataset(
    DynamicalDataset[GEFSDataVar, GefsForecast35DaySourceFileCoord]
):
    """GEFS 35-day forecast dataset implementation."""

    template_config: GefsForecast35DayTemplateConfig = GefsForecast35DayTemplateConfig()
    region_job_class: type[GefsForecast35DayRegionJob] = GefsForecast35DayRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""
        workers = 2 * self.num_variable_groups()
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            # New 00z init's f384 last perturbed member NOMADS last-modified ~init+6h28m;
            # +5 min buffer. The prior init (reprocessed each run, see operational_update_jobs)
            # is by now complete out to f840 (lands ~init+27h).
            schedule="33 6 * * *",
            pod_active_deadline=timedelta(minutes=30),
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
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="3 7 * * *",  # 30m (pod_active_deadline) after reformat at 06:33
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3",  # fit on 4 vCPU node
            memory="30G",  # fit on 32GB node
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            # The update ingests each init at init+6h33m; validation fires at init+7h03m.
            validation.CheckCurrentData(max_delay=timedelta(hours=7, minutes=3)),
            # The newest init_time stops at GEFS_PRE_EXTENSION_MAX, leaving 76 of
            # 181 lead times NaN at any spatial point: 0.42, or 0.422 for a variable
            # with no hour-0 value, whose lead_time=0 slice is dropped before the
            # fraction is computed. Older init_times are fully populated.
            validation.CheckRecentNans(max_nan_fraction=(0.45, 0.0)),
        )
