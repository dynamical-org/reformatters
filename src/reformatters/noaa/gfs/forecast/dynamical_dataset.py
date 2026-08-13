from collections.abc import Sequence
from datetime import timedelta

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.noaa.gfs.region_job import NoaaGfsSourceFileCoord
from reformatters.noaa.models import NoaaDataVar

from .region_job import NoaaGfsForecastRegionJob
from .template_config import NoaaGfsForecastTemplateConfig


class NoaaGfsForecastDataset(DynamicalDataset[NoaaDataVar, NoaaGfsSourceFileCoord]):
    template_config: NoaaGfsForecastTemplateConfig = NoaaGfsForecastTemplateConfig()
    region_job_class: type[NoaaGfsForecastRegionJob] = NoaaGfsForecastRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""
        workers = 2 * self.num_variable_groups()
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            # GFS f384 (full forecast) NOMADS p99 ~init+5h35m (recent files land
            # 5h13m-5h24m, drifting later), so fetch at init+5h38m.
            schedule="38 5,11,17,23 * * *",
            pod_active_deadline=timedelta(minutes=10),  # runs take <3 min
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3.5",
            memory="7G",
            shared_memory="1.5G",
            ephemeral_storage="20G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=workers,
            parallelism=workers,
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="48 5,11,17,23 * * *",  # 10m (pod_active_deadline) after reformat at :38
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        """The sequence of DataValidators to run on this dataset."""
        return (
            validation.check_forecast_current_data,
            validation.check_forecast_recent_nans,
        )
