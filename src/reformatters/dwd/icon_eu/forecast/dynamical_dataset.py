from collections.abc import Sequence

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob

from .region_job import DwdIconEuForecastRegionJob, DwdIconEuForecastSourceFileCoord
from .template_config import DwdIconEuDataVar, DwdIconEuForecastTemplateConfig


class DwdIconEuForecastDataset(
    DynamicalDataset[DwdIconEuDataVar, DwdIconEuForecastSourceFileCoord]
):
    template_config: DwdIconEuForecastTemplateConfig = DwdIconEuForecastTemplateConfig()
    region_job_class: type[DwdIconEuForecastRegionJob] = DwdIconEuForecastRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""
        # operational_update_cron_job = ReformatCronJob(
        #     name=f"{self.dataset_id}-operational-update",
        #     schedule=_OPERATIONAL_CRON_SCHEDULE,
        #     pod_active_deadline=timedelta(minutes=30),
        #     image=image_tag,
        #     dataset_id=self.dataset_id,
        #     cpu="14",
        #     memory="30G",
        #     shared_memory="12G",
        #     ephemeral_storage="30G",
        #     secret_names=[self.storage_config.k8s_secret_name],
        # )
        # validation_cron_job = ValidationCronJob(
        #     name=f"{self.dataset_id}-validation",
        #     schedule=_VALIDATION_CRON_SCHEDULE,
        #     pod_active_deadline=timedelta(minutes=10),
        #     image=image_tag,
        #     dataset_id=self.dataset_id,
        #     cpu="1.3",
        #     memory="7G",
        #     secret_names=[self.storage_config.k8s_secret_name],
        # )

        # return [operational_update_cron_job, validation_cron_job]
        raise NotImplementedError(
            f"Implement `operational_kubernetes_resources` on {self.__class__.__name__}"
        )

    def validators(self) -> Sequence[validation.DataValidator]:
        """Return a sequence of DataValidators to run on this dataset."""
        # return (
        #     validation.check_analysis_current_data,
        #     validation.check_analysis_recent_nans,
        # )
        raise NotImplementedError(
            f"Implement `validators` on {self.__class__.__name__}"
        )
