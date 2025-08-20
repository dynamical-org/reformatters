from collections.abc import Iterable, Sequence
from datetime import timedelta  # noqa: F401

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import (  # noqa: F401
    CronJob,
    ReformatCronJob,
    ValidationCronJob,
)
from reformatters.noaa.hrrr.hrrr_config_models import HRRRDataVar

from .region_job import (
    HRRRSourceFileCoord,
    NoaaHrrrForecast48HourRegionJob,
)
from .template_config import NoaaHrrrForecast48HourTemplateConfig
from .validators import (
    check_data_is_current,
    check_forecast_completeness,
    check_spatial_coverage,
)


class NoaaHrrrForecast48HourDataset(DynamicalDataset[HRRRDataVar, HRRRSourceFileCoord]):
    """DynamicalDataset implementation for NOAA HRRR 48-hour forecast data."""

    template_config: NoaaHrrrForecast48HourTemplateConfig = (
        NoaaHrrrForecast48HourTemplateConfig()
    )
    region_job_class: type[NoaaHrrrForecast48HourRegionJob] = (
        NoaaHrrrForecast48HourRegionJob
    )

    def validators(self) -> Sequence[validation.DataValidator]:
        """Return validation functions for HRRR forecast data quality checks."""
        return (
            check_data_is_current,
            check_forecast_completeness,
            check_spatial_coverage,
        )

    def operational_kubernetes_resources(self, image_tag: str) -> Iterable[CronJob]:
        """Define Kubernetes cron jobs for operational updates and validation."""

        raise NotImplementedError("Disabled until we deploy operationally.")

        # # HRRR operational update job
        # # Run every 6 hours at :30 to catch new 48-hour forecasts (00, 06, 12, 18 UTC)
        # # HRRR data typically becomes available ~30-45 minutes after init time
        # operational_update_cron_job = ReformatCronJob(
        #     name=f"{self.dataset_id}-operational-update",
        #     schedule="30 0,6,12,18 * * *",  # Every 6 hours at :30 minutes
        #     pod_active_deadline=timedelta(hours=2),  # HRRR processing can take time
        #     image=image_tag,
        #     dataset_id=self.dataset_id,
        #     cpu="6",  # HRRR files are large and processing is CPU-intensive
        #     memory="24G",  # Large memory for GRIB processing and zarr chunks
        #     shared_memory="8Gi",  # Shared memory for parallel processing
        #     ephemeral_storage="50G",  # HRRR files can be large
        #     secret_names=[self.storage_config.k8s_secret_name],
        # )

        # # Validation job - run 1 hour after operational update
        # validation_cron_job = ValidationCronJob(
        #     name=f"{self.dataset_id}-validation",
        #     schedule="30 1,7,13,19 * * *",  # 1 hour after operational updates
        #     pod_active_deadline=timedelta(minutes=20),
        #     image=image_tag,
        #     dataset_id=self.dataset_id,
        #     cpu="2",
        #     memory="8G",
        #     secret_names=[self.storage_config.k8s_secret_name],
        # )

        # return [operational_update_cron_job, validation_cron_job]
