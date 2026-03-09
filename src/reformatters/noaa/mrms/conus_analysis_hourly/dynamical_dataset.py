from collections.abc import Iterable, Sequence
from datetime import timedelta
from functools import partial

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import (
    CronJob,
    ReformatCronJob,
    ValidationCronJob,
)

from .region_job import NoaaMrmsRegionJob, NoaaMrmsSourceFileCoord
from .template_config import NoaaMrmsConusAnalysisHourlyTemplateConfig, NoaaMrmsDataVar


class NoaaMrmsConusAnalysisHourlyDataset(
    DynamicalDataset[NoaaMrmsDataVar, NoaaMrmsSourceFileCoord]
):
    template_config: NoaaMrmsConusAnalysisHourlyTemplateConfig = (
        NoaaMrmsConusAnalysisHourlyTemplateConfig()
    )
    region_job_class: type[NoaaMrmsRegionJob] = NoaaMrmsRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Iterable[CronJob]:
        # Pass 2 has ~60-min latency. Update every 3 hours, 3 min after Pass 2 is expected.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="3 */3 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="14",
            memory="85G",
            shared_memory="71G",
            ephemeral_storage="60G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="33 */3 * * *",
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.5",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        max_expected_delay = timedelta(hours=3, minutes=30)
        # Pass 1 and Pass 2 have gauge-collection latency (~60 min); all other vars are always current.
        gauge_latency_vars = [
            "precipitation_pass_1_surface",
            "precipitation_pass_2_surface",
        ]
        radar_only_var = ["precipitation_radar_only_surface"]
        return (
            partial(
                validation.check_analysis_current_data,
                max_expected_delay=max_expected_delay,
            ),
            partial(
                validation.check_analysis_recent_nans,
                max_expected_delay=max_expected_delay,
                # precipitation_surface worst-case quarter-sampled NaN is ~36% (most recent
                # timestamp falls back to radar-only with ~34% structural coverage gaps).
                # categorical (PrecipFlag) is radar-derived with no gauge latency, similar coverage to radar_only.
                max_nan_percentage=40,
                spatial_sampling="quarter",
                exclude_vars=gauge_latency_vars + radar_only_var,
            ),
            partial(
                validation.check_analysis_recent_nans,
                max_expected_delay=max_expected_delay,
                # pass_1 and pass_2 worst-case quarter-sampled NaN is ~38% (1 of 3 checked
                # timestamps is 100% NaN due to gauge-collection latency, rest are ~6%).
                max_nan_percentage=40,
                spatial_sampling="quarter",
                include_vars=gauge_latency_vars,
            ),
            partial(
                validation.check_analysis_recent_nans,
                max_expected_delay=max_expected_delay,
                # Radar only is ~%34 nan always. With quarter sampling this can get as high as 62.2%.
                # It is available at all timestamps.
                max_nan_percentage=63,
                spatial_sampling="quarter",
                include_vars=radar_only_var,
            ),
        )
