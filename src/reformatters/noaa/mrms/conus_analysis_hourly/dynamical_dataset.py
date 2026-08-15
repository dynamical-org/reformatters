from collections.abc import Sequence
from datetime import timedelta

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

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # Pass 2 has ~60-min latency. Update hourly, 3 min after Pass 2 is expected.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="3 * * * *",
            pod_active_deadline=timedelta(minutes=10),
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
            schedule="13 * * * *",
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.5",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_age=timedelta(hours=3, minutes=30)),
            validation.CheckRecentNans(
                # Gauge-corrected values arrive an hour late, leaving the newest
                # timestamp entirely NaN (excused by the leading 1.0). Measured
                # quarter-sampled NaN from the second-newest onward is 18.4% (6.2%
                # over the whole domain), constant across timestamps.
                max_nan_fraction=(1.0, 0.25),
                window=3,
                spatial_sampling="quarter",
                include_vars=[
                    "precipitation_pass_1_surface",
                    "precipitation_pass_2_surface",
                ],
            ),
            validation.CheckRecentNans(
                # The newest precipitation_surface is the radar-only field until gauge
                # data lands, so it carries radar-only's coverage gaps (52.9% in the
                # worst quarter), not the 18.4% its gauge-corrected older timestamps
                # show.
                max_nan_fraction=(0.63, 0.25),
                window=3,
                spatial_sampling="quarter",
                include_vars=["precipitation_surface"],
            ),
            validation.CheckRecentNans(
                # Radar coverage gaps only, identical at every timestamp: 34.1% over
                # the domain, 52.9% in the worst quarter.
                max_nan_fraction=0.63,
                spatial_sampling="quarter",
                include_vars=["precipitation_radar_only_surface"],
            ),
            validation.CheckRecentNans(
                # PrecipFlag is populated everywhere the grid is, measuring 0% NaN
                # across the domain at every timestamp.
                spatial_sampling="quarter",
                include_vars=["categorical_precipitation_type_surface"],
            ),
            validation.CheckRecentNans(
                # Outside radar/FFG coverage this is NaN: 64.2% over the domain, 77.3%
                # in the worst quarter. Its newest timestamp lands late like the
                # gauge-corrected fields (excused by the leading 1.0).
                max_nan_fraction=(1.0, 0.86),
                window=3,
                spatial_sampling="quarter",
                include_vars=["flash_qpe_ffg_max_surface"],
            ),
        )
