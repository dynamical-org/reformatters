from collections.abc import Sequence
from datetime import timedelta

from pydantic import Field

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.storage import (
    IcechunkVirtualConfig,
    manifest_append_dim_split,
)
from reformatters.noaa.gefs.gefs_config_models import NoaaGefsVirtualDataVar
from reformatters.noaa.gefs.virtual_region_job import (
    NoaaGefsForecastVirtualSourceFileCoord,
    gefs_virtual_chunk_containers,
)

from .region_job import NoaaGefsForecast10Day025DegreeVirtualRegionJob
from .template_config import NoaaGefsForecast10Day025DegreeVirtualTemplateConfig


class NoaaGefsForecast10Day025DegreeVirtualDataset(
    DynamicalDataset[NoaaGefsVirtualDataVar, NoaaGefsForecastVirtualSourceFileCoord]
):
    """NOAA GEFS 10 day 0.25 degree virtual (spatially-chunked, map-optimized icechunk) forecast dataset."""

    template_config: NoaaGefsForecast10Day025DegreeVirtualTemplateConfig = (
        NoaaGefsForecast10Day025DegreeVirtualTemplateConfig()
    )
    region_job_class: type[NoaaGefsForecast10Day025DegreeVirtualRegionJob] = (
        NoaaGefsForecast10Day025DegreeVirtualRegionJob
    )

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=gefs_virtual_chunk_containers(),
            # Four days of 6 hourly inits. Every array holds one ref per (lead time,
            # ensemble member), so a full manifest is 16 x 81 x 31 = 40176 refs, which
            # measures 0.68 MiB (17.8 bytes/ref) on this dataset's own manifests: well
            # inside the 3 MiB reader budget and far above the 1000 refs zstd location
            # compression needs. Splitting finer would cut per-commit flush cost but
            # multiply the manifest count, which every commit pays in the snapshot's
            # manifest list: 16 puts the archive near 550 splits per array, so ~21k
            # manifests across the 38 arrays. See "Manifest splitting" in
            # docs/virtual_datasets.md for the cost model.
            manifest_split=manifest_append_dim_split(split_size=16, dim="init_time"),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # The dataset id plus "-validate" exceeds the 52 character cron job name limit,
        # so the resolution takes its source file spelling.
        cron_job_name_prefix = self.dataset_id.replace("-0-25-degree", "-0p25")
        # The whole run publishes in one burst: f000 lands ~init+3h47m and the last
        # member's f240 ~init+5h37m. Fire just before the burst starts and poll through
        # it; the deadline clears the observed end by over half an hour and still ends
        # well before the next cycle's fire.
        operational_update_cron_job = ReformatCronJob(
            name=f"{cron_job_name_prefix}-update",
            schedule="45 3,9,15,21 * * *",
            pod_active_deadline=timedelta(hours=2, minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3.5",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=1,
            parallelism=1,
            suspend=True,
        )
        validation_cron_job = ValidationCronJob(
            name=f"{cron_job_name_prefix}-validate",
            # The update's fire plus its pod_active_deadline, plus 10 minutes: the
            # update stops polling 30 seconds before its deadline, so a validator
            # firing at exactly the deadline could read the store while the update is
            # still committing its last batch.
            schedule="25 6,12,18,0 * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=True,
        )

        return [operational_update_cron_job, validation_cron_job]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            # The newest init is 6h25m old when validation fires, so 12h adds a cycle
            # of slack: a run that rolls its last files to the next fire still passes,
            # while two cycles with nothing ingested fail. A cycle that published
            # nothing is caught here rather than by the completeness check below, which
            # skips append dim positions the store does not reach.
            validation.CheckCurrentData(max_delay=timedelta(hours=12)),
            # The leading tier covers the newest init, which the source may still be
            # finishing: 0.95 of its 81 x 31 files leaves room for the last four lead
            # times of every member, the tail the source lays down in its final ~20
            # minutes. That is looser than one whole member (81 files, 3.2%), so a
            # member missing entirely passes while its init is newest and fails at the
            # next fire, where the 18 hour window still covers it under the 1.0 tier.
            validation.CheckVirtualManifestCompleteness(
                min_present_fraction=(0.95, 1.0)
            ),
            validation.CheckVirtualDecodeHealth(),
        )
