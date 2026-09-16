from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

import pandas as pd
from pydantic import Field

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.storage import (
    IcechunkVirtualConfig,
    manifest_append_dim_split,
)
from reformatters.common.virtual_region_job import VirtualRegionJob
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from reformatters.noaa.hrrr.nomads_mirror import MIRROR_SECRET_NAME, parse_mirror_key
from reformatters.noaa.hrrr.virtual_region_job import (
    NoaaHrrrForecastVirtualSourceFileCoord,
)

from .region_job import (
    NoaaHrrrForecast18HourVirtualFastRegionJob,
    hrrr_18_hour_virtual_fast_chunk_containers,
)
from .template_config import NoaaHrrrForecast18HourVirtualFastTemplateConfig


class CheckMirrorRefsRepointed(validation.Validator):
    """Fail when a file first offered by the NOMADS mirror has still not been written
    from NODD `max_age` after its init: the mirror expires files three days after
    copying them, so this is the warning before refs into it break."""

    requires_virtual_dataset: ClassVar[bool] = True
    max_age: timedelta = timedelta(hours=36)

    def check(
        self, context: validation.ValidationContext
    ) -> validation.ValidationResult:
        job = context.virtual_region_job()
        assert isinstance(job, NoaaHrrrForecast18HourVirtualFastRegionJob), type(job)
        now = pd.Timestamp.now("UTC").tz_localize(None)
        pending = job.pending_repoints()
        overdue = []
        for key in pending:
            parsed = parse_mirror_key(key)
            assert parsed is not None, key
            if now - parsed[0] > self.max_age:
                overdue.append(key)
        if overdue:
            return validation.ValidationResult(
                passed=False,
                message=f"{len(overdue)} files initialized more than {self.max_age} ago "
                f"are still not written from NODD and may point at the NOMADS mirror, "
                f"which expires them: {overdue[:10]}",
                checked_count=len(pending),
            )
        return validation.ValidationResult(
            passed=True,
            message=f"{len(pending)} files await a rewrite from NODD, none older "
            f"than {self.max_age}",
            checked_count=len(pending),
        )


class NoaaHrrrForecast18HourVirtualFastDataset(
    DynamicalDataset[NoaaHrrrDataVar, NoaaHrrrForecastVirtualSourceFileCoord]
):
    template_config: NoaaHrrrForecast18HourVirtualFastTemplateConfig = (
        NoaaHrrrForecast18HourVirtualFastTemplateConfig()
    )
    region_job_class: type[NoaaHrrrForecast18HourVirtualFastRegionJob] = (
        NoaaHrrrForecast18HourVirtualFastRegionJob
    )

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=hrrr_18_hour_virtual_fast_chunk_containers(),
            # The 18-hour product's splits, so full manifest byte sizes match it.
            manifest_split=manifest_append_dim_split(
                split_size={
                    r"^/pressure_level/": 225,
                    r"^/model_level/": 200,
                    None: 1500,
                },
                dim="init_time",
            ),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        # Race the current init: f00 arrives near :51 and f18 normally by init + 86m.
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="50 * * * *",
            pod_active_deadline=timedelta(minutes=59),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="4",
            memory="3.7G",
            # The update lists and reads the mirror over R2's signed S3 API; validation
            # reads mirror-pointed chunks over the public domain like any reader.
            secret_names=[*self.store_factory.k8s_secret_names(), MIRROR_SECRET_NAME],
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            # The update's fire plus its pod_active_deadline, so the run being
            # validated has always stopped writing.
            schedule="49 * * * *",
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.5",
            memory="3.7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]

    def _virtual_validation_region_job(
        self,
        validators: Sequence[validation.Validator],
        reformat_job_name: str,
    ) -> (
        VirtualRegionJob[NoaaHrrrDataVar, NoaaHrrrForecastVirtualSourceFileCoord] | None
    ):
        # Validation asks "is the file in the store", not "is it still worth
        # rewriting from NODD", so it probes the manifest without the repoint override.
        job = super()._virtual_validation_region_job(validators, reformat_job_name)
        if job is None:
            return None
        job = job.model_copy(update={"repoint_mirrored": False})
        assert isinstance(job, NoaaHrrrForecast18HourVirtualFastRegionJob)
        job.bind_store_factory(self.store_factory)
        return job

    def validators(self) -> Sequence[validation.Validator]:
        return (
            # The hourly update polls each init from init+50m (f00 publishes ~init+51m);
            # validation fires at init+1h49m.
            validation.CheckCurrentData(max_delay=timedelta(hours=1, minutes=49)),
            # Newest ingested init: the run that just ended may have deferred late files
            # to the next fire, but f00 lands an hour before its poll deadline, so 5%
            # (3 of 57 files, one lead's worth) separates a deferral from a cycle that
            # published nothing.
            validation.CheckVirtualManifestCompleteness(
                min_present_fraction=(0.05, 1.0)
            ),
            validation.CheckVirtualDecodeHealth(),
            CheckMirrorRefsRepointed(),
        )
