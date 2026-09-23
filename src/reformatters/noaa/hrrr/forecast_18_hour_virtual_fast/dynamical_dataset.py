from collections.abc import Sequence
from datetime import timedelta
from typing import Annotated, Any

import pandas as pd
import typer
from pydantic import Field

from reformatters.common import validation
from reformatters.common.config_models import DataVar
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.storage import (
    IcechunkVirtualConfig,
    manifest_append_dim_split,
)
from reformatters.common.time_utils import whole_hours
from reformatters.common.virtual_region_job import VirtualRegionJob
from reformatters.noaa.hrrr.hrrr_config_models import (
    NoaaHrrrDataVar,
    NoaaHrrrFileType,
)
from reformatters.noaa.hrrr.nomads_mirror import (
    MIRROR_SECRET_NAME,
    mirror_gribs,
)
from reformatters.noaa.hrrr.virtual_region_job import (
    NoaaHrrrForecastVirtualSourceFileCoord,
)

from .region_job import (
    NoaaHrrrForecast18HourVirtualFastRegionJob,
    hrrr_18_hour_virtual_fast_chunk_containers,
)
from .template_config import (
    NoaaHrrrForecast18HourVirtualFastTemplateConfig,
)


class CheckMirrorWindow(validation.Validator):
    """Fail when the oldest data the window holds no longer decodes. The mirror expires
    files a fixed time after copying them, so a window that outlives them points at
    nothing; nothing else reads that end of the window."""

    def check(
        self, context: validation.ValidationContext
    ) -> validation.ValidationResult:
        init_times = context.ds.get_index(context.append_dim)
        file_type_vars: dict[NoaaHrrrFileType, DataVar[Any]] = {}
        for var in context.data_vars:
            if var.attrs.step_type == "instant":
                file_type_vars.setdefault(var.internal_attrs.hrrr_file_type, var)

        decoded = []
        for var in file_type_vars.values():
            # A position with no ref reads as fill without a fetch, so this walks to
            # the oldest chunk the mirror actually serves.
            for init_time in init_times:
                chunk = context.ds[var.path].sel(
                    init_time=init_time, lead_time=pd.Timedelta(0)
                )
                chunk = chunk.isel(
                    {dim: 0 for dim in chunk.dims if dim not in ("y", "x")}
                )
                try:
                    chunk.load()
                except Exception as e:  # noqa: BLE001 - any read failure is the finding
                    return validation.ValidationResult(
                        passed=False,
                        message=f"{var.path} at init_time {init_time}, the oldest the "
                        f"store references, does not decode: {e!r}",
                    )
                if chunk.notnull().any():
                    decoded.append(f"{var.path} at {init_time}")
                    break
        if len(decoded) < len(file_type_vars):
            return validation.ValidationResult(
                passed=False,
                message=f"Only {decoded} of {len(file_type_vars)} file types hold "
                "any data in the window",
            )
        return validation.ValidationResult(
            passed=True,
            message=f"The oldest data in the window decodes: {decoded}",
            checked_count=len(decoded),
        )


MIRROR_CRON_JOB_NAME = "noaa-hrrr-nomads-mirror-gribs"


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
            # The 18-hour product's splits. The window is shorter than each, so every
            # array keeps a single manifest.
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
        # Hourly with a deadline inside the hour (concurrencyPolicy Replace); sfc f00
        # lands ~init+51m and f18 by ~init+87m, so the fire starts polling at init+49m.
        mirror_cron_job = CronJob(
            command=["mirror-gribs"],
            workers_total=1,
            parallelism=1,
            name=MIRROR_CRON_JOB_NAME,
            schedule="45 * * * *",
            pod_active_deadline=timedelta(minutes=59),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1",
            memory="2G",
            ephemeral_storage="8G",
            secret_names=[MIRROR_SECRET_NAME],
        )
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
            # reads it over the public domain like any reader.
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

        return [mirror_cron_job, operational_update_cron_job, validation_cron_job]

    def mirror_gribs(
        self,
        reformat_job_name: Annotated[str, typer.Argument(envvar="JOB_NAME")],
        poll_start_minutes: int = 49,
    ) -> None:
        """Copy the current init's GRIBs and indexes from NOMADS into the mirror."""
        with self._monitor(
            CronJob, reformat_job_name, cron_job_name=MIRROR_CRON_JOB_NAME
        ):
            mirror_gribs(
                self._operational_cron_job(CronJob, MIRROR_CRON_JOB_NAME),
                poll_start_minutes,
            )

    def get_cli(self) -> typer.Typer:
        app = super().get_cli()
        app.command()(self.mirror_gribs)
        return app

    def _virtual_validation_region_job(
        self,
        validators: Sequence[validation.Validator],
        reformat_job_name: str,
    ) -> (
        VirtualRegionJob[NoaaHrrrDataVar, NoaaHrrrForecastVirtualSourceFileCoord] | None
    ):
        # The update sweeps the whole window; validation holds only the recent inits
        # to completeness, so a file NOMADS never published alerts for hours, not days.
        job = super()._virtual_validation_region_job(validators, reformat_job_name)
        if job is None:
            return None
        recent = whole_hours(NoaaHrrrForecast18HourVirtualFastRegionJob.poll_window)
        return job.model_copy(
            update={"region": slice(max(job.region.stop - recent, 0), job.region.stop)}
        )

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
            CheckMirrorWindow(),
        )
