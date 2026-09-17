from collections.abc import Sequence
from datetime import timedelta

import pandas as pd
from icechunk.store import IcechunkStore
from pydantic import Field

from reformatters.common import validation
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
    MIRROR_LOCATION_PREFIX,
    MIRROR_SECRET_NAME,
    parse_mirror_key,
)
from reformatters.noaa.hrrr.virtual_region_job import (
    NoaaHrrrForecastVirtualSourceFileCoord,
)

from .region_job import (
    NoaaHrrrForecast18HourVirtualFastRegionJob,
    hrrr_18_hour_virtual_fast_chunk_containers,
)
from .template_config import (
    RETENTION,
    NoaaHrrrForecast18HourVirtualFastTemplateConfig,
)


class CheckMirrorWindow(validation.Validator):
    """Fail when the store's moving window has stopped moving, or when the oldest init
    it references no longer decodes. The mirror expires files a fixed time after
    copying them, so a window that lags behind it points at nothing."""

    # The first init_time may trail now by RETENTION plus this: the hour a fire's
    # window covers, the hour until validation runs, and one missed fire.
    max_lag: timedelta = timedelta(hours=3)

    def check(
        self, context: validation.ValidationContext
    ) -> validation.ValidationResult:
        store = context.store
        assert isinstance(store, IcechunkStore)
        init_times = context.ds.get_index(context.append_dim)
        locations = store.session.all_virtual_chunk_locations()
        failure = self._window_failure(init_times) or _mirror_failure(
            context, locations
        )
        if failure is not None:
            return validation.ValidationResult(
                passed=False, message=failure, checked_count=len(locations)
            )
        return validation.ValidationResult(
            passed=True,
            message=f"init_time starts {init_times[0]} and the oldest init among "
            f"{len(locations)} referenced mirror files decodes",
            checked_count=len(locations),
        )

    def _window_failure(self, init_times: pd.Index) -> str | None:
        if len(init_times) == 0:
            return "Dataset has no init_time positions"
        oldest_allowed = (
            pd.Timestamp.now("UTC").tz_localize(None) - RETENTION - self.max_lag
        )
        if init_times[0] < oldest_allowed:
            return (
                f"init_time starts {init_times[0]}, before {oldest_allowed}: the update "
                "is not dropping inits and the mirror will expire their files"
            )
        max_length = RETENTION // pd.Timedelta("1h") + 1
        if len(init_times) > max_length:
            return (
                f"{len(init_times)} init_time positions exceed the {max_length} a "
                f"{RETENTION} window holds"
            )
        return None


def _mirror_failure(
    context: validation.ValidationContext, locations: Sequence[str]
) -> str | None:
    if not locations:
        return "The store references no files"
    foreign = [loc for loc in locations if not loc.startswith(MIRROR_LOCATION_PREFIX)]
    if foreign:
        return f"{len(foreign)} refs point outside the mirror: {foreign[:3]}"

    files = [_parse_location(location) for location in locations]
    oldest_init = min(init_time for init_time, _, _ in files)
    oldest_leads: dict[NoaaHrrrFileType, pd.Timedelta] = {}
    for init_time, lead_time, file_type in files:
        if init_time == oldest_init:
            oldest_leads[file_type] = min(
                lead_time, oldest_leads.get(file_type, lead_time)
            )
    # One variable from each file type's first lead: what a reader of the window's
    # oldest init fetches over the public domain.
    for file_type, lead_time in sorted(oldest_leads.items()):
        var = next(
            var
            for var in context.data_vars
            if var.internal_attrs.hrrr_file_type == file_type
            and var.attrs.step_type == "instant"
        )
        chunk = context.ds[var.path].sel(init_time=oldest_init, lead_time=lead_time)
        chunk = chunk.isel({dim: 0 for dim in chunk.dims if dim not in ("y", "x")})
        try:
            chunk.load()
        except Exception as e:  # noqa: BLE001 - any read failure is the finding
            return (
                f"{var.path} at the oldest referenced init {oldest_init} "
                f"(lead {lead_time}) does not decode: {e!r}"
            )
    return None


def _parse_location(
    location: str,
) -> tuple[pd.Timestamp, pd.Timedelta, NoaaHrrrFileType]:
    parsed = parse_mirror_key(location.removeprefix(MIRROR_LOCATION_PREFIX))
    assert parsed is not None, location
    return parsed


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

        return [operational_update_cron_job, validation_cron_job]

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
