from collections.abc import Sequence
from datetime import timedelta
from typing import Annotated, ClassVar

import icechunk
import pandas as pd
import typer
from pydantic import Field

from reformatters.common import kubernetes
from reformatters.common.kubernetes import CronJob
from reformatters.common.pydantic import replace
from reformatters.common.storage import (
    IcechunkVirtualConfig,
    manifest_append_dim_split,
)
from reformatters.common.time_utils import whole_hours
from reformatters.noaa.hrrr.archive_gribs.copy_files_from_nomads import (
    copy_files_from_nomads,
)
from reformatters.noaa.hrrr.forecast_48_hour_virtual.dynamical_dataset import (
    NoaaHrrrForecast48HourVirtualDataset,
)

from .region_job import (
    CACHE_LOCATION_PREFIX,
    NoaaHrrrForecast48HourVirtualFastRegionJob,
    hrrr_fast_virtual_chunk_containers,
)
from .template_config import NoaaHrrrForecast48HourVirtualFastTemplateConfig

_CACHE_SECRET_NAME = "noaa-hrrr-nomads-cache-storage-options-key"  # noqa: S105


class NoaaHrrrForecast48HourVirtualFastDataset(NoaaHrrrForecast48HourVirtualDataset):
    """NOAA HRRR 48-hour virtual forecast trimmed to the materialized dataset's
    variable set. Operational timing is inherited from the full virtual dataset so the
    two products' ingest latency differs only by variable set."""

    template_config: NoaaHrrrForecast48HourVirtualFastTemplateConfig = (
        NoaaHrrrForecast48HourVirtualFastTemplateConfig()
    )
    region_job_class: type[NoaaHrrrForecast48HourVirtualFastRegionJob] = (
        NoaaHrrrForecast48HourVirtualFastRegionJob
    )

    # Must be in the format `rclone` expects: `:s3:<bucket>/<path>`. No double slash
    # after `:s3:` - the leading colon tells `rclone` to create an on-the-fly remote
    # from the env vars we set.
    nomads_cache_rclone_root: ClassVar[str] = ":s3:dynamical-noaa-hrrr-nomads-cache/"

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=hrrr_fast_virtual_chunk_containers(),
            # The NOMADS cache is private, so readers resolving a leading-edge ref
            # must supply credentials; the AWS archive stays anonymous.
            container_credentials={
                CACHE_LOCATION_PREFIX: icechunk.s3_from_env_credentials()
            },
            # Root-only, so one split size: 600 inits x 49 refs at ~16.4 bytes/ref
            # is ~0.5 MiB of active manifest per array, matching the full virtual
            # dataset's root arrays. See "Manifest splitting" in docs/virtual_datasets.md.
            manifest_split=manifest_append_dim_split(split_size=600, dim="init_time"),
        )
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        mirror_job = CronJob(
            command=["mirror-nomads-gribs"],
            workers_total=1,
            parallelism=1,
            name=f"{self.dataset_id}-mirror",
            # Start a few minutes before f00 publishes on NOMADS (p99 ~init+54m) and
            # poll through f48 (p99 ~init+109m). Copying is the latency path, so the
            # pod stays up across the cycle's publication window rather than sweeping
            # once; the deadline stays well under the 6h gap so fires never overlap.
            schedule="46 0,6,12,18 * * *",
            pod_active_deadline=timedelta(hours=1, minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="2",
            memory="2G",
            # rclone streams HTTP -> S3, so files never land on disk.
            ephemeral_storage="1G",
            secret_names=[_CACHE_SECRET_NAME],
        )
        # The store has not been backfilled, so ingest stays suspended while the
        # mirror runs on its own; drop the suspend once the backfill completes.
        ingest_jobs = [
            replace(job, suspend=True)
            for job in super().operational_kubernetes_resources(image_tag)
        ]
        return [mirror_job, *ingest_jobs]

    def mirror_nomads_gribs(
        self,
        reformat_job_name: Annotated[str, typer.Argument(envvar="JOB_NAME")],
        dst_root_path: str = nomads_cache_rclone_root,
        lead_hours: int = 48,
        max_minutes: int = 75,
        poll_seconds: int = 15,
        stats_logging_freq: str = "1m",
    ) -> None:
        """Mirror the current init's wrfsfc GRIB2 files from NOMADS into the cache bucket.

        Args:
            dst_root_path: Destination root, in rclone form e.g. ':s3:bucket/prefix/'.
            lead_hours: Mirror leads 0..lead_hours inclusive.
            max_minutes: Stop waiting for unpublished files after this long.
            poll_seconds: Interval between NOMADS directory-index checks.
            stats_logging_freq: Period between stats logs, e.g. "1m".
        """
        with self._monitor(
            CronJob,
            reformat_job_name,
            cron_job_name=f"{self.dataset_id}-mirror",
        ):
            cycle_hours = whole_hours(self.template_config.append_dim_frequency)
            current_init_time = pd.Timestamp.now().floor(f"{cycle_hours}h")

            secret = kubernetes.load_secret(_CACHE_SECRET_NAME)
            if secret:
                rclone_env_vars = {
                    "RCLONE_S3_PROVIDER": "AWS",
                    "RCLONE_S3_ACCESS_KEY_ID": secret["access_key_id"],
                    "RCLONE_S3_SECRET_ACCESS_KEY": secret["secret_access_key"],
                    "RCLONE_S3_REGION": secret["region"],
                    "RCLONE_S3_FORCE_PATH_STYLE": "false",
                }
            else:
                rclone_env_vars = None

            copy_files_from_nomads(
                dst_root_path=dst_root_path,
                init_times=[current_init_time],
                lead_hours=range(lead_hours + 1),
                max_duration=timedelta(minutes=max_minutes),
                poll_interval=timedelta(seconds=poll_seconds),
                stats_logging_freq=stats_logging_freq,
                env_vars=rclone_env_vars,
            )

    def get_cli(self) -> typer.Typer:
        """Create a CLI app with dataset commands."""
        app = super().get_cli()
        app.command()(self.mirror_nomads_gribs)
        return app
