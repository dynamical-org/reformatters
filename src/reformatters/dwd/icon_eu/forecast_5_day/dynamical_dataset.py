import time
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import PurePosixPath
from typing import Annotated, ClassVar, Final

import typer

from reformatters.common import kubernetes, validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.logging import get_logger
from reformatters.dwd.archive_gribs.copy_files_from_dwd import copy_files_from_dwd_https
from reformatters.dwd.archive_gribs.copy_icosahedral_files_from_dwd import (
    copy_icosahedral_files_from_dwd_https,
)

from .region_job import (
    DwdIconEuForecast5DayRegionJob,
    DwdIconEuForecast5DaySourceFileCoord,
)
from .template_config import DwdIconEuDataVar, DwdIconEuForecast5DayTemplateConfig

log = get_logger(__name__)

ARCHIVE_GRIB_FILES_DEADLINE: Final[timedelta] = timedelta(hours=4)
# The icosahedral phase starts no run within this long of the deadline.
ICOSAHEDRAL_RUN_ALLOWANCE: Final[timedelta] = timedelta(minutes=30)
ICOSAHEDRAL_NWP_INIT_HOURS: Final[tuple[int, ...]] = (0, 3, 6, 9, 12, 15, 18, 21)
# DWD `lvt1` level types archived alongside single-level parameters: pressure and soil.
# Model levels (150) are excluded for their storage cost.
ICOSAHEDRAL_LEVEL_TYPES: Final[tuple[int, ...]] = (100, 106)


class DwdIconEuForecast5DayDataset(
    DynamicalDataset[DwdIconEuDataVar, DwdIconEuForecast5DaySourceFileCoord]
):
    template_config: DwdIconEuForecast5DayTemplateConfig = (
        DwdIconEuForecast5DayTemplateConfig()
    )
    region_job_class: type[DwdIconEuForecast5DayRegionJob] = (
        DwdIconEuForecast5DayRegionJob
    )

    # `dynamical_grib_archive_rclone_root` must be in the format that `rclone` expects:
    # `:s3:<bucket>/<path>`. Note that there is no double slash after `:s3:`. The leading colon
    # tells `rclone` to create an on the fly rclone backend and use the env variables we set.
    dynamical_grib_archive_rclone_root: ClassVar[str] = (
        ":s3:us-west-2.opendata.source.coop/dynamical/dwd-icon-grib/icon-eu/regular-lat-lon/"
    )
    dynamical_icosahedral_grib_archive_rclone_root: ClassVar[str] = (
        ":s3:us-west-2.opendata.source.coop/dynamical/dwd-icon-grib/icon-eu/icosahedral/"
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""
        archive_grib_files_job = CronJob(
            command=["archive-grib-files"],
            workers_total=1,
            parallelism=1,
            name=f"{self.dataset_id}-archive-grib-files",
            # We want the 00, 06, 12, and 18 ICON-EU runs. DWD's transfer to their http server starts
            # about 2 hours 15 mins after the init time, and finishes about 3 hours 45 minutes after
            # the init time. So, to avoid copying incomplete files, we fetch the files 4 hours after
            # each init. But note that, every time the cron job runs, the script checks all 4 NWP
            # inits. This design helps to keep the code simple, especially when recovering if the
            # script hasn't run for a while. It only takes 4 minutes to check an NWP run that we've
            # already transferred. The icosahedral files are copied after the regular lat/lon
            # files, in the same pod, so the two copies never load DWD's server at once.
            schedule="0 4,10,16,22 * * *",
            pod_active_deadline=ARCHIVE_GRIB_FILES_DEADLINE,
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.5",
            memory="6G",
            ephemeral_storage="1G",  # not used
            # Credentials to write to Source Coop
            secret_names=["source-coop-storage-options-key"],
        )

        # ICON-EU runs at 00, 06, 12, 18 UTC. DWD's complete forecast (f120) is available by
        # ~init+3h49m (p99). We schedule the reformat 3 minutes after that (3h52m after init).
        # The archive job may not have finished copying to Source Coop yet, in which case
        # download_file falls back to reading directly from DWD.
        workers = 2 * self.num_variable_groups()
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule="52 3,9,15,21 * * *",
            pod_active_deadline=timedelta(minutes=10),  # runs take <5 min
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3",
            memory="14G",
            shared_memory="400M",
            ephemeral_storage="30G",
            secret_names=self.store_factory.k8s_secret_names(),
            workers_total=workers,
            parallelism=workers,
        )

        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule="2 4,10,16,22 * * *",  # 10m (pod_active_deadline) after reformat at :52
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="0.7",
            memory="3.5G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [
            archive_grib_files_job,
            operational_update_cron_job,
            validation_cron_job,
        ]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            # The update ingests each init at init+3h52m; validation fires at init+4h02m.
            validation.CheckCurrentData(max_delay=timedelta(hours=4, minutes=2)),
            validation.CheckRecentNans(),
        )

    def archive_grib_files(
        self,
        reformat_job_name: Annotated[str, typer.Argument(envvar="JOB_NAME")],
        # It would've made more sense for `dst_root_path` to be a `PurePosixPath` but Typer doesn't
        # handle `PurePosixPath`, so we use a `str` to keep Typer happy.
        dst_root_path: str = dynamical_grib_archive_rclone_root,
        # The `ty: ignore` on the line below is because Typer doesn't understand the type hints
        # `tuple[int, ...]` or `Sequence[int]`, so we have to use `list[int]`.
        nwp_init_hours: list[int] = (0, 6, 12, 18),  # ty: ignore[invalid-parameter-default]
        icosahedral_dst_root_path: str = dynamical_icosahedral_grib_archive_rclone_root,
        icosahedral_nwp_init_hours: list[int] = ICOSAHEDRAL_NWP_INIT_HOURS,  # ty: ignore[invalid-parameter-default]
        icosahedral_level_types: list[int] = ICOSAHEDRAL_LEVEL_TYPES,  # ty: ignore[invalid-parameter-default]
        icosahedral_params: list[str] = (),  # ty: ignore[invalid-parameter-default]
        transfer_parallelism: int = 64,
        checkers: int = 32,
        stats_logging_freq: str = "1m",
    ) -> None:
        """Restructure DWD GRIB files from DWD's HTTPS server to a timestamped directory
        structure: first the regular lat/lon files, then the icosahedral files.

        Args:
            dst_root_path: The destination root directory. e.g. for S3, the dst_root could be: ':s3:bucket/foo/bar'
            nwp_init_hours: The ICON-EU NWP model runs to transfer.
            icosahedral_dst_root_path: The destination root directory for the icosahedral files.
            icosahedral_nwp_init_hours: The ICON-EU NWP model runs to transfer on the icosahedral grid.
            icosahedral_level_types: DWD `lvt1` level types to transfer in addition to
                single-level parameters: 100 pressure levels, 106 soil levels, 150 model levels.
            icosahedral_params: DWD parameter names to transfer, e.g. T_2M. Empty transfers all of them.
            transfer_parallelism: Number of concurrent workers during the copy operation.
                Each worker fetches a file from src_host, copies it to the destination, and waits for
                the destination to acknowledge completion before fetching another file from the source.
                When fetching from HTTPS and writing to object storage, this could be set arbitrarily
                high, although setting it too high (>256?) might be detrimental to performance.
            checkers: This number is passed to the `rclone --checkers` argument.
                In the context of recursive file listing, it appears `checkers` controls the number of
                directories that are listed in parallel. Note that more is not always better. For
                example, on a small VM with only 2 CPUs, `rclone` maxes out the CPUs if `checkers` is
                above 32, and this actually slows down file listing.
                For more info, see the rclone docs: https://rclone.org/docs/#checkers-int
            stats_logging_freq: The period between each stats log. e.g. "1m" to log stats every minute.
                See https://rclone.org/docs/#stats-duration
        """
        with self._monitor(
            CronJob,
            reformat_job_name,
            cron_job_name=f"{self.dataset_id}-archive-grib-files",
        ):
            # When running in prod, `secret` will be {'key': 'xxx', 'secret': 'xxxx'}.
            # When not running in prod, `secret` will be empty.
            secret = kubernetes.load_secret("source-coop-storage-options-key")
            if secret:
                s3_credentials_env_vars_for_rclone = {
                    "RCLONE_S3_PROVIDER": "AWS",
                    "RCLONE_S3_ACCESS_KEY_ID": secret["key"],
                    "RCLONE_S3_SECRET_ACCESS_KEY": secret["secret"],
                    "RCLONE_S3_REGION": "us-west-2",
                    "RCLONE_S3_FORCE_PATH_STYLE": "false",
                }
            else:
                s3_credentials_env_vars_for_rclone = None

            started = time.monotonic()

            def copy_regular_lat_lon() -> None:
                for nwp_init_hour in nwp_init_hours:
                    src_root_path = PurePosixPath(
                        f"/weather/nwp/icon-eu/grib/{nwp_init_hour:02d}"
                    )
                    copy_files_from_dwd_https(
                        src_host="https://opendata.dwd.de",
                        src_root_path=src_root_path,
                        dst_root_path=PurePosixPath(dst_root_path),
                        transfer_parallelism=transfer_parallelism,
                        checkers=checkers,
                        stats_logging_freq=stats_logging_freq,
                        env_vars=s3_credentials_env_vars_for_rclone,
                    )

            def copy_icosahedral() -> None:
                elapsed = timedelta(seconds=time.monotonic() - started)
                copy_icosahedral_files_from_dwd_https(
                    dst_root_path=icosahedral_dst_root_path,
                    nwp_init_hours=icosahedral_nwp_init_hours,
                    level_types=icosahedral_level_types,
                    params=icosahedral_params,
                    time_budget=ARCHIVE_GRIB_FILES_DEADLINE
                    - ICOSAHEDRAL_RUN_ALLOWANCE
                    - elapsed,
                    transfer_parallelism=transfer_parallelism,
                    checkers=checkers,
                    stats_logging_freq=stats_logging_freq,
                    env_vars=s3_credentials_env_vars_for_rclone,
                )

            # DWD keeps each run for about a day, so one phase failing must not stop the other.
            errors = [
                error
                for error in (
                    _run_phase("Regular lat/lon", copy_regular_lat_lon),
                    _run_phase("Icosahedral", copy_icosahedral),
                )
                if error is not None
            ]
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise ExceptionGroup("Both ICON-EU archive phases failed", errors)

    def get_cli(self) -> typer.Typer:
        """Create a CLI app with dataset commands."""
        app = super().get_cli()
        app.command()(self.archive_grib_files)
        return app


def _run_phase(name: str, phase: Callable[[], None]) -> Exception | None:
    """Run `phase`, logging how long it took, and return the exception it raised, if any."""
    started = time.monotonic()
    try:
        phase()
    except Exception as e:
        log.exception(f"{name} phase failed")
        return e
    finally:
        log.info(f"{name} phase took {timedelta(seconds=time.monotonic() - started)}")
    return None
