from collections.abc import Sequence
from datetime import timedelta
from typing import Annotated, ClassVar

import typer

from reformatters.common import kubernetes, validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob
from reformatters.eccc.hrdps.archive_gribs.copy_files_from_eccc import (
    copy_files_from_eccc_https,
)

from .region_job import (
    EcccHrdpsForecastTemporalRegionJob,
    EcccHrdpsForecastTemporalSourceFileCoord,
)
from .template_config import EcccHrdpsDataVar, EcccHrdpsForecastTemporalTemplateConfig


class EcccHrdpsForecastTemporalDynamicalDataset(
    DynamicalDataset[EcccHrdpsDataVar, EcccHrdpsForecastTemporalSourceFileCoord]
):
    """Not yet implemented."""

    template_config: EcccHrdpsForecastTemporalTemplateConfig = (
        EcccHrdpsForecastTemporalTemplateConfig()
    )
    region_job_class: type[EcccHrdpsForecastTemporalRegionJob] = (
        EcccHrdpsForecastTemporalRegionJob
    )

    # Must be in the format `rclone` expects: `:s3:<bucket>/<path>`. No double slash
    # after `:s3:` - the leading colon tells `rclone` to create an on-the-fly remote
    # from the env vars we set.
    dynamical_grib_archive_rclone_root: ClassVar[str] = (
        ":s3:us-west-2.opendata.source.coop/dynamical/eccc-hrdps-grib/"
    )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""
        archive_grib_files_job = CronJob(
            command=["archive-grib-files"],
            workers_total=1,
            parallelism=1,
            name=f"{self.dataset_id}-archive-grib-files",
            # HRDPS runs at 00, 06, 12, 18 UTC; its full 48h run is published by
            # ~init+3h49m (p99). Schedule 4h after each init, and recheck all 4
            # daily inits every run so a missed or slow run gets caught up automatically.
            schedule="0 4,10,16,22 * * *",
            pod_active_deadline=timedelta(hours=2),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1",
            memory="4G",
            ephemeral_storage="1G",  # not used
            secret_names=["source-coop-storage-options-key"],
        )

        return [archive_grib_files_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        """Return a sequence of DataValidators to run on this dataset."""
        raise NotImplementedError(
            f"Implement `validators` on {self.__class__.__name__}"
        )

    def archive_grib_files(
        self,
        reformat_job_name: Annotated[str, typer.Argument(envvar="JOB_NAME")],
        # It would've made more sense for `dst_root_path` to be a `PurePosixPath` but
        # Typer doesn't handle `PurePosixPath`, so we use a `str` to keep Typer happy.
        dst_root_path: str = dynamical_grib_archive_rclone_root,
        # The `ty: ignore` on the line below is because Typer doesn't understand the
        # type hints `tuple[int, ...]` or `Sequence[int]`, so we have to use `list[int]`.
        nwp_init_hours: list[int] = (0, 6, 12, 18),  # ty: ignore[invalid-parameter-default]
        days_back: int = 1,
        transfer_parallelism: int = 32,
        checkers: int = 16,
        stats_logging_freq: str = "1m",
    ) -> None:
        """Copy new HRDPS continental GRIB2 files from the MSC Datamart to Source Co-Op.

        Args:
            dst_root_path: The destination root directory. e.g. for S3: ':s3:bucket/foo/bar/'
            nwp_init_hours: The HRDPS NWP model runs to transfer.
            days_back: How many additional UTC calendar days before today to recheck.
            transfer_parallelism: Number of concurrent workers during the copy operation.
                Each worker fetches a file from the Datamart and writes it to the
                destination, waiting for the destination to acknowledge completion
                before fetching another file.
            checkers: This number is passed to the `rclone --checkers` argument.
            stats_logging_freq: The period between each stats log. e.g. "1m" to log stats every minute.
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

            copy_files_from_eccc_https(
                dst_root_path=dst_root_path,
                nwp_init_hours=nwp_init_hours,
                days_back=days_back,
                transfer_parallelism=transfer_parallelism,
                checkers=checkers,
                stats_logging_freq=stats_logging_freq,
                env_vars=s3_credentials_env_vars_for_rclone,
            )

    def get_cli(self) -> typer.Typer:
        """Create a CLI app with dataset commands."""
        app = super().get_cli()
        app.command()(self.archive_grib_files)
        return app
