import sys
from collections.abc import Sequence
from datetime import timedelta
from pathlib import PurePosixPath

import typer

from reformatters.common import kubernetes, validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob
from reformatters.dwd.copy_files_from_dwd_ftp import copy_files_from_dwd_ftp

from .region_job import DwdIconEuForecastRegionJob, DwdIconEuForecastSourceFileCoord
from .template_config import DwdIconEuDataVar, DwdIconEuForecastTemplateConfig


class DwdIconEuForecastDataset(
    DynamicalDataset[DwdIconEuDataVar, DwdIconEuForecastSourceFileCoord]
):
    template_config: DwdIconEuForecastTemplateConfig = DwdIconEuForecastTemplateConfig()
    region_job_class: type[DwdIconEuForecastRegionJob] = DwdIconEuForecastRegionJob

    # The `grib_archive_path` must be in the format that `rclone` expects: `:s3:<bucket>/<path>`.
    # Note that there is no double slash after `:s3:`.
    # The leading colon tells `rclone` to create an on the fly rclone backend and use the environment variables we set.
    grib_archive_path: str = ":s3:us-west-2.opendata.source.coop/dynamical/dwd-icon-grib/icon-eu/regular-lat-lon/"

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update and validate this dataset."""
        # TODO(Jack): Instantiate and return `ReformatCronJob` and `ValidationCronJob`. See example.
        suspend = True  # TODO(Jack): Remove when we're ready to run operationally!
        archive_grib_files_job = CronJob(
            command=["archive-grib-files"],
            workers_total=1,
            parallelism=1,
            name=f"{self.dataset_id}-archive-grib-files",
            # We want the 00, 06, 12, and 18 ICON-EU runs. DWD's transfer to their FTP server starts
            # about 2 hours 15 mins after the init time, and finishes about 3 hours 45 minutes after
            # the init time. So, to avoid copying incomplete files, we fetch the files 4 hours after
            # each init. But note that, every time the cron job runs, the script checks all 4 NWP
            # inits. This design helps to keep the code simple, especially when recovering if the
            # script hasn't run for a while. It only takes 4 minutes to check an NWP run that we've
            # already transferred.
            schedule="0 0 4,10,16,22 * *",
            # Copying 1 NWP run takes 45 minutes on a 1 Gbps internet connection. But, when this
            # script first runs, or if it hasn't run for >6 hours, then it'll transfer up to 4 NWP runs.
            pod_active_deadline=timedelta(hours=3),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="3",
            memory="7G",
            ephemeral_storage="30G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=suspend,
        )
        return [archive_grib_files_job]

    def validators(self) -> Sequence[validation.DataValidator]:
        """Return a sequence of DataValidators to run on this dataset."""
        # return (
        #     validation.check_analysis_current_data,
        #     validation.check_analysis_recent_nans,
        # )
        raise NotImplementedError(
            f"Implement `validators` on {self.__class__.__name__}"
        )

    def archive_grib_files(
        self,
        # It would've made more sense for `dst_root` to be a `PurePosixPath` but Typer doesn't
        # handle `PurePosixPath`, so we use a `str` to keep Typer happy.
        dst_root: str = grib_archive_path,
        # The `type: ignore` on the line below is because Typer doesn't understand the type hints
        # `tuple[int, ...]` or `Sequence[int]`, so we have to use `list[int]`.
        nwp_init_hours: list[int] = (0, 6, 12, 18),  # type: ignore[assignment]
        transfer_parallelism: int = 10,
        max_files_per_nwp_variable: int = sys.maxsize,
    ) -> None:
        """Restructure DWD GRIB files from FTP to a timestamped directory structure.

        Args:
            dst_root: The destination root directory. e.g. for S3, the dst_root could be: 's3:bucket/foo/bar'
            nwp_init_hours: All the ICON-EU NWP runs to transfer.
            transfer_parallelism: Number of parallel transfers. DWD appears to limit the number of parallel
                       transfers from one IP address to about 10.
            max_files_per_nwp_variable: Optional limit on the number of files to transfer per NWP variable.
                      This is useful for testing locally.
        """
        # When running in prod, `secret` will be {'key': 'xxx', 'secret': 'xxxx'}.
        # When not running in prod, `secret` will be empty.
        secret = kubernetes.load_secret("source-coop-storage-options-key")
        if secret:
            s3_credentials_env_vars_for_rclone = {
                "RCLONE_S3_PROVIDER": "AWS",
                "RCLONE_S3_ACCESS_KEY_ID": secret["key"],
                "RCLONE_S3_SECRET_ACCESS_KEY": secret["secret"],
                "RCLONE_S3_REGION": "us-west-2",
            }
        else:
            s3_credentials_env_vars_for_rclone = None

        for nwp_init_hour in nwp_init_hours:
            ftp_path = PurePosixPath(f"/weather/nwp/icon-eu/grib/{nwp_init_hour:02d}")
            copy_files_from_dwd_ftp(
                ftp_host="opendata.dwd.de",
                ftp_path=ftp_path,
                dst_root=PurePosixPath(dst_root),
                env_vars=s3_credentials_env_vars_for_rclone,
                transfer_parallelism=transfer_parallelism,
                max_files_per_nwp_variable=max_files_per_nwp_variable,
            )

    def get_cli(self) -> typer.Typer:
        """Create a CLI app with dataset commands."""
        app = super().get_cli()
        app.command()(self.archive_grib_files)
        return app
