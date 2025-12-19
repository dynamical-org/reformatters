import asyncio
from collections.abc import Sequence
from datetime import timedelta
from pathlib import PurePosixPath
from typing import Literal

import typer
from obstore.store import LocalStore, ObjectStore, S3Store

from reformatters.common import kubernetes, validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.ftp_to_obstore import copy_files_from_ftp_to_obstore
from reformatters.common.ftp_transfer_coordinator import FtpTransferCoordinator
from reformatters.common.kubernetes import ArchiveGribFilesCronJob, CronJob
from reformatters.common.logging import get_logger
from reformatters.common.object_storage_info_manager import ObjectStorageInfoManager

from .archive_grib_files import DwdFtpInfoExtractor
from .region_job import DwdIconEuForecastRegionJob, DwdIconEuForecastSourceFileCoord
from .template_config import DwdIconEuDataVar, DwdIconEuForecastTemplateConfig

log = get_logger(__name__)


class DwdIconEuForecastDataset(
    DynamicalDataset[DwdIconEuDataVar, DwdIconEuForecastSourceFileCoord]
):
    template_config: DwdIconEuForecastTemplateConfig = DwdIconEuForecastTemplateConfig()
    region_job_class: type[DwdIconEuForecastRegionJob] = DwdIconEuForecastRegionJob

    def archive_grib_files(
        self,
        nwp_init_hour: Literal["all", "0", "6", "12", "18"] = "all",
        filename_filter: str = "",
        local_dst_root_path: str | None = None,
    ) -> None:
        """
        Args:
            nwp_init_hour: The NWP initialisation hour to archive.
            filename_filter: An optional regex pattern to filter filenames by.
                For example, to only download single-level files, for forecast steps 0 to 5
                then use a regex pattern like "single-level_.*_00[0-5]_".
            local_dst_root_path: If set, then the GRIB files will be saved to this local path.
        """
        if local_dst_root_path:
            dst_root_path = PurePosixPath(local_dst_root_path)
            store: ObjectStore = LocalStore()
        else:
            dst_root_path = PurePosixPath(
                "dynamical/dwd-icon-grib/icon-eu/regular-lat-lon/"
            )
            store = get_source_coop_s3_store()

        ftp_info_extractor = DwdFtpInfoExtractor(ftp_host="opendata.dwd.de")
        obstore_info_manager = ObjectStorageInfoManager(
            dst_obstore=store, dst_root_path=dst_root_path
        )
        coordinator = FtpTransferCoordinator(
            ftp_info_extractor=ftp_info_extractor,
            obstore_info_manager=obstore_info_manager,
            filename_filter=filename_filter,
        )

        if nwp_init_hour == "all":
            files_to_download = asyncio.run(
                coordinator.calc_new_files_for_multiple_nwp_init_hours()
            )
        else:
            init_hour = int(nwp_init_hour)
            files_to_download = asyncio.run(
                coordinator.calc_new_files_for_single_nwp_init_hour(init_hour)
            )

        src_ftp_paths: list[PurePosixPath] = []
        dst_obstore_paths: list[str] = []
        for transfer_job in files_to_download:
            src_ftp_paths.append(transfer_job.src_ftp_path)
            dst_obstore_paths.append(str(transfer_job.dst_obstore_path))

        asyncio.run(
            copy_files_from_ftp_to_obstore(
                ftp_host=ftp_info_extractor.ftp_host,
                src_ftp_paths=src_ftp_paths,
                dst_obstore_paths=dst_obstore_paths,
                dst_store=store,
            )
        )

        log.info(
            "Finished downloading %d files from ftp://%s",
            len(files_to_download),
            ftp_info_extractor.ftp_host,
        )

    def get_cli(self) -> typer.Typer:
        """Create a CLI app with dataset commands."""
        app = super().get_cli()
        app.command()(self.archive_grib_files)
        return app

    def validators(self) -> Sequence[validation.DataValidator]:
        """Return a sequence of DataValidators to run on this dataset."""
        # return (
        #     validation.check_analysis_current_data,
        #     validation.check_analysis_recent_nans,
        # )
        return ()  # TODO(Jack): Return appropriate `validation.check_*`

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update
        and validate this dataset."""
        suspend = True  # Defaults to False, remove when you're ready to run operational updates and validation
        ftp_to_obstore_cron_job = ArchiveGribFilesCronJob(
            name=f"{self.dataset_id}-archive-grib-files",
            # The files for each NWP init are ready 4 hours after the NWP init time. For example,
            # the 00Z NWP init is ready at 4am. So let's run 4 times per day, starting at 4am.
            schedule="0 4,10,16,22 * * *",
            pod_active_deadline=timedelta(
                minutes=120
            ),  # TODO(Jack): Check how long it *actually* takes!
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="14",
            memory="30G",
            shared_memory="12G",
            ephemeral_storage="30G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=suspend,
        )

        # TODO(Jack): Implement ValidationCronJob, ReformatCronJob, and return these

        return [ftp_to_obstore_cron_job]


def get_source_coop_s3_store() -> S3Store:
    secret = kubernetes.load_secret("source-coop-storage-options-key")
    bucket = "us-west-2.opendata.source.coop"
    region = "us-west-2"

    # When running in prod, `secret` will be {'key': 'xxx', 'secret': 'xxxx'}.
    # When not running in prod, `secret` will be empty. `S3Store()` won't
    # accept `None` for `access_key_id` or `secret_access_key`.
    if secret:
        return S3Store(
            bucket=bucket,
            region=region,
            access_key_id=secret["key"],
            secret_access_key=secret["secret"],
        )
    else:
        return S3Store(bucket=bucket, region=region)
