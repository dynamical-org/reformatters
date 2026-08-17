"""Shared operational surface of the ECMWF IFS ENS extended range (S2S) datasets.

Each dataset retrieves the ECDS variables it needs into dynamical's GRIB archive with its
own `archive-grib-files` command, then reformats from that archive.
"""

import os
from collections.abc import Sequence
from typing import Annotated, Any, Final

import pandas as pd
import typer

from reformatters.common import kubernetes
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob
from reformatters.common.logging import get_logger
from reformatters.ecmwf.archive_gribs.archive import (
    DEFAULT_CONCURRENT_REQUESTS,
    archive_initialization,
)
from reformatters.ecmwf.archive_gribs.request_shards import initialization_selections
from reformatters.ecmwf.ifs_ens.s2s_config_models import EcmwfS2sDataVar
from reformatters.ecmwf.ifs_ens.s2s_region_job import (
    ARCHIVE_RCLONE_ROOT,
    EcmwfS2sSourceFileCoord,
)

log = get_logger(__name__)

# ECMWF's licence sets a 48 hour minimum delay, but ECDS publishes an initialization
# about 51.6 hours after its 00 UTC reference time (measured 51.4-52.1 h daily,
# 2026-06-26 to 2026-08-11). With the archive cron at 06 UTC, the initialization this
# selects is one published a couple of hours earlier.
PUBLICATION_DELAY = pd.Timedelta("53h")

SOURCE_COOP_SECRET_NAME: Final = "source-coop-storage-options-key"  # noqa: S105
ECDS_API_KEY_SECRET_NAME: Final = "ecmwf-ecds-api-key"  # noqa: S105


class EcmwfS2sDynamicalDataset(
    DynamicalDataset[EcmwfS2sDataVar, EcmwfS2sSourceFileCoord]
):
    def archive_grib_files(
        self,
        reformat_job_name: Annotated[str, typer.Argument(envvar="JOB_NAME")],
        # Typer does not handle PurePosixPath, so the rclone destination stays a str.
        dst_root_path: str = ARCHIVE_RCLONE_ROOT,
        init_times_back: int = 3,
        checkers: int = 32,
        concurrent_requests: int = DEFAULT_CONCURRENT_REQUESTS,
    ) -> None:
        """Retrieve this dataset's ECDS variables for recent initializations into the archive.

        Initializations are archived newest first, so an interrupted run leaves the most
        recent data archived.

        Args:
            dst_root_path: The destination root in the form rclone expects,
                e.g. ':s3:bucket/foo/bar'.
            init_times_back: How many initializations back from the newest available
                one to check. Already archived requests are skipped, so re-checking
                recent initializations is how an interrupted transfer resumes.
            checkers: Passed to `rclone --checkers` when listing the destination.
            concurrent_requests: How many ECDS requests to retrieve at once.
        """
        with self._monitor(
            CronJob,
            reformat_job_name,
            cron_job_name=f"{self.dataset_id}-archive-grib-files",
        ):
            _set_ecds_api_key_from_secret()
            selections = initialization_selections(self.ecds_variables())
            newest_init_time = (pd.Timestamp.now("UTC") - PUBLICATION_DELAY).normalize()
            init_times = pd.date_range(
                end=newest_init_time,
                periods=init_times_back,
                freq=self.template_config.append_dim_frequency,
            )[::-1]
            for init_time in init_times:
                if init_time.tz_localize(None) < self.template_config.append_dim_start:
                    continue
                log.info("Archiving %s", init_time)
                archive_initialization(
                    init_time.tz_localize(None),
                    selections,
                    dst_root_path,
                    checkers=checkers,
                    concurrent_requests=concurrent_requests,
                    env_vars=_source_coop_rclone_env_vars(),
                )

    def ecds_variables(self) -> Sequence[str]:
        return sorted(
            {
                data_var.internal_attrs.ecds_variable
                for data_var in self.template_config.data_vars
            }
        )

    def get_cli(self) -> typer.Typer:
        app = super().get_cli()
        app.command()(self.archive_grib_files)
        return app


def _set_ecds_api_key_from_secret() -> None:
    """Put the mounted ECDS key where `EcdsRequest` looks for it.

    Outside prod no secret is mounted and the client falls back to `~/.cdsapirc`.
    """
    secret = kubernetes.load_secret(ECDS_API_KEY_SECRET_NAME)
    if secret:
        os.environ["ECDS_API_KEY"] = secret["key"]


def _source_coop_rclone_env_vars() -> dict[str, Any] | None:
    secret = kubernetes.load_secret(SOURCE_COOP_SECRET_NAME)
    if not secret:
        return None
    return {
        "RCLONE_S3_PROVIDER": "AWS",
        "RCLONE_S3_ACCESS_KEY_ID": secret["key"],
        "RCLONE_S3_SECRET_ACCESS_KEY": secret["secret"],
        "RCLONE_S3_REGION": "us-west-2",
        "RCLONE_S3_FORCE_PATH_STYLE": "false",
    }
