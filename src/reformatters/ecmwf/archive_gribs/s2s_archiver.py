"""The operational surface of the ECMWF S2S GRIB archive.

The archive is upstream of every dataset built from it: it has its own bucket, its own
retrieval schedule and no store, so it deploys as its own cron rather than as part of a
dataset's operational resources. `ECDS_VARIABLES` is the archive's contract with those
datasets — what a reformatter reading this bucket can expect to find.
"""

import os
from collections.abc import Sequence
from datetime import timedelta
from typing import Annotated, Any, Final

import pandas as pd
import typer

from reformatters.common import kubernetes
from reformatters.common.kubernetes import CronJob
from reformatters.common.logging import get_logger
from reformatters.common.operational import OperationalResources
from reformatters.ecmwf.archive_gribs.archive import (
    DEFAULT_CONCURRENT_REQUESTS,
    archive_initialization,
)
from reformatters.ecmwf.archive_gribs.request_shards import initialization_selections

log = get_logger(__name__)

ARCHIVE_PREFIX: Final = "dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens-forecast-46-day"
ARCHIVE_RCLONE_ROOT: Final = f":s3:us-west-2.opendata.source.coop/{ARCHIVE_PREFIX}/"
ARCHIVE_BASE_URL: Final = f"https://s3-us-west-2.amazonaws.com/us-west-2.opendata.source.coop/{ARCHIVE_PREFIX}"

# ECMWF's licence sets a 48 hour minimum delay, but ECDS publishes an initialization
# about 51.6 hours after its 00 UTC reference time (measured 51.4-52.1 h daily,
# 2026-06-26 to 2026-08-11). With the archive cron at 06 UTC, the initialization this
# selects is one published a couple of hours earlier.
PUBLICATION_DELAY: Final = pd.Timedelta("53h")
# ECMWF S2S initializes at 00 UTC only.
INIT_FREQUENCY: Final = pd.Timedelta("1D")
EARLIEST_INIT_TIME: Final = pd.Timestamp("2023-06-28")

SOURCE_COOP_SECRET_NAME: Final = "source-coop-storage-options-key"  # noqa: S105
ECDS_API_KEY_SECRET_NAME: Final = "ecmwf-ecds-api-key"  # noqa: S105

# The ECDS variables archived for every initialization. A dataset reading this archive
# maps its data variables onto these names; adding one here is what makes it available
# to be mapped.
ECDS_VARIABLES: Final[Sequence[str]] = (
    "2_m_dewpoint_temperature",
    "2_m_temperature",
    "convective_precipitation",
    "eastward_turbulent_surface_stress",
    "geopotential_height",
    "mean_sea_level_pressure",
    "northward_turbulent_surface_stress",
    "sea_ice_area_fraction",
    "sea_surface_temperature",
    "skin_temperature",
    "snow_albedo",
    "snow_density",
    "snow_depth_water_equivalent",
    "snow_fall_water_equivalent",
    "soil_moisture_top_100_cm",
    "soil_moisture_top_20_cm",
    "soil_temperature_top_100_cm",
    "soil_temperature_top_20_cm",
    "specific_humidity",
    "surface_latent_heat_flux",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "surface_pressure",
    "surface_runoff",
    "surface_sensible_heat_flux",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "temperature",
    "top_net_thermal_radiation",
    "total_cloud_cover",
    "total_column_water",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "water_runoff_and_drainage",
)


class EcmwfS2sGribArchiver(OperationalResources):
    """Retrieves ECMWF S2S initializations from ECDS into dynamical's GRIB archive."""

    @property
    def dataset_id(self) -> str:
        return "ecmwf-s2s-gribs"

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        return [
            CronJob(
                command=["archive-grib-files"],
                workers_total=1,
                parallelism=1,
                name=f"{self.dataset_id}-archive-grib-files",
                schedule="0 6 * * *",
                pod_active_deadline=timedelta(hours=6),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1.5",
                memory="8G",
                ephemeral_storage="60G",
                secret_names=[SOURCE_COOP_SECRET_NAME, ECDS_API_KEY_SECRET_NAME],
            )
        ]

    def archive_grib_files(
        self,
        reformat_job_name: Annotated[str, typer.Argument(envvar="JOB_NAME")],
        # Typer does not handle PurePosixPath, so the rclone destination stays a str.
        dst_root_path: str = ARCHIVE_RCLONE_ROOT,
        init_times_back: int = 3,
        checkers: int = 32,
        concurrent_requests: int = DEFAULT_CONCURRENT_REQUESTS,
    ) -> None:
        """Retrieve `ECDS_VARIABLES` for recent initializations into the archive.

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
            selections = initialization_selections(ECDS_VARIABLES)
            for init_time in self.init_times_to_archive(init_times_back):
                log.info("Archiving %s", init_time)
                archive_initialization(
                    init_time,
                    selections,
                    dst_root_path,
                    checkers=checkers,
                    concurrent_requests=concurrent_requests,
                    env_vars=_source_coop_rclone_env_vars(),
                )

    def init_times_to_archive(
        self, init_times_back: int, now: pd.Timestamp | None = None
    ) -> Sequence[pd.Timestamp]:
        """The initializations one run checks, newest first."""
        now = now if now is not None else pd.Timestamp.now("UTC")
        newest_init_time = (now - PUBLICATION_DELAY).normalize().tz_localize(None)
        init_times = pd.date_range(
            end=newest_init_time, periods=init_times_back, freq=INIT_FREQUENCY
        )
        return [t for t in reversed(init_times) if t >= EARLIEST_INIT_TIME]

    def get_cli(self) -> typer.Typer:
        app = typer.Typer()
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
