"""The operational surface of the ECMWF IFS ENS 46-day GRIB archive.

The archive is upstream of every dataset built from it: it has its own bucket, its own
retrieval schedule and no store, so it deploys as its own cron rather than as part of a
dataset's operational resources. `ECDS_VARIABLES` is the archive's contract with those
datasets — what a reformatter reading this bucket can expect to find.
"""

import os
from collections.abc import Sequence
from datetime import timedelta
from typing import Annotated, Any, Final, Literal

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
# 2026-06-26 to 2026-08-11, i.e. 03:24-04:06 UTC two days later). The newest
# initialization a run considers is `now - PUBLICATION_DELAY`, so 51 h makes the
# two-day-old initialization a candidate from 03:00 UTC; a fire before ECDS has
# published it makes only the cheap availability probe and skips it. The delay
# stays above the licence minimum; it selects candidates, it does not gate access.
PUBLICATION_DELAY: Final = pd.Timedelta("51h")
# One fire shortly before the observed publication window. The run then probes
# ECDS every PUBLICATION_POLL_INTERVAL (cheap, unauthenticated, no retrieval job) for
# up to PUBLICATION_WAIT and retrieves as soon as the initialization is published,
# instead of a fixed fire that trails publication by up to a day. A transfer took
# 67-102 minutes on 2026-09-19 to 09-23; the wait plus a transfer fits the pod's
# 6 hour deadline. The 06:15 fire is a second, independent attempt on a day the
# first run's wait ran out or the job failed, close to the previous single 06:00
# fire. The wait ends by ~05:50 (2 h 30 min after a start that follows pod startup
# and the archive listing) so a run that only waited has exited before 06:15;
# a run still transferring makes concurrency_policy "Forbid" skip the 06:15
# fire instead of replacing the run and abandoning its in-flight retrievals.
ARCHIVE_CRON_SCHEDULE: Final = "15 3,6 * * *"
PUBLICATION_WAIT: Final = pd.Timedelta(hours=2, minutes=30)
PUBLICATION_POLL_INTERVAL: Final = pd.Timedelta(minutes=5)
# ECMWF IFS ENS 46-day initializes at 00 UTC only.
INIT_FREQUENCY: Final = pd.Timedelta("1D")
EARLIEST_INIT_TIME: Final = pd.Timestamp("2023-06-28")

SOURCE_COOP_SECRET_NAME: Final = "source-coop-storage-options-key"  # noqa: S105
ECDS_API_KEY_SECRET_NAME: Final = "ecmwf-ecds-api-key"  # noqa: S105

type MaterializedProductFrequency = Literal["6-hourly", "daily"]

MATERIALIZED_PRODUCT_ECDS_VARIABLES: Final[
    dict[MaterializedProductFrequency, tuple[str, ...]]
] = {
    "6-hourly": (
        "10_m_u_component_of_wind",
        "10_m_v_component_of_wind",
        "maximum_2_m_temperature_in_the_last_6_hours",
        "minimum_2_m_temperature_in_the_last_6_hours",
        "total_precipitation",
    ),
    "daily": (
        "10_m_u_component_of_wind",
        "10_m_v_component_of_wind",
        "2_m_dewpoint_temperature",
        "2_m_temperature",
        "convective_available_potential_energy",
        "convective_precipitation",
        "eastward_turbulent_surface_stress",
        "geopotential_height",
        "maximum_2_m_temperature_in_the_last_6_hours",
        "mean_sea_level_pressure",
        "minimum_2_m_temperature_in_the_last_6_hours",
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
        "total_precipitation",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
        "water_runoff_and_drainage",
    ),
}

ECDS_VARIABLES: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            variable
            for product_variables in MATERIALIZED_PRODUCT_ECDS_VARIABLES.values()
            for variable in product_variables
        }
    )
)


class EcmwfIfsEns46DayGribArchiver(OperationalResources):
    """Retrieves ECMWF IFS ENS 46-day initializations from ECDS into dynamical's GRIB archive."""

    @property
    def dataset_id(self) -> str:
        return "ecmwf-ifs-ens-46-day-gribs"

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        return [
            CronJob(
                command=["archive-grib-files"],
                workers_total=1,
                parallelism=1,
                name=f"{self.dataset_id}-archive-grib-files",
                schedule=ARCHIVE_CRON_SCHEDULE,
                concurrency_policy="Forbid",
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
        wait_for_publication_minutes: int = int(PUBLICATION_WAIT.total_seconds() // 60),
        publication_poll_minutes: int = int(
            PUBLICATION_POLL_INTERVAL.total_seconds() // 60
        ),
    ) -> None:
        """Retrieve `ECDS_VARIABLES` for recent initializations into the archive.

        Initializations are archived newest first, so an interrupted run leaves the most
        recent data archived. The newest initialization is waited for (see
        `wait_for_publication_minutes`); older ones are skipped if unpublished, as before.

        Args:
            dst_root_path: The destination root in the form rclone expects,
                e.g. ':s3:bucket/foo/bar'.
            init_times_back: How many initializations back from the newest available
                one to check. Already archived requests are skipped, so re-checking
                recent initializations is how an interrupted transfer resumes.
            checkers: Passed to `rclone --checkers` when listing the destination.
            concurrent_requests: How many ECDS requests to retrieve at once.
            wait_for_publication_minutes: How long to keep probing ECDS for the newest
                initialization when it is not published yet; 0 skips it immediately.
            publication_poll_minutes: Minutes between those probes.
        """
        with self._monitor(
            CronJob,
            reformat_job_name,
            cron_job_name=f"{self.dataset_id}-archive-grib-files",
        ):
            _set_ecds_api_key_from_secret()
            selections = initialization_selections(ECDS_VARIABLES)
            for i, init_time in enumerate(self.init_times_to_archive(init_times_back)):
                log.info("Archiving %s", init_time)
                archive_initialization(
                    init_time,
                    selections,
                    dst_root_path,
                    checkers=checkers,
                    concurrent_requests=concurrent_requests,
                    env_vars=_source_coop_rclone_env_vars(),
                    wait_for_publication=pd.Timedelta(
                        minutes=wait_for_publication_minutes if i == 0 else 0
                    ),
                    publication_poll_interval=pd.Timedelta(
                        minutes=publication_poll_minutes
                    ),
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
