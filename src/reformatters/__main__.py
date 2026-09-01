import contextlib
import faulthandler
import multiprocessing
import os
from collections.abc import Sequence
from typing import Any

# Spawn new processes since fork isn't safe with threads
with contextlib.suppress(RuntimeError):  # skip if already set
    multiprocessing.set_start_method("spawn", force=True)

import sentry_sdk
import typer
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.typer import TyperIntegration
from sentry_sdk.types import Hint, Log

from reformatters.common import deploy as deploy_module
from reformatters.common import monitoring
from reformatters.common.config import Config
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.initialize_new_integration import initialize_new_integration
from reformatters.common.operational import (
    OperationalResources,
    register_run_monitor,
)
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.contrib.nasa.smap.level3_36km_v9 import NasaSmapLevel336KmV9Dataset
from reformatters.contrib.noaa.ndvi_cdr.analysis import (
    NoaaNdviCdrAnalysisDataset,
)
from reformatters.contrib.uarizona.swann.analysis import UarizonaSwannAnalysisDataset
from reformatters.dwd.icon_eu.forecast_5_day import DwdIconEuForecast5DayDataset
from reformatters.eccc.hrdps.forecast import EcccHrdpsForecastDataset
from reformatters.ecmwf.aifs_ens.forecast import (
    EcmwfAifsEnsForecastDataset,
)
from reformatters.ecmwf.aifs_single.forecast import (
    EcmwfAifsSingleForecastDataset,
)
from reformatters.ecmwf.aifs_single.forecast_virtual import (
    EcmwfAifsSingleForecastVirtualDataset,
)
from reformatters.ecmwf.archive_gribs.forecast_46_day_archiver import (
    EcmwfIfsEns46DayGribArchiver,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day025DegreeDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_1_5_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast46Day15DegreeDataset,
)
from reformatters.google.weathernext2.forecast_historical_virtual import (
    GoogleWeathernext2ForecastHistoricalVirtualDataset,
)
from reformatters.google.weathernext2.forecast_operational_virtual import (
    GoogleWeathernext2ForecastOperationalVirtualDataset,
)
from reformatters.nasa.imerg.analysis_early import NasaImergAnalysisEarlyDataset
from reformatters.nasa.imerg.analysis_late import NasaImergAnalysisLateDataset
from reformatters.noaa.gefs.analysis.dynamical_dataset import GefsAnalysisDataset
from reformatters.noaa.gefs.analysis_0_25_degree_virtual.dynamical_dataset import (
    NoaaGefsAnalysis025DegreeVirtualDataset,
)
from reformatters.noaa.gefs.forecast_35_day.dynamical_dataset import (
    GefsForecast35DayDataset,
)
from reformatters.noaa.gfs.analysis import NoaaGfsAnalysisDataset
from reformatters.noaa.gfs.analysis_virtual import NoaaGfsAnalysisVirtualDataset
from reformatters.noaa.gfs.forecast import NoaaGfsForecastDataset
from reformatters.noaa.gfs.forecast_virtual import NoaaGfsForecastVirtualDataset
from reformatters.noaa.hrrr.analysis.dynamical_dataset import (
    NoaaHrrrAnalysisDataset,
)
from reformatters.noaa.hrrr.analysis_virtual.dynamical_dataset import (
    NoaaHrrrAnalysisVirtualDataset,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual.dynamical_dataset import (
    NoaaHrrrForecast18HourVirtualDataset,
)
from reformatters.noaa.hrrr.forecast_48_hour.dynamical_dataset import (
    NoaaHrrrForecast48HourDataset,
)
from reformatters.noaa.hrrr.forecast_48_hour_virtual.dynamical_dataset import (
    NoaaHrrrForecast48HourVirtualDataset,
)
from reformatters.noaa.mrms.conus_analysis_hourly.dynamical_dataset import (
    NoaaMrmsConusAnalysisHourlyDataset,
)

faulthandler.enable()


class NoaaHrrrIcechunkAwsOpenDataDatasetStorageConfig(StorageConfig):
    """Configuration for the storage of a AWS Open Data dataset."""

    base_path: str = "s3://dynamical-noaa-hrrr"
    k8s_secret_name: str = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK


class NoaaGfsIcechunkAwsOpenDataDatasetStorageConfig(StorageConfig):
    """NOAA GFS in Icechunk on AWS Open Data."""

    base_path: str = "s3://dynamical-noaa-gfs"
    k8s_secret_name: str = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK


class NoaaGefsIcechunkAwsOpenDataDatasetStorageConfig(StorageConfig):
    """NOAA GEFS in Icechunk on AWS Open Data."""

    base_path: str = "s3://dynamical-noaa-gefs"
    k8s_secret_name: str = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK


class EcmwfIfsEnsIcechunkAwsOpenDataDatasetStorageConfig(StorageConfig):
    """ECMWF IFS Ens in Icechunk on AWS Open Data."""

    base_path: str = "s3://dynamical-ecmwf-ifs-ens"
    k8s_secret_name: str = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK


class EcmwfAifsSingleIcechunkAwsOpenDataDatasetStorageConfig(StorageConfig):
    """ECMWF AIFS Single in Icechunk on AWS Open Data."""

    base_path: str = "s3://dynamical-ecmwf-aifs-single"
    k8s_secret_name: str = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK


class EcmwfAifsEnsIcechunkAwsOpenDataDatasetStorageConfig(StorageConfig):
    """ECMWF AIFS ENS in Icechunk on AWS Open Data."""

    base_path: str = "s3://dynamical-ecmwf-aifs-ens"
    k8s_secret_name: str = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK


class NoaaMrmsIcechunkAwsOpenDataDatasetStorageConfig(StorageConfig):
    """NOAA MRMS in Icechunk on AWS Open Data."""

    base_path: str = "s3://dynamical-noaa-mrms"
    k8s_secret_name: str = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK


class DwdIconEuIcechunkAwsOpenDataDatasetStorageConfig(StorageConfig):
    """DWD ICON-EU in Icechunk on AWS Open Data."""

    base_path: str = "s3://dynamical-dwd-icon-eu"
    k8s_secret_name: str = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK


class NasaImergIcechunkAwsOpenDataDatasetStorageConfig(StorageConfig):
    """NASA IMERG on AWS Open Data."""

    base_path: str = "s3://dynamical-nasa-imerg"
    k8s_secret_name: str = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK


class EcccHrdpsIcechunkAwsOpenDataDatasetStorageConfig(StorageConfig):
    """ECCC HRDPS in Icechunk on AWS Open Data."""

    base_path: str = "s3://dynamical-eccc-hrdps"
    k8s_secret_name: str = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK


class SourceCoopZarrDatasetStorageConfig(StorageConfig):
    """Configuration for the storage of a SourceCoop dataset."""

    base_path: str = "s3://us-west-2.opendata.source.coop/dynamical"
    k8s_secret_name: str = "source-coop-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ZARR3


class UpstreamGriddedZarrsDatasetStorageConfig(StorageConfig):
    """Configuration for storage in the Upstream gridded zarrs bucket."""

    # This bucket is actually an R2 bucket.
    # The R2 endpoint URL is stored within our k8s secret and will be set
    # when it's imported into the env.
    base_path: str = "s3://upstream-gridded-zarrs"
    k8s_secret_name: str = "upstream-gridded-zarrs-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ZARR3


class Weathernext2IcechunkDatasetStorageConfig(StorageConfig):
    """Icechunk storage in the private WeatherNext 2 bucket.

    This is an R2 bucket. Its endpoint URL and `force_path_style` come from the
    k8s secret, which holds `icechunk.s3_storage` kwargs.
    """

    base_path: str = "s3://dynamical-google-weathernext2"
    k8s_secret_name: str = "weathernext2-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK


# Registry of all DynamicalDatasets.
DYNAMICAL_DATASETS: Sequence[DynamicalDataset[Any, Any]] = [
    # NOAA
    NoaaGfsForecastDataset(
        primary_storage_config=NoaaGfsIcechunkAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[SourceCoopZarrDatasetStorageConfig()],
    ),
    NoaaGfsAnalysisDataset(
        primary_storage_config=NoaaGfsIcechunkAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[SourceCoopZarrDatasetStorageConfig()],
    ),
    NoaaGfsAnalysisVirtualDataset(
        primary_storage_config=NoaaGfsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaGfsForecastVirtualDataset(
        primary_storage_config=NoaaGfsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    GefsAnalysisDataset(
        primary_storage_config=NoaaGefsIcechunkAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[SourceCoopZarrDatasetStorageConfig()],
    ),
    GefsForecast35DayDataset(
        primary_storage_config=NoaaGefsIcechunkAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[SourceCoopZarrDatasetStorageConfig()],
    ),
    NoaaGefsAnalysis025DegreeVirtualDataset(
        primary_storage_config=NoaaGefsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaHrrrForecast48HourDataset(
        primary_storage_config=NoaaHrrrIcechunkAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[SourceCoopZarrDatasetStorageConfig()],
    ),
    NoaaHrrrAnalysisDataset(
        primary_storage_config=NoaaHrrrIcechunkAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[SourceCoopZarrDatasetStorageConfig()],
    ),
    NoaaHrrrAnalysisVirtualDataset(
        primary_storage_config=NoaaHrrrIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaHrrrForecast48HourVirtualDataset(
        primary_storage_config=NoaaHrrrIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaHrrrForecast18HourVirtualDataset(
        primary_storage_config=NoaaHrrrIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    NoaaMrmsConusAnalysisHourlyDataset(
        primary_storage_config=NoaaMrmsIcechunkAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[SourceCoopZarrDatasetStorageConfig()],
    ),
    # ECMWF
    EcmwfIfsEnsForecast15Day025DegreeDataset(
        primary_storage_config=EcmwfIfsEnsIcechunkAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[SourceCoopZarrDatasetStorageConfig()],
    ),
    EcmwfIfsEnsForecast46Day15DegreeDataset(
        primary_storage_config=EcmwfIfsEnsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    EcmwfAifsSingleForecastDataset(
        primary_storage_config=EcmwfAifsSingleIcechunkAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[SourceCoopZarrDatasetStorageConfig()],
    ),
    EcmwfAifsSingleForecastVirtualDataset(
        primary_storage_config=EcmwfAifsSingleIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    EcmwfAifsEnsForecastDataset(
        primary_storage_config=EcmwfAifsEnsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    # DWD
    DwdIconEuForecast5DayDataset(
        primary_storage_config=DwdIconEuIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    # ECCC
    EcccHrdpsForecastDataset(
        primary_storage_config=EcccHrdpsIcechunkAwsOpenDataDatasetStorageConfig(),
    ),
    # Google
    GoogleWeathernext2ForecastHistoricalVirtualDataset(
        primary_storage_config=Weathernext2IcechunkDatasetStorageConfig(),
    ),
    GoogleWeathernext2ForecastOperationalVirtualDataset(
        primary_storage_config=Weathernext2IcechunkDatasetStorageConfig(),
    ),
    # NASA
    NasaImergAnalysisEarlyDataset(
        primary_storage_config=NasaImergIcechunkAwsOpenDataDatasetStorageConfig()
    ),
    NasaImergAnalysisLateDataset(
        primary_storage_config=NasaImergIcechunkAwsOpenDataDatasetStorageConfig()
    ),
    # Contrib
    UarizonaSwannAnalysisDataset(
        primary_storage_config=UpstreamGriddedZarrsDatasetStorageConfig()
    ),
    NoaaNdviCdrAnalysisDataset(
        primary_storage_config=UpstreamGriddedZarrsDatasetStorageConfig()
    ),
    NasaSmapLevel336KmV9Dataset(
        primary_storage_config=UpstreamGriddedZarrsDatasetStorageConfig()
    ),
]

register_run_monitor(monitoring.monitor_cron)

if Config.is_sentry_enabled:
    cron_job_name = os.getenv("CRON_JOB_NAME")
    job_name = os.getenv("JOB_NAME")
    pod_name = os.getenv("POD_NAME")

    def before_log(log: Log, _hint: Hint) -> Log | None:
        if cron_job_name:
            log["attributes"]["cron_job_name"] = cron_job_name
        if job_name:
            log["attributes"]["job_name"] = job_name
        if pod_name:
            log["attributes"]["pod_name"] = pod_name
        return log

    sentry_sdk.init(
        dsn=Config.sentry_dsn,
        environment=Config.env.value,
        project_root="src/",
        in_app_include=["reformatters"],
        default_integrations=True,
        # Connection idles cause us to lose events after quiet periods
        keep_alive=True,
        before_send_log=before_log,
        integrations=[
            LoggingIntegration(capture_sentry_logs=True),
            TyperIntegration(),
        ],
    )
    sentry_sdk.set_tag("env", Config.env.value)
    sentry_sdk.set_tag("cron_job_name", cron_job_name)
    sentry_sdk.set_tag("job_name", job_name)
    sentry_sdk.set_tag("pod_name", pod_name)


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.callback()
def startup() -> None:
    monitoring.install_sigterm_logger()


app.command()(initialize_new_integration)


# Source archives that feed a dataset but have no store of their own. They deploy
# their own cronjobs and are not datasets, so they carry no update/validate/backfill.
OPERATIONAL_ARCHIVERS: Sequence[OperationalResources] = [EcmwfIfsEns46DayGribArchiver()]

for dataset in DYNAMICAL_DATASETS:
    app.add_typer(dataset.get_cli(), name=dataset.dataset_id)

for archiver in OPERATIONAL_ARCHIVERS:
    app.add_typer(archiver.get_cli(), name=archiver.dataset_id)

deploy_module.register_commands(app, DYNAMICAL_DATASETS, OPERATIONAL_ARCHIVERS)


if not __debug__:
    raise RuntimeError(
        "This project relies on assert statements. Do not run with python -O."
    )

if __name__ == "__main__":
    app()
