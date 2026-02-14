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
from sentry_sdk.integrations.typer import TyperIntegration
from sentry_sdk.types import Hint, Log

from reformatters.common import deploy as deploy_module
from reformatters.common.config import Config
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.contrib.nasa.smap.level3_36km_v9 import NasaSmapLevel336KmV9Dataset
from reformatters.contrib.noaa.ndvi_cdr.analysis import (
    NoaaNdviCdrAnalysisDataset,
)
from reformatters.contrib.uarizona.swann.analysis import UarizonaSwannAnalysisDataset
from reformatters.dwd.icon_eu.forecast import DwdIconEuForecastDataset
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day025DegreeDataset,
)
from reformatters.example.new_dataset import initialize_new_integration
from reformatters.noaa.gefs.analysis.dynamical_dataset import GefsAnalysisDataset
from reformatters.noaa.gefs.forecast_35_day.dynamical_dataset import (
    GefsForecast35DayDataset,
)
from reformatters.noaa.gfs.analysis import NoaaGfsAnalysisDataset
from reformatters.noaa.gfs.forecast import NoaaGfsForecastDataset
from reformatters.noaa.hrrr.analysis.dynamical_dataset import (
    NoaaHrrrAnalysisDataset,
)
from reformatters.noaa.hrrr.forecast_48_hour.dynamical_dataset import (
    NoaaHrrrForecast48HourDataset,
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


# Registry of all DynamicalDatasets.
# Datasets that have not yet been ported over to the new DynamicalDataset pattern
# are excluded here until they are refactored.
DYNAMICAL_DATASETS: Sequence[DynamicalDataset[Any, Any]] = [
    # NOAA
    NoaaGfsForecastDataset(
        primary_storage_config=SourceCoopZarrDatasetStorageConfig(),
        replica_storage_configs=[NoaaGfsIcechunkAwsOpenDataDatasetStorageConfig()],
    ),
    NoaaGfsAnalysisDataset(
        primary_storage_config=SourceCoopZarrDatasetStorageConfig(),
        replica_storage_configs=[NoaaGfsIcechunkAwsOpenDataDatasetStorageConfig()],
    ),
    GefsAnalysisDataset(
        primary_storage_config=SourceCoopZarrDatasetStorageConfig(),
        replica_storage_configs=[NoaaGefsIcechunkAwsOpenDataDatasetStorageConfig()],
    ),
    GefsForecast35DayDataset(
        primary_storage_config=SourceCoopZarrDatasetStorageConfig(),
        replica_storage_configs=[NoaaGefsIcechunkAwsOpenDataDatasetStorageConfig()],
    ),
    NoaaHrrrForecast48HourDataset(
        primary_storage_config=SourceCoopZarrDatasetStorageConfig(),
        replica_storage_configs=[NoaaHrrrIcechunkAwsOpenDataDatasetStorageConfig()],
    ),
    NoaaHrrrAnalysisDataset(
        primary_storage_config=SourceCoopZarrDatasetStorageConfig(),
        replica_storage_configs=[NoaaHrrrIcechunkAwsOpenDataDatasetStorageConfig()],
    ),
    # ECMWF
    EcmwfIfsEnsForecast15Day025DegreeDataset(
        primary_storage_config=SourceCoopZarrDatasetStorageConfig(),
        replica_storage_configs=[EcmwfIfsEnsIcechunkAwsOpenDataDatasetStorageConfig()],
    ),
    # DWD
    DwdIconEuForecastDataset(
        primary_storage_config=SourceCoopZarrDatasetStorageConfig()
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
        enable_logs=True,
        before_send_log=before_log,
        integrations=[
            TyperIntegration(),
        ],
    )
    sentry_sdk.set_tag("env", Config.env.value)
    sentry_sdk.set_tag("cron_job_name", cron_job_name)
    sentry_sdk.set_tag("job_name", job_name)
    sentry_sdk.set_tag("pod_name", pod_name)


app = typer.Typer(pretty_exceptions_show_locals=False)
app.command()(initialize_new_integration)


for dataset in DYNAMICAL_DATASETS:
    app.add_typer(dataset.get_cli(), name=dataset.dataset_id)


@app.command()
def deploy(
    docker_image: str | None = None,
) -> None:
    deploy_module.deploy_operational_resources(DYNAMICAL_DATASETS, docker_image)


if not __debug__:
    raise RuntimeError("This project relies on assert statements. Do not run with python -O.")

if __name__ == "__main__":
    app()
