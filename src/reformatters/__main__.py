import multiprocessing
from collections.abc import Sequence
from typing import Any

# Spawn new processes since fork isn't safe with threads
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # Already set, ignore
    pass


import sentry_sdk
import typer
from sentry_sdk.integrations.typer import TyperIntegration

from reformatters.common import deploy
from reformatters.common.config import Config
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.contrib.noaa.ndvi_cdr.analysis import (
    NoaaNdviCdrAnalysisDataset,
)
from reformatters.contrib.uarizona.swann.analysis import UarizonaSwannAnalysisDataset
from reformatters.dwd.icon_eu.forecast import DwdIconEuForecastDataset
from reformatters.example.new_dataset import initialize_new_integration
from reformatters.noaa.gefs.analysis.dynamical_dataset import GefsAnalysisDataset
from reformatters.noaa.gefs.forecast_35_day.dynamical_dataset import (
    GefsForecast35DayDataset,
)
from reformatters.noaa.gfs.forecast import NoaaGfsForecastDataset
from reformatters.noaa.hrrr.forecast_48_hour.dynamical_dataset import (
    NoaaHrrrForecast48HourDataset,
)


class NoaaHrrrAwsOpenDataDatasetStorageConfig(StorageConfig):
    """Configuration for the storage of a AWS Open Data dataset."""

    base_path: str = "s3://dynamical-noaa-hrrr"
    k8s_secret_name: str = "aws-open-data-storage-options-key"  # noqa:S105
    format: DatasetFormat = DatasetFormat.ZARR3


class NoaaGfsAwsOpenDataDatasetStorageConfig(StorageConfig):
    """NOAA GFS in Icechunk on AWS Open Data."""

    base_path: str = "s3://dynamical-noaa-gfs"
    k8s_secret_name: str = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK


class SourceCoopDatasetStorageConfig(StorageConfig):
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
    UarizonaSwannAnalysisDataset(
        primary_storage_config=UpstreamGriddedZarrsDatasetStorageConfig()
    ),
    NoaaNdviCdrAnalysisDataset(
        primary_storage_config=UpstreamGriddedZarrsDatasetStorageConfig()
    ),
    NoaaGfsForecastDataset(
        primary_storage_config=SourceCoopDatasetStorageConfig(),
        replica_storage_configs=[NoaaGfsAwsOpenDataDatasetStorageConfig()],
    ),
    DwdIconEuForecastDataset(primary_storage_config=SourceCoopDatasetStorageConfig()),
    GefsAnalysisDataset(primary_storage_config=SourceCoopDatasetStorageConfig()),
    GefsForecast35DayDataset(primary_storage_config=SourceCoopDatasetStorageConfig()),
    NoaaHrrrForecast48HourDataset(
        primary_storage_config=NoaaHrrrAwsOpenDataDatasetStorageConfig(),
        replica_storage_configs=[
            NoaaHrrrAwsOpenDataDatasetStorageConfig(
                k8s_secret_name="aws-open-data-icechunk-storage-options-key",  # noqa: S106
                format=DatasetFormat.ICECHUNK,
            )
        ],
    ),
]

if Config.is_sentry_enabled:
    sentry_sdk.init(
        dsn=Config.sentry_dsn,
        environment=Config.env.value,
        project_root="src/",
        in_app_include=["reformatters"],
        default_integrations=True,
        integrations=[
            TyperIntegration(),
        ],
    )


app = typer.Typer(pretty_exceptions_show_locals=False)
app.command()(initialize_new_integration)


for dataset in DYNAMICAL_DATASETS:
    app.add_typer(dataset.get_cli(), name=dataset.dataset_id)


@app.command()
def deploy_operational_updates(
    docker_image: str | None = None,
) -> None:
    deploy.deploy_operational_updates(DYNAMICAL_DATASETS, docker_image)


if __name__ == "__main__":
    app()
