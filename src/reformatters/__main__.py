from collections.abc import Sequence
from typing import Any

import sentry_sdk
import typer
from sentry_sdk.integrations.typer import TyperIntegration

import reformatters.noaa.gefs.analysis.cli as noaa_gefs_analysis
import reformatters.noaa.gefs.forecast_35_day.cli as noaa_gefs_forecast_35_day
import reformatters.noaa.hrrr.forecast_48_hour.cli as noaa_hrrr_forecast_48_hour
from reformatters.common import deploy
from reformatters.common.config import Config
from reformatters.common.dynamical_dataset import (
    DynamicalDataset,
    DynamicalDatasetStorageConfig,
)
from reformatters.contrib.uarizona.swann.analysis import UarizonaSwannAnalysisDataset
from reformatters.example.new_dataset import initialize_new_integration
from reformatters.noaa.gfs.forecast import NoaaGfsForecastDataset


class SourceCoopDatasetStorageConfig(DynamicalDatasetStorageConfig):
    """Configuration for the storage of a SourceCoop dataset."""

    base_path: str = "s3://us-west-2.opendata.source.coop/dynamical"
    k8s_secret_name: str = "source-coop-key"  # noqa: S105


# Registry of all DynamicalDatasets.
# Datasets that have not yet been ported over to the new DynamicalDataset pattern
# are excluded here until they are refactored.
DYNAMICAL_DATASETS: Sequence[DynamicalDataset[Any, Any]] = [
    UarizonaSwannAnalysisDataset(storage_config=SourceCoopDatasetStorageConfig()),
    NoaaGfsForecastDataset(storage_config=SourceCoopDatasetStorageConfig()),
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
app.add_typer(noaa_gefs_forecast_35_day.app, name=noaa_gefs_forecast_35_day.DATASET_ID)
app.add_typer(noaa_gefs_analysis.app, name=noaa_gefs_analysis.DATASET_ID)
app.add_typer(
    noaa_hrrr_forecast_48_hour.app, name=noaa_hrrr_forecast_48_hour.DATASET_ID
)

for dataset in DYNAMICAL_DATASETS:
    app.add_typer(dataset.get_cli(), name=dataset.dataset_id)


@app.command()
def deploy_operational_updates(
    docker_image: str | None = None,
) -> None:
    deploy.deploy_operational_updates(DYNAMICAL_DATASETS, docker_image)


if __name__ == "__main__":
    app()
