from typing import Any

from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.operational import OperationalResources
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.ecmwf.aifs_ens.forecast_fast import EcmwfAifsEnsForecastFastDataset
from reformatters.ecmwf.ifs_ens.forecast_0_25_degree_fast import (
    EcmwfIfsEnsForecast025DegreeFastDataset,
)

CLUSTER_DYNAMICAL_DATASETS: list[DynamicalDataset[Any, Any]] = [
    EcmwfIfsEnsForecast025DegreeFastDataset(
        primary_storage_config=StorageConfig(
            base_path="s3://dynamical-ecmwf-ifs-ens",
            k8s_secret_name="dynamical-ecmwf-output-r2-storage-options-key",  # noqa: S106
            format=DatasetFormat.ICECHUNK,
        ),
    ),
    EcmwfAifsEnsForecastFastDataset(
        primary_storage_config=StorageConfig(
            base_path="s3://dynamical-ecmwf-aifs-ens",
            k8s_secret_name="dynamical-ecmwf-output-r2-storage-options-key",  # noqa: S106
            format=DatasetFormat.ICECHUNK,
        ),
    ),
]
CLUSTER_OPERATIONAL_ARCHIVERS: list[OperationalResources] = []
