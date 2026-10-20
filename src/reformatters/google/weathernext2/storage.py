from reformatters.common.storage import DatasetFormat, StorageConfig


class Weathernext2IcechunkDatasetStorageConfig(StorageConfig):
    """Icechunk storage in the private WeatherNext 2 bucket.

    This is an R2 bucket. Its endpoint URL and `force_path_style` come from the
    k8s secret, which holds `icechunk.s3_storage` kwargs.
    """

    base_path: str = "s3://dynamical-google-weathernext2"
    public_base_path: str | None = "https://google-weathernext2.r2.dynamical.org"
    k8s_secret_name: str = "weathernext2-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK
