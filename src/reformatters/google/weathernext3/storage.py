from reformatters.common.storage import DatasetFormat, StorageConfig


class Weathernext3IcechunkDatasetStorageConfig(StorageConfig):
    base_path: str = "s3://dynamical-google-weathernext3"
    public_base_path: str | None = "https://google-weathernext3.r2.dynamical.org"
    k8s_secret_name: str = "weathernext3-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK
