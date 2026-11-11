from reformatters.common.storage import DatasetFormat, StorageConfig


class NoaaHrrrIcechunkAwsOpenDataDatasetStorageConfig(StorageConfig):
    """Configuration for the storage of a AWS Open Data dataset."""

    base_path: str = "s3://dynamical-noaa-hrrr"
    k8s_secret_name: str = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK


class NoaaRtmaIcechunkAwsOpenDataDatasetStorageConfig(StorageConfig):
    base_path: str = "s3://dynamical-noaa-rtma"
    k8s_secret_name: str = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK


class NoaaUrmaIcechunkAwsOpenDataDatasetStorageConfig(StorageConfig):
    base_path: str = "s3://dynamical-noaa-urma"
    k8s_secret_name: str = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK


class NoaaRrfsIcechunkAwsOpenDataDatasetStorageConfig(StorageConfig):
    base_path: str = "s3://dynamical-noaa-rrfs"
    k8s_secret_name: str = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK


class NoaaRrfsEnsIcechunkAwsOpenDataDatasetStorageConfig(StorageConfig):
    base_path: str = "s3://dynamical-noaa-rrfs-ens"
    k8s_secret_name: str = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK


class NoaaRefsIcechunkAwsOpenDataDatasetStorageConfig(StorageConfig):
    base_path: str = "s3://dynamical-noaa-refs"
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


class UcsbChcChirpsIcechunkAwsOpenDataDatasetStorageConfig(StorageConfig):
    """UCSB CHC CHIRPS in Icechunk on AWS Open Data."""

    base_path: str = "s3://dynamical-ucsb-chc-chirps"
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


class Weathernext3IcechunkDatasetStorageConfig(StorageConfig):
    base_path: str = "s3://dynamical-google-weathernext3"
    k8s_secret_name: str = "weathernext3-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK
