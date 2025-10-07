import functools
from collections.abc import Sequence
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any, Literal, assert_never
from urllib.parse import urlparse
from uuid import uuid4

import fsspec  # type: ignore[import-untyped]
import icechunk
import xarray as xr
import zarr
from icechunk.store import IcechunkStore
from pydantic import Field, computed_field, field_validator

from reformatters.common import kubernetes
from reformatters.common.config import Config, Env
from reformatters.common.logging import get_logger
from reformatters.common.pydantic import FrozenBaseModel
from reformatters.common.retry import retry

log = get_logger(__name__)

_LOCAL_ZARR_STORE_BASE_PATH = "data/output"

# This is a sentinel value to indicate that we should not try to load the storage options from a Kubernetes secret.
# This is useful in the test and dev environments where we don't have a Kubernetes secret mounted.
_NO_SECRET_NAME = "no-secret"  # noqa: S105


class DatasetFormat(StrEnum):
    ZARR3 = "zarr3"
    ICECHUNK = "icechunk"


class StorageConfig(FrozenBaseModel):
    """Configuration for the storage of a dataset in production."""

    base_path: str
    k8s_secret_name: str = _NO_SECRET_NAME
    format: DatasetFormat

    def load_storage_options(self) -> dict[str, Any]:
        """Load the storage options from the Kubernetes secret."""
        if self.k8s_secret_name == _NO_SECRET_NAME:
            return {}

        return kubernetes.load_secret(self.k8s_secret_name)


class StoreFactory(FrozenBaseModel):
    primary_storage_config: StorageConfig
    replica_storage_configs: Sequence[StorageConfig] = Field(default_factory=tuple)
    dataset_id: str
    template_config_version: str

    @field_validator("primary_storage_config")
    @classmethod
    def validate_primary_storage_not_icechunk(cls, v: StorageConfig) -> StorageConfig:
        # Currently, we do not support icechunk stores for the primary store.
        # This is because the format for the cloud storage credentials does not match
        # the storage_options format that we pass to fsspec in primary_store_fsspec_filesystem.
        #
        # To support this, we will need to add a translation helper to be able to initialize
        # the icechunk storage with the stored secrets and also to initialize the fsspec filesystem
        # with those same values.
        if v.format == DatasetFormat.ICECHUNK:
            raise ValueError("Primary storage config cannot be set to Icechunk format.")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def version(self) -> str:
        match Config.env:
            case Env.dev:
                return "dev"
            case Env.prod | Env.test:
                return self.template_config_version
            case _ as unreachable:
                assert_never(unreachable)

    def k8s_secret_names(self) -> list[str]:
        return [
            self.primary_storage_config.k8s_secret_name,
            *[config.k8s_secret_name for config in self.replica_storage_configs],
        ]

    def primary_store(self) -> zarr.abc.store.Store:
        store_path = _get_store_path(
            self.dataset_id,
            self.version,
            self.primary_storage_config,
        )

        return _get_store(store_path, self.primary_storage_config)

    def replica_stores(self) -> list[zarr.abc.store.Store]:
        # Disable replica stores in dev environment
        if Config.is_dev:
            return []

        stores = []
        for config in self.replica_storage_configs:
            store_path = _get_store_path(self.dataset_id, self.version, config)
            store = _get_store(store_path, config)
            stores.append(store)

        return stores

    def mode(self) -> Literal["w", "w-"]:
        return "w" if self.version == "dev" else "w-"

    def primary_store_fsspec_filesystem(
        self,
    ) -> tuple[fsspec.spec.AbstractFileSystem, str]:
        """Returns a concrete filesystem implementation and relative path.

        The filesystem type depends on the store_path (e.g., LocalFileSystem
        for file://, S3FileSystem for s3://, etc.).
        """
        store_path = _get_store_path(
            self.dataset_id,
            self.version,
            self.primary_storage_config,
        )
        storage_options = self.primary_storage_config.load_storage_options()

        fs, relative_path = fsspec.core.url_to_fs(store_path, **storage_options)
        assert isinstance(fs, fsspec.spec.AbstractFileSystem)

        return fs, relative_path

    def all_stores_exist(self) -> bool:
        """Check if all stores exist."""
        for store in [self.primary_store(), *self.replica_stores()]:
            try:
                xr.open_zarr(store, decode_timedelta=True)
            except Exception:
                log.error(f"Store {store} does not exist")
                return False
        return True


@cache
def get_local_tmp_store() -> Path:
    return Path(f"data/tmp/{uuid4()}-tmp.zarr").absolute()


def _get_store_path(
    dataset_id: str, version: str, storage_config: StorageConfig
) -> str:
    if Config.is_prod:
        base_path = storage_config.base_path
    else:
        base_path = _LOCAL_ZARR_STORE_BASE_PATH

    match storage_config.format:
        case DatasetFormat.ZARR3:
            extension = "zarr"
        case DatasetFormat.ICECHUNK:
            extension = "icechunk"
        case _ as unreachable:
            assert_never(unreachable)

    return f"{base_path}/{dataset_id}/v{version}.{extension}"


def _get_store(store_path: str, storage_config: StorageConfig) -> zarr.abc.store.Store:
    match storage_config.format:
        case DatasetFormat.ICECHUNK:
            assert store_path.endswith(".icechunk")
            return _get_icechunk_store(store_path, storage_config)
        case DatasetFormat.ZARR3:
            assert store_path.endswith(".zarr")
            return _get_zarr3_store(store_path, storage_config)
        case _ as unreachable:
            assert_never(unreachable)


def _get_zarr3_store(
    store_path: str, storage_config: StorageConfig
) -> zarr.abc.store.Store:
    if Config.is_prod:
        return zarr.storage.FsspecStore.from_url(
            store_path, storage_options=storage_config.load_storage_options()
        )
    else:
        return zarr.storage.LocalStore(Path(store_path).absolute())


def _get_icechunk_store(
    store_path: str, storage_config: StorageConfig
) -> IcechunkStore:
    if Config.is_prod:
        parsed_path = urlparse(store_path)

        scheme = parsed_path.scheme
        bucket = parsed_path.netloc
        prefix = parsed_path.path.lstrip("/")

        match scheme:
            case "s3":
                storage = icechunk.s3_storage(
                    bucket=bucket,
                    prefix=prefix,
                    **storage_config.load_storage_options(),
                )
            case _:
                # We are currently only working with s3 stores (and s3 compatible stores like R2).
                # Icechunk supports additional storage backends, which we can add support for
                # as needed. See https://icechunk.io/en/latest/storage/
                raise ValueError(
                    f"{scheme} Icechunk stores are not currently supported."
                )
    else:
        storage = icechunk.local_filesystem_storage(store_path)

    repo = icechunk.Repository.open_or_create(storage)
    session = repo.writable_session("main")
    return session.store


def commit_if_icechunk(
    message: str,
    primary_store: zarr.storage.StoreLike,
    replica_stores: Sequence[zarr.abc.store.Store],
) -> None:
    """Conveience function to handle committing to icechunk stores.

    By separating out the primary store from the replica stores, we are able
    to ensure that the replicas are updated before the primary.

    Concurrency handling:

    Dynamical datasets may be written in parallel across multiple Kubernetes jobs.
    Because these jobs are each responsible for their own discrete region in the zarr,
    we do not need to coordinate these separate writes. We therefore follow icechunk's
    "Uncooperative distributed writes" strategy, see documentation:
    https://icechunk.io/en/latest/parallel/#cooperative-distributed-writes.


    Each job however may need to rebase before it is able to commit. We use the rebase_with
    argument which will handle automatic retries until the commit succeeds.
    """

    def _commit(icechunk_store: IcechunkStore) -> None:
        icechunk_store.session.commit(
            message=message, rebase_with=icechunk.ConflictDetector()
        )

    for store in replica_stores:
        if isinstance(store, IcechunkStore):
            retry(
                functools.partial(_commit, store),
                max_attempts=10,
            )

    if isinstance(primary_store, IcechunkStore):
        retry(
            functools.partial(_commit, primary_store),
            max_attempts=10,
        )
