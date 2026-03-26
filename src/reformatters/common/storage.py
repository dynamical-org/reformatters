import contextlib
import functools
from collections.abc import Sequence
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any, Literal, assert_never
from urllib.parse import urlparse
from uuid import uuid4

import fsspec
import icechunk
import xarray as xr
import zarr
import zarr.abc.store
import zarr.storage
from icechunk.store import IcechunkStore
from pydantic import Field, computed_field
from zarr.abc.store import Store

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

    @computed_field
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

    def primary_store(self, writable: bool = False, branch: str = "main") -> Store:
        store_path = _get_store_path(
            self.dataset_id,
            self.version,
            self.primary_storage_config,
        )

        return _get_store(store_path, self.primary_storage_config, writable, branch)

    def replica_stores(
        self, writable: bool = False, branch: str = "main"
    ) -> list[Store]:
        # Disable replica stores in dev environment
        if Config.is_dev:
            return []

        stores = []
        for config in self.replica_storage_configs:
            store_path = _get_store_path(self.dataset_id, self.version, config)
            store = _get_store(store_path, config, writable, branch)
            stores.append(store)

        return stores

    def mode(self) -> Literal["w", "w-"]:
        return "w" if self.version == "dev" else "w-"

    def all_stores_exist(self) -> bool:
        """Check if all stores exist."""
        for store in [self.primary_store(), *self.replica_stores()]:
            try:
                xr.open_zarr(store, decode_timedelta=True)
            except Exception:
                log.exception(f"Store {store} does not exist")
                return False
        return True

    def icechunk_repos(self) -> list[tuple[str, icechunk.Repository]]:
        """Returns (role, Repository) for each icechunk store. Primary first, then replicas."""
        repos: list[tuple[str, icechunk.Repository]] = []
        all_configs = [
            ("primary", self.primary_storage_config),
            *[
                (f"replica-{i}", config)
                for i, config in enumerate(self.replica_storage_configs)
            ],
        ]
        # In dev, skip replicas (same as replica_stores behavior)
        if Config.is_dev:
            all_configs = [all_configs[0]]

        for role, config in all_configs:
            if config.format != DatasetFormat.ICECHUNK:
                continue
            store_path = _get_store_path(self.dataset_id, self.version, config)
            storage = _get_icechunk_storage(store_path, config)
            repo = icechunk.Repository.open_or_create(storage)
            repos.append((role, repo))
        return repos

    def _coordination_base_path(self) -> str:
        if Config.is_prod:
            base_path = self.primary_storage_config.base_path
        else:
            base_path = _LOCAL_ZARR_STORE_BASE_PATH
        return f"{base_path}/{self.dataset_id}/_internal"

    def _coordination_fs(self) -> fsspec.AbstractFileSystem:
        coordination_path = self._coordination_base_path()
        if Config.is_prod:
            storage_options = self.primary_storage_config.load_storage_options()
            protocol = (
                coordination_path.split("://")[0]
                if "://" in coordination_path
                else "file"
            )
            return fsspec.filesystem(protocol, **storage_options)
        else:
            return fsspec.filesystem("file")

    def write_coordination_file(self, job_name: str, key: str, data: bytes) -> None:
        base = self._coordination_base_path()
        path = f"{base}/{job_name}/{key}"
        parent = path.rsplit("/", 1)[0]
        fs = self._coordination_fs()
        fs.mkdirs(parent, exist_ok=True)
        fs.pipe_file(path, data)

    def read_all_coordination_files(self, job_name: str, prefix: str) -> list[bytes]:
        base = f"{self._coordination_base_path()}/{job_name}/{prefix}"
        fs = self._coordination_fs()
        try:
            files = fs.ls(base, detail=False)
        except FileNotFoundError:
            return []
        return [fs.cat_file(f) for f in sorted(files)]

    def clear_coordination_files(self, job_name: str) -> None:
        path = f"{self._coordination_base_path()}/{job_name}"
        fs = self._coordination_fs()
        with contextlib.suppress(FileNotFoundError):
            fs.rm(path, recursive=True)


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


def _get_store(
    store_path: str, storage_config: StorageConfig, writable: bool, branch: str = "main"
) -> Store:
    match storage_config.format:
        case DatasetFormat.ICECHUNK:
            assert store_path.endswith(".icechunk")
            return _get_icechunk_store(store_path, storage_config, writable, branch)
        case DatasetFormat.ZARR3:
            assert store_path.endswith(".zarr")
            return _get_zarr3_store(store_path, storage_config, writable)
        case _ as unreachable:
            assert_never(unreachable)


def _get_zarr3_store(
    store_path: str, storage_config: StorageConfig, writable: bool
) -> Store:
    if Config.is_prod:
        return zarr.storage.FsspecStore.from_url(
            store_path, storage_options=storage_config.load_storage_options()
        ).with_read_only(not writable)
    else:
        return zarr.storage.LocalStore(Path(store_path).absolute()).with_read_only(
            not writable
        )


def _get_icechunk_storage(
    store_path: str, storage_config: StorageConfig
) -> icechunk.Storage:
    if Config.is_prod:
        parsed_path = urlparse(store_path)

        scheme = parsed_path.scheme
        bucket = parsed_path.netloc
        prefix = parsed_path.path.lstrip("/")

        match scheme:
            case "s3":
                return icechunk.s3_storage(
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
        return icechunk.local_filesystem_storage(store_path)


def _get_icechunk_store(
    store_path: str,
    storage_config: StorageConfig,
    writable: bool,
    branch: str = "main",
) -> IcechunkStore:
    storage = _get_icechunk_storage(store_path, storage_config)

    if writable:
        log.info(
            f"Opening icechunk store {store_path} on branch {branch} in writable mode"
        )
        repo = icechunk.Repository.open_or_create(storage)
        session = repo.writable_session(branch)
        return session.store
    else:
        log.info(
            f"Opening icechunk store {store_path} on branch {branch} in readonly mode"
        )
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session(branch)
        return session.store


def commit_if_icechunk(
    message: str,
    primary_store: zarr.storage.StoreLike,
    replica_stores: Sequence[Store],
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
