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
from pydantic import Field, InstanceOf, computed_field
from zarr.abc.store import Store

from reformatters.common import kubernetes
from reformatters.common.config import Config, Env
from reformatters.common.logging import get_logger
from reformatters.common.pydantic import FrozenBaseModel
from reformatters.common.retry import retry
from reformatters.common.types import AppendDim

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
    icechunk_virtual_config: IcechunkVirtualConfig | None = (
        None  # None for materialized datasets
    )

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

    def primary_url(self) -> str:
        """Canonical production URL for the primary store, regardless of environment."""
        return _build_dataset_url(
            self.dataset_id, self.template_config_version, self.primary_storage_config
        )

    def replica_urls(self) -> list[str]:
        """Canonical production URLs for replica stores, regardless of environment."""
        return [
            _build_dataset_url(self.dataset_id, self.template_config_version, config)
            for config in self.replica_storage_configs
        ]

    def primary_store(self, writable: bool = False, branch: str = "main") -> Store:
        store_path = _get_store_path(
            self.dataset_id,
            self.version,
            self.primary_storage_config,
        )

        return _get_store(
            store_path,
            self.primary_storage_config,
            writable,
            branch,
            self.icechunk_virtual_config,
        )

    def replica_stores(
        self, writable: bool = False, branch: str = "main"
    ) -> list[Store]:
        # Disable replica stores in dev environment
        if Config.is_dev:
            return []

        stores = []
        for config in self.replica_storage_configs:
            store_path = _get_store_path(self.dataset_id, self.version, config)
            store = _get_store(
                store_path, config, writable, branch, self.icechunk_virtual_config
            )
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

    def all_stores_icechunk(self) -> bool:
        return all(
            config.format == DatasetFormat.ICECHUNK
            for config in (self.primary_storage_config, *self.replica_storage_configs)
        )

    def icechunk_primary_and_replica_repos(
        self,
    ) -> tuple[icechunk.Repository, tuple[icechunk.Repository, ...]]:
        """Returns (primary_repo, (replica_repos, ...)) for the common case where
        the caller just wants the primary and its replicas. Use icechunk_repos
        when you also need roles or a specific primary-first/last ordering."""
        repos = self.icechunk_repos(sort="primary-first")
        primaries = [repo for role, repo in repos if role == "primary"]
        # Virtual datasets require an icechunk primary (enforced by the
        # DynamicalDataset validator), so exactly one primary repo is expected.
        assert len(primaries) == 1, (
            f"expected exactly one icechunk primary, got {len(primaries)}"
        )
        replicas = tuple(repo for role, repo in repos if role != "primary")
        return primaries[0], replicas

    def _icechunk_storages(self) -> list[tuple[str, str, icechunk.Storage]]:
        """(role, store_path, storage) for each icechunk store, primary first then
        replicas (replicas skipped in dev, matching replica_stores)."""
        all_configs = [
            ("primary", self.primary_storage_config),
            *[
                (f"replica-{i}", config)
                for i, config in enumerate(self.replica_storage_configs)
            ],
        ]
        if Config.is_dev:
            all_configs = [all_configs[0]]

        storages: list[tuple[str, str, icechunk.Storage]] = []
        for role, config in all_configs:
            if config.format != DatasetFormat.ICECHUNK:
                continue
            store_path = _get_store_path(self.dataset_id, self.version, config)
            storages.append(
                (role, store_path, _get_icechunk_storage(store_path, config))
            )
        return storages

    def icechunk_repos(
        self, *, sort: Literal["primary-first", "primary-last"]
    ) -> list[tuple[str, icechunk.Repository]]:
        """Returns (role, Repository) for each icechunk store in the specified order.

        `role` uniquely identifies the repo within this StoreFactory:
        "primary" for the primary store, "replica-0", "replica-1", ... for replicas.
        """
        repo_config, credentials = _virtual_repository_config_and_credentials(
            self.icechunk_virtual_config
        )
        repos: list[tuple[str, icechunk.Repository]] = []
        for role, _store_path, ic_storage in self._icechunk_storages():
            # Retry to resolve multiple workers racing to create a new repo
            repo = retry(
                functools.partial(
                    icechunk.Repository.open_or_create,
                    ic_storage,
                    config=repo_config,
                    authorize_virtual_chunk_access=credentials,
                ),
                max_attempts=3,
            )
            repos.append((role, repo))

        match sort:
            case "primary-first":
                return sorted(repos, key=lambda r: r[0] != "primary")
            case "primary-last":
                return sorted(repos, key=lambda r: r[0] == "primary")
            case _ as unreachable:
                assert_never(unreachable)

    def persist_virtual_config(self) -> None:
        """Persist the in-code virtual chunk container set into each icechunk repo so
        external readers recover it via Repository.fetch_config and need only supply the
        anonymous authorize map at open (credentials are never persisted). No-op for
        materialized datasets. Single-writer: call once from parallel_setup. Repointing a
        container (same url_prefix, new backend) is a deliberate admin re-save; see "PR 5"
        in docs/plans/virtual_icechunk_datasets.md.
        """
        if self.icechunk_virtual_config is None:
            return
        in_code_prefixes = {
            container.url_prefix
            for container in self.icechunk_virtual_config.containers
        }
        repo_config, credentials = _virtual_repository_config_and_credentials(
            self.icechunk_virtual_config
        )
        for role, store_path, ic_storage in self._icechunk_storages():
            if icechunk.Repository.exists(ic_storage):
                persisted = icechunk.Repository.fetch_config(ic_storage)
                if (
                    persisted is not None
                    and set(persisted.virtual_chunk_containers or ())
                    == in_code_prefixes
                ):
                    continue
            repo = icechunk.Repository.open_or_create(
                ic_storage,
                config=repo_config,
                authorize_virtual_chunk_access=credentials,
            )
            repo.save_config()
            log.info(f"Persisted virtual chunk container config to {role} {store_path}")

    def _coordination_base_path(self) -> str:
        if Config.is_prod:
            base_path = self.primary_storage_config.base_path
        else:
            base_path = _LOCAL_ZARR_STORE_BASE_PATH
        return f"{base_path}/{self.dataset_id}/_internal"

    def _coordination_fs(self) -> fsspec.AbstractFileSystem:
        coordination_path = self._coordination_base_path()
        if not Config.is_prod:
            return fsspec.filesystem("file")

        storage_options = self.primary_storage_config.load_storage_options()
        # Icechunk secrets are keyed for `icechunk.s3_storage(**options)`
        # (e.g. access_key_id, secret_access_key, region). Coordination
        # files go through fsspec/s3fs, which uses different option names.
        if self.primary_storage_config.format == DatasetFormat.ICECHUNK:
            storage_options = _icechunk_to_s3fs_storage_options(storage_options)
        protocol = urlparse(coordination_path).scheme or "file"
        return fsspec.filesystem(protocol, **storage_options)

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
        fs.invalidate_cache(base)
        try:
            files = fs.ls(base, detail=False)
        except FileNotFoundError:
            return []
        if not files:
            return []
        # fs.cat runs reads concurrently on async backends like s3fs.
        contents = fs.cat(files)
        return [contents[f] for f in sorted(files)]

    def count_coordination_files(self, job_name: str, prefix: str) -> int:
        base = f"{self._coordination_base_path()}/{job_name}/{prefix}"
        fs = self._coordination_fs()
        fs.invalidate_cache(base)
        try:
            return len(fs.ls(base, detail=False))
        except FileNotFoundError:
            return 0

    def clear_coordination_files(self, job_name: str) -> None:
        path = f"{self._coordination_base_path()}/{job_name}"
        fs = self._coordination_fs()
        with contextlib.suppress(FileNotFoundError):
            fs.rm(path, recursive=True)


_ICECHUNK_TO_S3FS_CREDENTIAL_KEYS = {
    "access_key_id": "key",
    "secret_access_key": "secret",
    "session_token": "token",
}


def _icechunk_to_s3fs_storage_options(options: dict[str, Any]) -> dict[str, Any]:
    """Translate `icechunk.s3_storage` option names to s3fs/fsspec ones."""
    translated: dict[str, Any] = {}
    client_kwargs: dict[str, Any] = dict(options.get("client_kwargs") or {})
    for k, v in options.items():
        if k == "client_kwargs":
            continue
        if k == "region":
            client_kwargs["region_name"] = v
        elif k in _ICECHUNK_TO_S3FS_CREDENTIAL_KEYS:
            translated[_ICECHUNK_TO_S3FS_CREDENTIAL_KEYS[k]] = v
        else:
            translated[k] = v
    if client_kwargs:
        translated["client_kwargs"] = client_kwargs
    return translated


@cache
def get_local_tmp_store() -> Path:
    return Path(f"data/tmp/{uuid4()}-tmp.zarr").absolute()


def _format_dataset_path(
    base_path: str, dataset_id: str, version: str, dataset_format: DatasetFormat
) -> str:
    match dataset_format:
        case DatasetFormat.ZARR3:
            extension = "zarr"
        case DatasetFormat.ICECHUNK:
            extension = "icechunk"
        case _ as unreachable:
            assert_never(unreachable)
    return f"{base_path}/{dataset_id}/v{version}.{extension}"


def _get_store_path(
    dataset_id: str, version: str, storage_config: StorageConfig
) -> str:
    base_path = (
        storage_config.base_path if Config.is_prod else _LOCAL_ZARR_STORE_BASE_PATH
    )
    return _format_dataset_path(base_path, dataset_id, version, storage_config.format)


def _build_dataset_url(
    dataset_id: str, version: str, storage_config: StorageConfig
) -> str:
    """Canonical production URL for a dataset, regardless of `Config.env`.

    Unlike `_get_store_path`, this always uses `storage_config.base_path` and
    is intended for human-readable output (e.g. the `dataset-urls` CLI),
    not for opening a store.
    """
    return _format_dataset_path(
        storage_config.base_path, dataset_id, version, storage_config.format
    )


def _get_store(
    store_path: str,
    storage_config: StorageConfig,
    writable: bool,
    branch: str = "main",
    virtual_config: IcechunkVirtualConfig | None = None,
) -> Store:
    match storage_config.format:
        case DatasetFormat.ICECHUNK:
            assert store_path.endswith(".icechunk")
            return _get_icechunk_store(
                store_path, storage_config, writable, branch, virtual_config
            )
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
                #
                # If you add a new storage backend, also check the _coordination_fs
                # method to see if you need to add support for mapping its storage options
                # keys / kwargs from icechunk to fsspec names.
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
    virtual_config: IcechunkVirtualConfig | None = None,
) -> IcechunkStore:
    storage = _get_icechunk_storage(store_path, storage_config)
    repo_config, credentials = _virtual_repository_config_and_credentials(
        virtual_config
    )

    if writable:
        log.info(
            f"Opening icechunk store {store_path} on branch {branch} in writable mode"
        )
        repo = icechunk.Repository.open_or_create(
            storage, config=repo_config, authorize_virtual_chunk_access=credentials
        )
        session = repo.writable_session(branch)
        return session.store
    else:
        log.info(
            f"Opening icechunk store {store_path} on branch {branch} in readonly mode"
        )
        repo = icechunk.Repository.open(
            storage, config=repo_config, authorize_virtual_chunk_access=credentials
        )
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


def manifest_append_dim_split(
    *, split_size: int, dim: AppendDim
) -> icechunk.ManifestSplittingConfig:
    """Split every array's manifest along `dim` every `split_size` indices.

    The common-case terse constructor for `IcechunkVirtualConfig.manifest_split`;
    drop to `icechunk.ManifestSplittingConfig.from_dict` for per-array or
    multi-dimensional split policies.
    """
    return icechunk.ManifestSplittingConfig.from_dict(
        {
            icechunk.ManifestSplitCondition.AnyArray(): {
                icechunk.ManifestSplitDimCondition.DimensionName(dim): split_size
            }
        }
    )


class IcechunkVirtualConfig(FrozenBaseModel):
    """Per-dataset configuration for an icechunk virtual-chunk store."""

    # Source buckets the refs point into, registered as virtual chunk containers.
    containers: tuple[InstanceOf[icechunk.VirtualChunkContainer], ...] = Field(
        min_length=1
    )
    # Per-array manifest splitting policy (see manifest_append_dim_split).
    manifest_split: InstanceOf[icechunk.ManifestSplittingConfig]


def _virtual_repository_config_and_credentials(
    virtual_config: IcechunkVirtualConfig | None,
) -> tuple[icechunk.RepositoryConfig | None, dict[str, Any] | None]:
    """Build the icechunk `RepositoryConfig` override and anonymous authorize map for a
    virtual dataset, or `(None, None)` for a materialized one (icechunk uses defaults)."""
    if virtual_config is None:
        return None, None

    config = icechunk.RepositoryConfig.default()
    for container in virtual_config.containers:
        config.set_virtual_chunk_container(container)
    config.manifest = icechunk.ManifestConfig(splitting=virtual_config.manifest_split)
    # Cap the chunk-ref cache; the default OOMs streaming-commit writers, see "Chunk-ref cache OOM" in docs/plans/virtual_icechunk_datasets.md.
    config.caching = icechunk.CachingConfig(num_chunk_refs=1_000_000)

    # Every production source is S3 or S3-compatible (NOAA NODD, ECMWF, Source
    # Coop) and anonymous-read, and icechunk only ships an S3 anonymous credential
    # constructor; local-filesystem containers (dev/test) need no credentials. Map
    # each container to the right credential explicitly rather than silently handing
    # an S3 credential to a GCS/Azure container. To support a non-S3 or private /
    # requester-pays source, add an optional per-container credentials field to
    # IcechunkVirtualConfig and prefer it over these defaults.
    s3_compatible_stores = (
        icechunk.ObjectStoreConfig.S3,
        icechunk.ObjectStoreConfig.S3Compatible,
        icechunk.ObjectStoreConfig.Tigris,
    )
    credentials_by_prefix: dict[str, Any] = {}
    for container in virtual_config.containers:
        if isinstance(container.store, s3_compatible_stores):
            credentials_by_prefix[container.url_prefix] = (
                icechunk.s3_anonymous_credentials()
            )
        elif isinstance(container.store, icechunk.ObjectStoreConfig.LocalFileSystem):
            credentials_by_prefix[container.url_prefix] = None  # local files: no creds
        else:
            raise AssertionError(
                f"Virtual chunk container {container.url_prefix} uses an unsupported "
                f"store ({type(container.store).__name__}); only S3-compatible "
                "(anonymous) and local-filesystem sources are supported. Add explicit "
                "credentials to IcechunkVirtualConfig."
            )

    credentials = icechunk.containers_credentials(credentials_by_prefix)
    return config, credentials


# IcechunkVirtualConfig is defined below StoreFactory (which references it), so
# resolve the forward reference now that the name exists.
StoreFactory.model_rebuild()
