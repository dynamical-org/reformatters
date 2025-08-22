import base64
import json
import os
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any, Literal, assert_never
from uuid import uuid4

import fsspec  # type: ignore
import zarr
from pydantic import Field, computed_field

from reformatters.common import kubernetes
from reformatters.common.config import Config, Env
from reformatters.common.pydantic import FrozenBaseModel

_LOCAL_ZARR_STORE_BASE_PATH = "data/output"
_SECRET_MOUNT_PATH = "/secrets"  # noqa: S105 this not a real secret
_STORAGE_OPTIONS_KEY = "storage_options.json"

# This is a sentinel value to indicate that we should not try to load the storage options from a Kubernetes secret.
# This is useful in the test and dev environments where we don't have a Kubernetes secret mounted.
_NO_SECRET_NAME = "no-secret"  # noqa: S105


class DatasetFormat(StrEnum):
    ZARR3 = "zarr3"


class StorageConfig(FrozenBaseModel):
    """Configuration for the storage of a dataset in production."""

    base_path: str
    k8s_secret_name: str = _NO_SECRET_NAME
    format: DatasetFormat

    def load_storage_options(self) -> dict[str, Any]:
        """Load the storage options from the Kubernetes secret."""
        if not Config.is_prod or self.k8s_secret_name == _NO_SECRET_NAME:
            return {}

        secret_file = Path(_SECRET_MOUNT_PATH) / f"{self.k8s_secret_name}.json"

        # When we backfill, we need to write the template metadata to our cloud stoage
        # location. To do this, we need the credentials to write to the base path, but
        # because this happens locally, we don't have the secret mounted. In this case
        # we will attempt to load the secrets from kubernetes locally. This assumes that
        # we are connected to the kubernetes cluster locally. We have a guard on JOB_NAME
        # to ensure that we don't try to do this when this is run in the cluster.
        if not secret_file.exists() and os.getenv("JOB_NAME") is None:
            return self._load_storage_options_locally()

        with open(secret_file) as f:
            options = json.load(f)
            assert isinstance(options, dict)
            return options

    def _load_storage_options_locally(self) -> dict[str, Any]:
        assert self.k8s_secret_name is not None
        secret_data = kubernetes.load_k8s_secrets_locally(self.k8s_secret_name)
        storage_options_json = base64.b64decode(
            secret_data[_STORAGE_OPTIONS_KEY]
        ).decode("utf-8")
        options = json.loads(storage_options_json)
        assert isinstance(options, dict)
        return options


class StoreFactory(FrozenBaseModel):
    primary_storage_config: StorageConfig
    replica_storage_configs: list[StorageConfig] = Field(default_factory=list)
    dataset_id: str
    template_config_version: str

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
            self.primary_storage_config.base_path,
        )

        return _get_store(store_path, self.primary_storage_config)

    def replica_stores(self) -> list[zarr.abc.store.Store]:
        stores = []
        for config in self.replica_storage_configs:
            store_path = _get_store_path(
                self.dataset_id,
                self.version,
                config.base_path,
            )
            store = _get_store(store_path, config)
            stores.append(store)

        return stores

    def mode(self) -> Literal["w", "w-"]:
        return "w" if self.version == "dev" else "w-"

    def fsspec_filesystem(self) -> tuple[fsspec.spec.AbstractFileSystem, str]:
        """Returns a concrete filesystem implementation and relative path.

        The filesystem type depends on the store_path (e.g., LocalFileSystem
        for file://, S3FileSystem for s3://, etc.).
        """
        store_path = _get_store_path(
            self.dataset_id,
            self.version,
            self.primary_storage_config.base_path,
        )
        storage_options = self.primary_storage_config.load_storage_options()

        fs, relative_path = fsspec.core.url_to_fs(store_path, **storage_options)
        assert isinstance(fs, fsspec.spec.AbstractFileSystem)

        return fs, relative_path


@cache
def get_local_tmp_store() -> Path:
    return Path(f"data/tmp/{uuid4()}-tmp.zarr").absolute()


def _get_store_path(dataset_id: str, version: str, base_path: str) -> str:
    if not Config.is_prod:
        base_path = _LOCAL_ZARR_STORE_BASE_PATH

    return f"{base_path}/{dataset_id}/v{version}.zarr"


def _get_store(store_path: str, storage_config: StorageConfig) -> zarr.abc.store.Store:
    if not Config.is_prod:
        return zarr.storage.LocalStore(Path(store_path).absolute())

    return zarr.storage.FsspecStore.from_url(
        store_path, storage_options=storage_config.load_storage_options()
    )
