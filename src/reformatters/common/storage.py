import json
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any, Literal, assert_never
from uuid import uuid4

import fsspec  # type: ignore
import zarr
from pydantic import computed_field

from reformatters.common.config import Config, Env
from reformatters.common.pydantic import FrozenBaseModel

_LOCAL_ZARR_STORE_BASE_PATH = "data/output"
_SECRET_MOUNT_PATH = "/secrets"  # noqa: S105 this not a real secret


class DatasetFormat(StrEnum):
    ZARR3 = "zarr3"


class StorageConfig(FrozenBaseModel):
    """Configuration for the storage of a dataset in production."""

    base_path: str
    k8s_secret_name: str = ""
    format: DatasetFormat

    def load_storage_options(self) -> dict[str, Any]:
        """Load the storage options from the Kubernetes secret."""
        if not Config.is_prod or self.k8s_secret_name == "":
            return {}

        secret_file = Path(_SECRET_MOUNT_PATH) / f"{self.k8s_secret_name}.json"
        with open(secret_file) as f:
            options = json.load(f)
            assert isinstance(options, dict)
            return options


class StoreFactory(FrozenBaseModel):
    storage_config: StorageConfig
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def store_path(self) -> str:
        if Config.is_prod:
            base_path = self.storage_config.base_path
        else:
            base_path = _LOCAL_ZARR_STORE_BASE_PATH

        return f"{base_path}/{self.dataset_id}/v{self.version}.zarr"

    def store(self) -> zarr.abc.store.Store:
        if not Config.is_prod:
            # This should work, but it gives FileNotFoundError when it should be creating a new zarr.
            # return zarr.storage.FsspecStore.from_url(
            #     "file://data/output/noaa/gefs/forecast/dev.zarr"
            # )
            local_path = Path(self.store_path).absolute()
            return zarr.storage.LocalStore(local_path)

        storage_options = self.storage_config.load_storage_options()
        return zarr.storage.FsspecStore.from_url(
            self.store_path, storage_options=storage_options
        )

    def mode(self) -> Literal["w", "w-"]:
        return "w" if self.version == "dev" else "w-"

    def fsspec_filesystem(self) -> tuple[fsspec.spec.AbstractFileSystem, str]:
        """Returns a concrete filesystem implementation and relative path.

        The filesystem type depends on the store_path (e.g., LocalFileSystem
        for file://, S3FileSystem for s3://, etc.).
        """
        storage_options = self.storage_config.load_storage_options()
        fs, relative_path = fsspec.core.url_to_fs(self.store_path, **storage_options)
        assert isinstance(fs, fsspec.spec.AbstractFileSystem)
        return fs, relative_path


@cache
def get_local_tmp_store() -> Path:
    return Path(f"data/tmp/{uuid4()}-tmp.zarr").absolute()
