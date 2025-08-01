from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal, assert_never

import fsspec  # type: ignore[import-untyped]
import zarr
from pydantic import Field, computed_field

from reformatters.common.config import Config, Env
from reformatters.common.pydantic import FrozenBaseModel
from reformatters.common.zarr import _LOCAL_ZARR_STORE_BASE_PATH


class DatasetFormat(StrEnum):
    ZARR3 = "zarr3"


class StorageConfig(FrozenBaseModel):
    """Configuration for the storage of a dataset in production."""

    base_path: str
    k8s_secret_names: Sequence[str] = Field(default_factory=tuple)
    format: DatasetFormat


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

    def store(self) -> zarr.abc.store.Store:
        if not Config.is_prod:
            # This should work, but it gives FileNotFoundError when it should be creating a new zarr.
            # return zarr.storage.FsspecStore.from_url(
            #     "file://data/output/noaa/gefs/forecast/dev.zarr"
            # )
            # Instead make a zarr LocalStore and attach an fsspec filesystem to it.
            # Technically that filesystem should be an AsyncFileSystem to match
            # zarr.storage.FsspecStore but AsyncFileSystem does not support _put.
            local_path = Path(
                f"{_LOCAL_ZARR_STORE_BASE_PATH}/{self.dataset_id}/v{self.version}.zarr"
            ).absolute()

            store = zarr.storage.LocalStore(local_path)

            fs = fsspec.implementations.local.LocalFileSystem(auto_mkdir=True)
            store.fs = fs  # type: ignore[attr-defined]
            store.path = str(local_path)  # type: ignore[attr-defined]
            # We are duck typing this LocalStore as a FsspecStore
            return store

        return zarr.storage.FsspecStore.from_url(
            f"{self.storage_config.base_path}/{self.dataset_id}/v{self.version}.zarr"
        )

    def mode(self) -> Literal["w", "w-"]:
        return "w" if self.version == "dev" else "w-"
