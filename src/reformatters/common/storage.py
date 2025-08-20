from collections.abc import Sequence
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Literal, assert_never
from uuid import uuid4

import zarr
from pydantic import Field, computed_field

from reformatters.common.config import Config, Env
from reformatters.common.pydantic import FrozenBaseModel

_LOCAL_ZARR_STORE_BASE_PATH = "data/output"


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

        return zarr.storage.FsspecStore.from_url(self.store_path)

    def mode(self) -> Literal["w", "w-"]:
        return "w" if self.version == "dev" else "w-"


@cache
def get_local_tmp_store() -> Path:
    return Path(f"data/tmp/{uuid4()}-tmp.zarr").absolute()
