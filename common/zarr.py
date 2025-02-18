import logging
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import fsspec  # type: ignore
import s3fs  # type: ignore
import xarray as xr

from common.config import Config
from common.config_models import DataVar
from common.types import StoreLike

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_store() -> fsspec.FSMap:
    if Config.is_dev:
        local_store: StoreLike = fsspec.get_mapper(
            "data/output/noaa/gefs/forecast/dev.zarr"
        )
        return local_store

    s3 = s3fs.S3FileSystem(
        key=Config.source_coop.aws_access_key_id,
        secret=Config.source_coop.aws_secret_access_key,
    )
    store: StoreLike = s3.get_mapper(
        "s3://us-west-2.opendata.source.coop/dynamical/noaa-gefs-forecast/v0.1.0.zarr"
    )
    return store


@cache
def get_local_tmp_store() -> Path:
    return Path(f"data/tmp/{uuid4()}-tmp.zarr").absolute()


def get_mode(store: StoreLike) -> Literal["w-", "w"]:
    store_root = store.name if isinstance(store, Path) else getattr(store, "root", "")
    if store_root.endswith("dev.zarr") or store_root.endswith("-tmp.zarr"):
        return "w"  # Allow overwritting dev store

    return "w-"  # Safe default - don't overwrite


def copy_data_var(
    data_var: DataVar[Any],
    chunk_index: int,
    tmp_store: Path,
    final_store: fsspec.FSMap,
) -> Callable[[], None]:
    files_to_copy = list(
        tmp_store.glob(f"{data_var.name}/{chunk_index}.*.*.*")
    )  # matches any chunk with 4 or more dimensions

    def mv_files() -> None:
        logger.info(
            f"Copying data var chunks to final store ({final_store.root}) for {data_var.name}."
        )
        try:
            fs = final_store.fs
            fs.auto_mkdir = True
            fs.put(
                files_to_copy, final_store.root + f"/{data_var.name}/", auto_mkdir=True
            )
        except Exception:
            logger.exception("Failed to upload chunk")
        try:
            # Delete data to conserve space.
            for file in files_to_copy:
                file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete chunk after upload: {e}")

    return mv_files


def copy_zarr_metadata(
    template_ds: xr.Dataset, tmp_store: Path, final_store: fsspec.FSMap
) -> None:
    logger.info(
        f"Copying metadata to final store ({final_store.root}) from {tmp_store}"
    )
    metadata_files = []
    # Coordinates
    for coord in template_ds.coords:
        metadata_files.extend(list(tmp_store.glob(f"{coord}/*")))
    # zattrs, zarray, zgroup and zmetadata
    metadata_files.extend(list(tmp_store.glob("**/.z*")))
    # deduplicate
    metadata_files = list(dict.fromkeys(metadata_files))
    for file in metadata_files:
        relative = file.relative_to(tmp_store)
        final_store.fs.put_file(file, f"{final_store.root}/{relative}")
