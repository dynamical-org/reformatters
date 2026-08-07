from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import takewhile
from typing import ClassVar

import icechunk
import obstore
import pandas as pd
import xarray as xr

from reformatters.common.config_models import ROOT
from reformatters.common.download import gcs_store
from reformatters.common.logging import get_logger
from reformatters.common.region_job import CoordinateValue, SourceFileCoord
from reformatters.common.types import Dim, Timedelta, Timestamp
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob

from .template_config import (
    PER_INIT_STORE_DATE,
    PRESSURE_LEVELS,
    GoogleWeathernext2DataVar,
)

log = get_logger(__name__)

SOURCE_LOCATION_PREFIX = "gs://weathernext/"
_SOURCE_BUCKET_URL = SOURCE_LOCATION_PREFIX.removesuffix("/")
# The ensemble-mean product; the 64 member product lives under weathernext_2_0_0/.
_SOURCE_ZARR_PREFIX = f"{SOURCE_LOCATION_PREFIX}weathernext_2_0_0_mean/zarr/"

# The source stores levels ascending; our pressure_level dimension is descending, so
# each reference maps its level label to the level's index in the source chunk grid.
_SOURCE_LEVEL_INDEX = {
    level: index for index, level in enumerate(sorted(PRESSURE_LEVELS))
}

# A coord is a whole zarr store, which has no single data-file length. Each reference's
# length is the size of its own chunk object, read from the store listing in file_refs.
_NO_SINGLE_FILE_SIZE = 0

# A coord covers every lead time of one init, so out_loc names one lead for the per-file
# manifest probe; the whole store commits atomically, so that lead's presence implies
# the init's. See docs/virtual_datasets.md.
_PROBE_LEAD_TIME = pd.Timedelta("6h")


def weathernext2_virtual_chunk_containers() -> tuple[
    icechunk.VirtualChunkContainer, ...
]:
    """Fresh container objects per call; icechunk containers can't be shared
    pydantic defaults."""
    return (
        icechunk.VirtualChunkContainer(SOURCE_LOCATION_PREFIX, icechunk.gcs_store()),
    )


class GoogleWeathernext2ForecastVirtualSourceFileCoord(SourceFileCoord):
    """One init time's source zarr store and the vars it contributes.

    Before PER_INIT_STORE_DATE the store spans a whole calendar year and the init is a
    position along its leading dimension; from that date each init has its own store.
    """

    init_time: Timestamp
    data_vars: Sequence[GoogleWeathernext2DataVar]

    @property
    def is_per_init_store(self) -> bool:
        return self.init_time >= PER_INIT_STORE_DATE

    def get_url(self) -> str:
        if self.is_per_init_store:
            return (
                f"{_SOURCE_ZARR_PREFIX}2025_to_present/"
                f"{self.init_time:%Y%m%d}_{self.init_time:%H}hr_01_preds/predictions.zarr"
            )
        year = self.init_time.year
        return f"{_SOURCE_ZARR_PREFIX}{year}_to_{year + 1}/predictions.zarr"

    def get_success_marker_url(self) -> str:
        """The zero-byte marker the source writes beside a store once it is complete."""
        return self.get_url().removesuffix("predictions.zarr") + "success"

    def chunk_key_prefix(self, var: GoogleWeathernext2DataVar) -> str:
        """The store-relative key prefix shared by every chunk object this coord needs
        of `var` — the whole variable in a per-init store, one init's slice of it in a
        yearly store."""
        source_name = var.internal_attrs.source_name
        if self.is_per_init_store:
            return f"{source_name}/"
        year_start = pd.Timestamp(f"{self.init_time.year}-01-01")
        store_init_index = (self.init_time - year_start) // pd.Timedelta("6h")
        return f"{source_name}/{store_init_index}."

    def chunk_key(
        self, var: GoogleWeathernext2DataVar, lead_index: int, level_index: int | None
    ) -> str:
        """The store-relative key of one source chunk object."""
        indices = [lead_index] if level_index is None else [lead_index, level_index]
        # Latitude and longitude are single-chunk in every source array.
        return self.chunk_key_prefix(var) + ".".join(str(i) for i in [*indices, 0, 0])

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"init_time": self.init_time, "lead_time": _PROBE_LEAD_TIME}


class GoogleWeathernext2ForecastVirtualRegionJob(
    VirtualRegionJob[
        GoogleWeathernext2DataVar, GoogleWeathernext2ForecastVirtualSourceFileCoord
    ]
):
    # Fire time is init+6h55m, so at fire the newest published init plus the three prior
    # cycles sit 6h55m to 24h55m back; 30h covers all four, so a couple of missed runs
    # still self-heal. Publication lags the 6h cycle, so the window's newest position is
    # always a cycle the source has not published yet; the fire polls for it until its
    # deadline and then leaves it to the next fire.
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("30h")

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[GoogleWeathernext2DataVar],
    ) -> Sequence[GoogleWeathernext2ForecastVirtualSourceFileCoord]:
        coords = []
        for init_time in pd.to_datetime(processing_region_ds["init_time"].values):
            available_vars = [
                var
                for var in data_var_group
                if (date := var.internal_attrs.date_available) is None
                or date <= init_time
            ]
            if available_vars:
                coords.append(
                    GoogleWeathernext2ForecastVirtualSourceFileCoord(
                        init_time=init_time, data_vars=available_vars
                    )
                )
        return coords

    def discover_available(
        self, pending: list[GoogleWeathernext2ForecastVirtualSourceFileCoord]
    ) -> list[tuple[GoogleWeathernext2ForecastVirtualSourceFileCoord, int]]:
        """The pending coords whose store has its success marker.

        The marker is written last, so its presence means every chunk object of the
        store has landed. Yearly-store coords share one marker, so it is probed once.
        """
        store = gcs_store(_SOURCE_BUCKET_URL)
        marker_keys = sorted(
            {_store_key(coord.get_success_marker_url()) for coord in pending}
        )
        with ThreadPoolExecutor(self.download_concurrency) as pool:
            landed = {
                key
                for key, exists in zip(
                    marker_keys,
                    pool.map(partial(_object_exists, store), marker_keys),
                    strict=True,
                )
                if exists
            }
        return [
            (coord, _NO_SINGLE_FILE_SIZE)
            for coord in pending
            if _store_key(coord.get_success_marker_url()) in landed
        ]

    def file_refs(
        self,
        coord: GoogleWeathernext2ForecastVirtualSourceFileCoord,
        file_size: int,  # noqa: ARG002 - a store has no single length, see _NO_SINGLE_FILE_SIZE
    ) -> list[VirtualRef]:
        store = gcs_store(_SOURCE_BUCKET_URL)
        store_key_prefix = _store_key(coord.get_url()) + "/"
        refs = []
        for var in coord.data_vars:
            sizes = _list_chunk_sizes(
                store, store_key_prefix, coord.chunk_key_prefix(var)
            )
            template_var = self.template_ds[var.path]
            levels: Sequence[tuple[int | None, int | None]] = (
                [(None, None)]
                if var.group is ROOT
                else [
                    (int(level), _SOURCE_LEVEL_INDEX[int(level)])
                    for level in template_var.get_index("pressure_level")
                ]
            )
            # The template's lead_time axis is the source's lead axis in the same order,
            # so a lead's position is its index in the source chunk grid.
            for lead_index, lead_time in enumerate(template_var.get_index("lead_time")):
                for level, level_index in levels:
                    key = coord.chunk_key(var, lead_index, level_index)
                    # An absent chunk object gets no reference and reads as fill.
                    if (size := sizes.get(key)) is None:
                        continue
                    out_loc: dict[Dim, CoordinateValue] = {
                        "init_time": coord.init_time,
                        "lead_time": lead_time,
                    }
                    if level is not None:
                        out_loc["pressure_level"] = level
                    refs.append(
                        VirtualRef(
                            data_var=var,
                            out_loc=out_loc,
                            location=SOURCE_LOCATION_PREFIX + store_key_prefix + key,
                            offset=0,
                            length=size,
                        )
                    )
        return refs


def _store_key(url: str) -> str:
    return url.removeprefix(SOURCE_LOCATION_PREFIX)


def _object_exists(store: obstore.store.ObjectStore, key: str) -> bool:
    # obstore lists only *under* a prefix, so an exact object path is not listable and
    # existence has to be a head request.
    try:
        obstore.head(store, key)
    except FileNotFoundError:
        return False
    return True


def _list_chunk_sizes(
    store: obstore.store.ObjectStore, store_key_prefix: str, chunk_key_prefix: str
) -> dict[str, int]:
    """Every source chunk object under `chunk_key_prefix`, keyed store-relative, mapped
    to its size — a source chunk is a whole object, so its size is a reference length.

    obstore matches a listing prefix by whole path component, so a yearly store's
    `<var>/<init>.` prefix is not listable on its own. The variable's directory is
    listed from an offset just below the wanted keys, which are lexicographically
    contiguous, and the scan stops at the first non-match rather than walking the rest
    of the year.
    """
    full_prefix = store_key_prefix + chunk_key_prefix
    variable_dir = (
        store_key_prefix + chunk_key_prefix[: chunk_key_prefix.index("/") + 1]
    )
    offset = None if full_prefix == variable_dir else full_prefix.removesuffix(".")
    listed = takewhile(
        lambda item: item[0].startswith(full_prefix),
        _list_objects(store, variable_dir, offset),
    )
    return {key.removeprefix(store_key_prefix): size for key, size in listed}


def _list_objects(
    store: obstore.store.ObjectStore, prefix: str, offset: str | None
) -> Iterator[tuple[str, int]]:
    for batch in obstore.list(store, prefix=prefix, offset=offset, chunk_size=10_000):
        for meta in batch:
            yield meta["path"], meta["size"]
