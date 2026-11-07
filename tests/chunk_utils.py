import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pytest
import xarray as xr
from zarr.abc.store import Store

from reformatters.common import (
    materialized_region_job as materialized_region_job_module,
)
from reformatters.common import shared_memory_utils
from reformatters.common.iterating import shard_slice_indexers
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import AppendDim


def _shrink_dim(size: int, chunk: int, shard: int) -> tuple[int, int]:
    """
    Smallest (chunk, shard) that tile `size` with the same chunk and shard
    counts as production at this `size`, each capped at two. So a dim that spans
    multiple chunks or multiple shards stays that way, and a single one stays
    single, while the counts (and thus the write cost) stay small.
    """
    assert shard % chunk == 0, f"shard {shard} is not a multiple of chunk {chunk}"

    # n_chunks >= n_shards always, so capping both at 2 keeps n_chunks >= n_shards.
    chunks_target = min(math.ceil(size / chunk), 2)
    shards_target = min(math.ceil(size / shard), 2)

    new_chunk = math.ceil(size / chunks_target)
    new_shard = new_chunk * (chunks_target // shards_target)
    return new_chunk, new_shard


def shrink_chunks_and_shards[T: (xr.Dataset, xr.DataTree)](
    ds: T, dims: Iterable[str] | None = None
) -> T:
    """
    Shrink every data var's chunk and shard encoding to the smallest layout that
    still exercises the same structure as production at this dataset's sizes.

    Accepts a template as either a Dataset or a DataTree (per-node shrink).

    Integration tests trim the append/lead/ensemble dims then write a small
    region. With production chunk geometry that means allocating, filling, and
    compressing a full production-sized chunk buffer for every chunk touched
    (often thousands of near-empty chunks), which dominates test runtime without
    adding coverage.

    For each dim, this keeps the production chunk and shard counts at the current
    size, capped at two: a dim that spans multiple chunks or multiple shards in
    production stays multi, and a single one stays single, so the shrink cannot
    accidentally collapse a chunk or shard boundary a test depends on. (It does
    not preserve how many chunks pack into a shard, which is zarr-internal.)

    Call this *after* trimming the template (e.g. after `.sel(lead_time=...)`) so
    the sizes reflect what the test actually writes. Pass `dims` to shrink only
    those dimensions, e.g. shrink the spatial dims while leaving the append dim
    at production size for a shard-boundary test.
    """
    if isinstance(ds, xr.DataTree):
        return cast(
            "T",
            xr.DataTree.from_dict(
                {
                    node.path: shrink_chunks_and_shards(node.to_dataset(), dims)
                    for node in ds.subtree
                }
            ),
        )
    return cast("T", _shrink_dataset(ds, dims))


def _shrink_dataset(ds: xr.Dataset, dims: Iterable[str] | None) -> xr.Dataset:
    selected = set(dims) if dims is not None else None
    for var in ds.data_vars.values():
        chunks = list(var.encoding["chunks"])
        has_shards = var.encoding.get("shards") is not None
        shards = list(var.encoding["shards"]) if has_shards else list(chunks)
        for axis, dim in enumerate(var.dims):
            if selected is not None and dim not in selected:
                continue
            chunks[axis], shards[axis] = _shrink_dim(
                ds.sizes[dim], chunks[axis], shards[axis]
            )
        var.encoding["chunks"] = tuple(chunks)
        if has_shards:
            var.encoding["shards"] = tuple(shards)
    return ds


def patch_write_first_shard_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only write the first spatial shard (all indexer slices start at 0).

    Replaces write_shards rather than write_shard_to_zarr because the latter
    is pickled for ProcessPoolExecutor and local functions can't be pickled.
    """
    _orig_write_shard_to_zarr = shared_memory_utils.write_shard_to_zarr

    def _first_shard_only_write_shards(
        processing_region_da_template: xr.DataArray,
        shared_buffer: object,
        append_dim: AppendDim,
        output_region_ds: xr.Dataset,
        store: Store | Path,
        cpu_process_executor: object,
    ) -> None:
        all_indexers = shard_slice_indexers(
            output_region_ds[processing_region_da_template.name]
        )
        for si in all_indexers:
            if all(s.start == 0 for s in si):
                _orig_write_shard_to_zarr(
                    processing_region_da_template,
                    shared_buffer.name,  # ty: ignore[unresolved-attribute]
                    append_dim,
                    output_region_ds,
                    store,
                    si,
                )

    monkeypatch.setattr(
        materialized_region_job_module, "write_shards", _first_shard_only_write_shards
    )


def patch_template_to_variables(
    monkeypatch: pytest.MonkeyPatch,
    template_config_type: type[TemplateConfig[Any]],
    wanted: Iterable[str],
) -> None:
    """Shrink a template config's template and data_vars to the variables named by
    bare name or `group/name`, dropping groups left empty. Composes with an earlier
    get_template patch on the same type."""
    names = set(wanted)
    original_get_template = template_config_type.get_template
    original_data_vars = template_config_type.data_vars.fget
    assert original_data_vars is not None

    def selected(path: str, name: str) -> bool:
        return name in names or path in names

    def trimmed_get_template(self: TemplateConfig[Any], end_time: Any) -> xr.DataTree:  # noqa: ANN401
        nodes = {}
        for node in original_get_template(self, end_time).subtree:
            prefix = node.path.strip("/")
            keep = [
                name
                for name in map(str, node.data_vars)
                if selected(f"{prefix}/{name}" if prefix else name, name)
            ]
            if keep or node.path == "/":
                nodes[node.path] = node.to_dataset().drop_vars(
                    [name for name in node.data_vars if name not in keep]
                )
        return xr.DataTree.from_dict(nodes)

    monkeypatch.setattr(template_config_type, "get_template", trimmed_get_template)
    monkeypatch.setattr(
        template_config_type,
        "data_vars",
        property(
            lambda self: [
                v for v in original_data_vars(self) if selected(v.path, v.name)
            ]
        ),
    )
