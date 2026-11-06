from typing import Any

import icechunk
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from icechunk.store import IcechunkStore
from xarray.coding.times import decode_cf_datetime
from zarr.core.metadata import ArrayV3Metadata

from reformatters.common.iterating import node_path_prefix, walk_data_arrays
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import Timestamp


def _coordinate_index(array: zarr.Array) -> pd.DatetimeIndex:
    units = array.attrs["units"]
    calendar = array.attrs.get("calendar", "standard")
    assert isinstance(units, str)
    assert isinstance(calendar, str)
    return pd.DatetimeIndex(
        decode_cf_datetime(np.asarray(array[:]), units, calendar, use_cftime=False)
    )


def _append_index(
    store: IcechunkStore, config: TemplateConfig[Any]
) -> pd.DatetimeIndex:
    array = zarr.open_array(store, path=config.append_dim, mode="r")
    assert isinstance(array.metadata, ArrayV3Metadata)
    assert array.metadata.dimension_names == (config.append_dim,)
    if array.shape[0] == 0:
        return pd.DatetimeIndex([])
    labels = _coordinate_index(array)
    start = pd.Timestamp(config.append_dim_start)
    if start.tzinfo is not None:
        start = start.tz_convert(None)
    assert labels[0] >= start, "Stored origin is before the configured start"
    assert (labels[0] - start) % config.append_dim_frequency == pd.Timedelta(0), (
        "Stored origin is off the append grid"
    )
    assert labels.equals(
        pd.date_range(labels[0], periods=len(labels), freq=config.append_dim_frequency)
    ), "Stored append coordinate is not a regular ascending grid"
    return labels


def retention_start(store: IcechunkStore, config: TemplateConfig[Any]) -> Timestamp:
    labels = _append_index(store, config)
    start = labels[0] if len(labels) else pd.Timestamp(config.append_dim_start)
    return start.tz_convert(None) if start.tzinfo is not None else start


def store_template(
    store: IcechunkStore,
    config: TemplateConfig[Any],
    *,
    end: Timestamp | None = None,
) -> xr.DataTree:
    labels = _append_index(store, config)
    start = labels[0] if len(labels) else pd.Timestamp(config.append_dim_start)
    if start.tzinfo is not None:
        start = start.tz_convert(None)
    stored_end = start + len(labels) * config.append_dim_frequency
    if end is not None:
        end = end.tz_convert(None) if end.tzinfo is not None else end
        stored_end = max(stored_end, end)
    return config.get_template(stored_end, start_time=start)


def trim_retention(
    session: icechunk.Session, config: TemplateConfig[Any], positions: int
) -> bool:
    """Remove excess leading positions without reading data-variable payloads."""
    assert positions > 0, "Rolling positions must be positive"
    labels = _append_index(session.store, config)
    removed = len(labels) - positions
    if removed <= 0:
        return False
    template = config.get_template(
        labels[-1] + config.append_dim_frequency, start_time=labels[0]
    )
    expected = {
        f"{node_path_prefix(node)}{name}": variable
        for node in template.subtree
        for name, variable in node.to_dataset().variables.items()
    }
    data_paths = dict(walk_data_arrays(template))
    arrays = {
        path: array
        for path, array in zarr.open_group(session.store, mode="r+").members(
            max_depth=None
        )
        if isinstance(array, zarr.Array)
    }
    assert arrays.keys() <= expected.keys(), (
        f"Stored arrays omitted by config: {arrays.keys() - expected.keys()}"
    )
    for path, array in arrays.items():
        assert isinstance(array.metadata, ArrayV3Metadata)
        dims = array.metadata.dimension_names or ()
        assert tuple(expected[path].dims) == dims, f"{path}: dimensions differ"
        if config.append_dim not in dims:
            continue
        axis = dims.index(config.append_dim)
        assert array.shape[axis] == len(labels), f"{path}: append size differs"
        if path in data_paths:
            assert array.shards is None, f"{path}: retention requires unsharded data"
            assert array.chunks[axis] == 1, (
                f"{path}: retention requires unit append chunks"
            )
        if (
            dims == (config.append_dim,)
            and path.rsplit("/", 1)[-1] == config.append_dim
        ):
            assert _coordinate_index(array).equals(labels), (
                f"{path}: duplicate append coordinate differs"
            )

    for path, array in arrays.items():
        assert isinstance(array.metadata, ArrayV3Metadata)
        dims = array.metadata.dimension_names or ()
        if config.append_dim not in dims:
            continue
        axis = dims.index(config.append_dim)
        selection = [slice(None)] * array.ndim
        selection[axis] = slice(removed, None)
        values = None if path in data_paths else array[tuple(selection)]
        if path in data_paths:
            offset = [0] * array.ndim
            offset[axis] = -removed
            session.shift_array(f"/{path}", offset)
        shape = list(array.shape)
        shape[axis] = positions
        array.resize(tuple(shape))
        if values is not None:
            array[:] = values
    return True
