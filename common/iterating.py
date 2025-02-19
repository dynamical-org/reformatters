from collections import deque
from collections.abc import Iterable
from itertools import islice, pairwise, starmap

import numpy as np
import xarray as xr


def chunk_i_slices(ds: xr.Dataset, dim: str) -> Iterable[slice]:
    """Returns the integer offset slices which correspond to each chunk along `dim` of `ds`."""
    vars_dim_chunk_sizes = {var.chunksizes[dim] for var in ds.data_vars.values()}
    assert (
        len(vars_dim_chunk_sizes) == 1
    ), f"Inconsistent chunk sizes among data variables along update dimension ({dim}): {vars_dim_chunk_sizes}"
    dim_chunk_sizes = next(iter(vars_dim_chunk_sizes))  # eg. 2, 2, 2
    stop_idx = np.cumsum(dim_chunk_sizes)  # eg.    2, 4, 6
    start_idx = np.insert(stop_idx, 0, 0)  # eg. 0, 2, 4, 6
    return starmap(slice, pairwise(start_idx))  # eg. slice(0,2), slice(2,4), slice(4,6)


def consume[T](iterator: Iterable[T], n: int | None = None) -> None:
    "Advance the iterator n-steps ahead. If n is None, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)
