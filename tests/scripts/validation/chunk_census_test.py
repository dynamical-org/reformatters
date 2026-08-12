import numpy as np
import xarray as xr
from zarr.storage import MemoryStore

from scripts.validation.chunk_census import census_array_chunks


def test_census_array_chunks_finds_absent_chunks_inside_present_shards() -> None:
    values = np.ones((8, 4), dtype=np.float32)
    values[:2, :2] = 0.0
    values[4:, :] = 0.0
    ds = xr.Dataset({"var": (("time", "x"), values)})
    ds["var"].encoding.update(
        {
            "chunks": (2, 2),
            "shards": (4, 4),
            "fill_value": 0.0,
        }
    )
    store = MemoryStore()
    ds.to_zarr(
        store,
        mode="w",
        consolidated=False,
        write_empty_chunks=False,
    )

    result = census_array_chunks(store, "var")

    assert result.expected_shards == 2
    assert result.present_shards == 1
    assert result.absent_shards == 1
    assert result.expected_chunks == 8
    assert result.present_chunks == 3
    assert result.absent_chunks == 5
