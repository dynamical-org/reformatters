from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from gribberish.zarr import GribberishCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import MemoryStore

from reformatters.common.download import s3_read_bytes
from reformatters.noaa.urma.analysis_virtual.region_job import (
    NoaaUrmaAnalysisVirtualRegionJob,
    NoaaUrmaAnalysisVirtualSourceFileCoord,
)
from tests.noaa.urma.analysis_virtual.region_job_test import WAVE


@pytest.mark.slow
def test_real_wave_through_zarr_matches_independent_eccodes_mask_and_values(
    tmp_path: Path, source_fixture: dict[str, Any]
) -> None:
    if not source_fixture["extra_ranges"]:
        pytest.skip("Source precedes approved bitmap onset")
    time = pd.Timestamp(source_fixture["time"])
    job = NoaaUrmaAnalysisVirtualRegionJob(
        tmp_store=tmp_path,
        template_ds=xr.DataTree(xr.Dataset(coords={"time": [time]})),
        data_vars=[WAVE],
        append_dim="time",
        region=slice(0, 1),
        reformat_job_name="test-codec",
    )
    coord = NoaaUrmaAnalysisVirtualSourceFileCoord(time=time, data_vars=[WAVE])
    (ref,) = job.file_refs(coord, source_fixture["size"])
    raw = s3_read_bytes(
        ref.location, region="us-east-1", start=ref.offset, end=ref.offset + ref.length
    )
    store = MemoryStore()
    zarr.create_array(
        store=store,
        shape=(1, 1597, 2345),
        chunks=(1, 1597, 2345),
        dtype="float64",
        fill_value=np.nan,
        serializer=GribberishCodec(
            var="HTSGW", north_up=True, adjust_longitude_range=True
        ),
        compressors=(),
    )
    sync(store.set("c/0/0/0", default_buffer_prototype().buffer.from_bytes(raw)))
    array = np.asarray(zarr.open_array(store, mode="r")[:])[0]
    expected = next(m for m in source_fixture["messages"] if m["element"] == "HTSGW")
    assert array.dtype == np.float64
    assert array.shape == (1597, 2345)
    mask = np.isnan(array)
    assert int(mask.sum()) == expected["nan_count"]
    assert sha256(mask.tobytes()).hexdigest() == expected["nan_mask_sha256"]
    actual_values = [array[p["y"], p["x"]] for p in source_fixture["points"]]
    expected_values = [np.nan if v is None else v for v in expected["values"]]
    np.testing.assert_allclose(actual_values, expected_values, rtol=0, atol=1e-12)
