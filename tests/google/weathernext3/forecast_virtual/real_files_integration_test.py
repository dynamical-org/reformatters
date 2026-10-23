import os

import httpx
import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.codecs import BytesCodec, ZstdCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import MemoryStore

from reformatters.google.weathernext3.forecast_virtual.dynamical_dataset import (
    GoogleWeathernext3ForecastVirtualDataset,
)
from reformatters.google.weathernext3.forecast_virtual.region_job import (
    GoogleWeathernext3ForecastVirtualSourceFileCoord,
)
from tests.google.weathernext3.forecast_virtual.datasets_test import DATASETS, job_for

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("WEATHERNEXT3_PROXY_TESTS") != "1",
        reason="requires the deployed WeatherNext 3 proxy",
    ),
]


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
@pytest.mark.parametrize("hour", [0, 1])
def test_real_source_conversion_and_etag(
    dataset: GoogleWeathernext3ForecastVirtualDataset, hour: int
) -> None:
    job = job_for(dataset)
    init = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour)
    with httpx.Client(timeout=120) as client:
        for var in dataset.template_config.data_vars:
            coord = GoogleWeathernext3ForecastVirtualSourceFileCoord(
                init_time=init, lead_time=pd.Timedelta("1h"), data_vars=(var,)
            )
            available = job.discover_available([coord])
            assert available == [(coord, 0)]
            location = coord.chunk_location(var, "mean")
            metadata = coord.chunk_metadata[location]
            response = client.get(
                location, headers={"If-Match": metadata.etag_checksum}
            )
            response.raise_for_status()
            assert len(response.content) == metadata.size
            stale = client.get(
                location, headers={"If-Match": '"stale"', "Range": "bytes=0-15"}
            )
            assert stale.status_code == 412
            chunks = var.encoding.chunks
            assert isinstance(chunks, tuple)
            source_shape = chunks[-3:]
            source = zarr.create_array(
                store=MemoryStore(),
                shape=source_shape,
                chunks=source_shape,
                dtype="float32",
                serializer=BytesCodec(endian="little"),
                compressors=[ZstdCodec(level=0, checksum=False)],
            )
            payload = default_buffer_prototype().buffer.from_bytes(response.content)
            sync(source.store.set("c/0/0/0", payload))
            target = zarr.create_array(
                store=MemoryStore(),
                shape=chunks,
                chunks=chunks,
                dtype="float32",
                serializer=BytesCodec(endian="little"),
                compressors=[ZstdCodec(level=0, checksum=False)],
                filters=var.encoding.filters,
            )
            sync(target.store.set("c/0/0/0/0/0", payload))
            raw = np.asarray(source[:])
            expected = raw
            if var.encoding.filters:
                conversion = var.encoding.filters[0]["configuration"]
                expected = raw / conversion.get("scale", 1) + conversion.get(
                    "offset", 0
                )
            np.testing.assert_allclose(
                np.asarray(target[:]).reshape(raw.shape),
                expected,
                rtol=1e-6,
                atol=1e-4,
                equal_nan=True,
            )
            if var.name == "sea_surface_temperature":
                y = round((25 + 90) / dataset.template_config.grid_degrees)
                x = round(10 / dataset.template_config.grid_degrees)
                assert np.isnan(raw[0, y, x])
