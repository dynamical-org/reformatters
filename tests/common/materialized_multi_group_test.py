"""End-to-end test for a MATERIALIZED dataset with vertical-dimension groups.

The materialized pipeline rechunks and rewrites bytes, so unlike the virtual case a
group var's chunks travel through a shared-memory buffer, a tmp zarr store and a
chunk-file copy before landing in the primary store. Both vertical groups here carry a
variable named ``temperature``: a pipeline keyed by bare variable name cannot tell the
two apart, so distinct per-group values catch a write landing in the wrong array.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import dask.array
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import template_utils
from reformatters.common.config_models import (
    BaseInternalAttrs,
    DataVar,
    DataVarAttrs,
    Encoding,
)
from reformatters.common.materialized_region_job import MaterializedRegionJob
from reformatters.common.region_job import SourceFileCoord
from reformatters.common.storage import (
    DatasetFormat,
    StorageConfig,
    StoreFactory,
    get_local_tmp_store,
)
from reformatters.common.types import ArrayFloat32, Timestamp

N_LAT = 2
N_LON = 3
PRESSURE_LEVELS = [1000.0, 850.0]
MODEL_LEVELS = [1, 2, 3]
N_TIME = 4
APPEND_DIM_START = pd.Timestamp("2025-01-01")
APPEND_DIM_FREQ = pd.Timedelta("1h")
SHARD_TIME = 2  # two shards along time
DATASET_ID = "test-materialized-multi-group-dataset"


def _values(base: float, shape: tuple[int, ...]) -> ArrayFloat32:
    return (base + np.arange(np.prod(shape))).reshape(shape).astype(np.float32)


def _root_values(time_idx: int) -> ArrayFloat32:
    return _values(100.0 * time_idx, (N_LAT, N_LON))


def _pressure_values(time_idx: int) -> ArrayFloat32:
    return _values(10_000.0 + 100.0 * time_idx, (N_LAT, N_LON, len(PRESSURE_LEVELS)))


def _model_values(time_idx: int) -> ArrayFloat32:
    return _values(50_000.0 + 100.0 * time_idx, (N_LAT, N_LON, len(MODEL_LEVELS)))


class MultiGroupDataVar(DataVar[BaseInternalAttrs]):
    attrs: DataVarAttrs = DataVarAttrs(
        units="K",
        long_name="Temperature",
        short_name="t",
        step_type="instant",
    )
    internal_attrs: BaseInternalAttrs = BaseInternalAttrs(
        keep_mantissa_bits="no-rounding"
    )


def _encoding_kwargs(n_levels: int | None) -> dict[str, Any]:
    if n_levels is None:
        return {
            "dtype": "float32",
            "fill_value": np.nan,
            "chunks": (1, N_LAT, N_LON),
            "shards": (SHARD_TIME, N_LAT, N_LON),
        }
    return {
        "dtype": "float32",
        "fill_value": np.nan,
        "chunks": (1, N_LAT, N_LON, 1),
        "shards": (SHARD_TIME, N_LAT, N_LON, n_levels),
    }


def _encoding(n_levels: int | None) -> Encoding:
    return Encoding(**_encoding_kwargs(n_levels))


def _data_vars() -> list[MultiGroupDataVar]:
    return [
        MultiGroupDataVar(name="temperature_2m", encoding=_encoding(None)),
        MultiGroupDataVar(
            name="temperature",
            group="pressure_level",
            encoding=_encoding(len(PRESSURE_LEVELS)),
        ),
        MultiGroupDataVar(
            name="temperature",
            group="model_level",
            encoding=_encoding(len(MODEL_LEVELS)),
        ),
    ]


def _create_template_ds() -> xr.DataTree:
    times = pd.date_range(APPEND_DIM_START, periods=N_TIME, freq=APPEND_DIM_FREQ)
    shared_coords = {
        "time": times,
        "latitude": np.arange(N_LAT, dtype="float64"),
        "longitude": np.arange(N_LON, dtype="float64"),
    }

    def _variable(level_dim: str | None, n_levels: int | None) -> xr.Variable:
        dims = ["time", "latitude", "longitude"]
        shape = [N_TIME, N_LAT, N_LON]
        if level_dim is not None:
            assert n_levels is not None
            dims.append(level_dim)
            shape.append(n_levels)
        return xr.Variable(
            dims,
            dask.array.full(tuple(shape), np.nan, dtype="float32", chunks=-1),
            encoding=_encoding_kwargs(n_levels),
        )

    root = xr.Dataset(
        {"temperature_2m": _variable(None, None)},
        coords=shared_coords,
        attrs={"dataset_id": DATASET_ID},
    )
    pressure = xr.Dataset(
        {"temperature": _variable("pressure_level", len(PRESSURE_LEVELS))},
        # Shared coords are duplicated into the group so it can be opened on its own.
        coords={**shared_coords, "pressure_level": PRESSURE_LEVELS},
    )
    model = xr.Dataset(
        {"temperature": _variable("model_level", len(MODEL_LEVELS))},
        coords={**shared_coords, "model_level": MODEL_LEVELS},
    )
    for ds in (root, pressure, model):
        ds["time"].encoding.update(
            {
                "dtype": "int64",
                "fill_value": -1,
                "units": "seconds since 1970-01-01 00:00:00",
                "calendar": "proleptic_gregorian",
            }
        )
        ds["latitude"].encoding["fill_value"] = np.nan
        ds["longitude"].encoding["fill_value"] = np.nan
    pressure["pressure_level"].encoding["fill_value"] = np.nan
    model["model_level"].encoding["fill_value"] = -1
    return xr.DataTree.from_dict(
        {"/": root, "/pressure_level": pressure, "/model_level": model}
    )


class MultiGroupSourceFileCoord(SourceFileCoord):
    time: Timestamp

    def get_url(self) -> str:
        return f"https://test.org/{self.time.isoformat()}"


class MultiGroupRegionJob(
    MaterializedRegionJob[MultiGroupDataVar, MultiGroupSourceFileCoord]
):
    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[MultiGroupDataVar],  # noqa: ARG002
    ) -> Sequence[MultiGroupSourceFileCoord]:
        return [
            MultiGroupSourceFileCoord(time=pd.Timestamp(time))
            for time in processing_region_ds[self.append_dim].values
        ]

    def download_file(self, coord: MultiGroupSourceFileCoord) -> Path:  # noqa: ARG002
        return Path("unused-source-file")

    def read_data(
        self,
        coord: MultiGroupSourceFileCoord,
        data_var: MultiGroupDataVar,
    ) -> ArrayFloat32:
        time_idx = int((coord.time - APPEND_DIM_START) / APPEND_DIM_FREQ)
        match data_var.path:
            case "temperature_2m":
                return _root_values(time_idx)
            case "pressure_level/temperature":
                return _pressure_values(time_idx)
            case "model_level/temperature":
                return _model_values(time_idx)
            case unknown:
                raise AssertionError(f"unexpected variable {unknown}")


@pytest.fixture
def store_factory() -> StoreFactory:
    return StoreFactory(
        primary_storage_config=StorageConfig(
            base_path="fake-prod-path", format=DatasetFormat.ZARR3
        ),
        dataset_id=DATASET_ID,
        template_config_version="v1.0",
    )


def _process_all_jobs(store_factory: StoreFactory) -> xr.DataTree:
    template_ds = _create_template_ds()
    template_utils.write_metadata(template_ds, store_factory)

    tmp_store = get_local_tmp_store()
    jobs = MultiGroupRegionJob.get_jobs(
        tmp_store=tmp_store,
        template_ds=template_ds,
        append_dim="time",
        all_data_vars=_data_vars(),
        reformat_job_name="test-job",
    )
    assert len(jobs) == N_TIME // SHARD_TIME

    primary_store = store_factory.primary_store(writable=True)
    replica_stores = store_factory.replica_stores(writable=True)
    for job in jobs:
        template_utils.write_metadata(job.template_ds, tmp_store)
        job.process(primary_store, replica_stores)

    store: Any = store_factory.primary_store()
    return xr.open_datatree(store, engine="zarr", chunks=None)


def test_materialized_multi_group_write_reads_back(
    store_factory: StoreFactory,
) -> None:
    tree = _process_all_jobs(store_factory)

    for time_idx in range(N_TIME):
        np.testing.assert_array_equal(
            tree.to_dataset()["temperature_2m"].isel(time=time_idx).values,
            _root_values(time_idx),
        )
        np.testing.assert_array_equal(
            tree["pressure_level"]
            .to_dataset()["temperature"]
            .isel(time=time_idx)
            .values,
            _pressure_values(time_idx),
        )
        np.testing.assert_array_equal(
            tree["model_level"].to_dataset()["temperature"].isel(time=time_idx).values,
            _model_values(time_idx),
        )


def test_materialized_group_opens_standalone(store_factory: StoreFactory) -> None:
    _process_all_jobs(store_factory)

    store: Any = store_factory.primary_store()
    pressure_ds = xr.open_zarr(store, group="pressure_level", chunks=None)
    assert set(pressure_ds["temperature"].dims) == {
        "time",
        "latitude",
        "longitude",
        "pressure_level",
    }
    np.testing.assert_array_equal(pressure_ds["pressure_level"].values, PRESSURE_LEVELS)
    np.testing.assert_array_equal(
        pressure_ds["temperature"].isel(time=0).values, _pressure_values(0)
    )
