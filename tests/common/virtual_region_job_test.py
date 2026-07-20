"""Tests for VirtualRegionJob and the virtual fork in _process_region_jobs.

The synthetic dataset below stores its "GRIB messages" as raw little-endian
float64 blocks in a local file and points virtual refs at byte ranges within it,
using a local-filesystem virtual chunk container. The data var uses zarr's
default BytesCodec (no GribberishCodec), so a real value read-back round-trip
catches bad chunk keys *and* bad byte ranges without needing real GRIB. A
separate test pins that a GribberishCodec serializer threads through dimension
expansion without the decode-only codec ever being invoked.
"""

import asyncio
import json
from collections.abc import Iterator, Mapping, Sequence
from datetime import timedelta
from itertools import batched
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import dask.array
import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from gribberish.zarr import GribberishCodec
from pydantic import ValidationError, computed_field
from zarr.core.buffer import default_buffer_prototype
from zarr.core.metadata import ArrayV3Metadata

from reformatters.common import template_utils, validation
from reformatters.common.config_models import (
    ROOT,
    BaseInternalAttrs,
    DataVar,
    DataVarAttrs,
    Encoding,
    Group,
)
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.iterating import get_worker_jobs
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.region_job import (
    CoordinateValue,
    SourceFileCoord,
)
from reformatters.common.storage import (
    DatasetFormat,
    IcechunkVirtualConfig,
    StorageConfig,
    StoreFactory,
    _virtual_repository_config_and_credentials,
    manifest_append_dim_split,
)
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.virtual_region_job import (
    VirtualRef,
    VirtualRegionJob,
    _exists_many,
)

pytestmark = pytest.mark.slow

N_LAT = 2
N_LON = 3
LEAD_TIMES = pd.timedelta_range("0h", periods=2, freq="6h")
N_LEADS = len(LEAD_TIMES)
BLOCK_NBYTES = N_LAT * N_LON * 8  # one float64 message
APPEND_DIM_START = pd.Timestamp("2024-01-01")
APPEND_DIM_FREQ = pd.Timedelta("6h")
DATASET_ID = "test-virtual-dataset"


def _block_values(init_idx: int, lead_idx: int) -> np.ndarray:
    """Deterministic per-message values; distinct per (init, lead)."""
    base = 1000.0 * init_idx + 10.0 * lead_idx
    return (base + np.arange(N_LAT * N_LON)).reshape(N_LAT, N_LON).astype("<f8")


def _message_offset_length(init_idx: int, lead_idx: int) -> tuple[int, int]:
    order = init_idx * N_LEADS + lead_idx
    return order * BLOCK_NBYTES, BLOCK_NBYTES


def _write_messages_file(path: Path, n_inits: int) -> None:
    blocks = [
        _block_values(init_idx, lead_idx).tobytes()
        for init_idx in range(n_inits)
        for lead_idx in range(N_LEADS)
    ]
    path.write_bytes(b"".join(blocks))


def _create_template_ds(
    n_inits: int,
    *,
    serializer: dict[str, Any] | None = None,
    chunks: tuple[int, ...] = (1, 1, N_LAT, N_LON),
) -> xr.DataTree:
    """Forecast-shaped virtual template (no shards; one chunk per message)."""
    init_times = pd.date_range(APPEND_DIM_START, periods=n_inits, freq=APPEND_DIM_FREQ)
    encoding: dict[str, Any] = {
        "dtype": "float64",
        "chunks": chunks,
        "fill_value": np.nan,
        "compressors": None,
        "filters": None,
    }
    if serializer is not None:
        encoding["serializer"] = serializer
    ds = xr.Dataset(
        {
            "temperature_2m": xr.Variable(
                ("init_time", "lead_time", "latitude", "longitude"),
                dask.array.full(
                    (n_inits, N_LEADS, N_LAT, N_LON),
                    np.nan,
                    dtype="float64",
                    chunks=-1,
                ),
                encoding=encoding,
            )
        },
        coords={
            "init_time": ("init_time", init_times),
            "lead_time": ("lead_time", LEAD_TIMES),
            "latitude": ("latitude", np.arange(N_LAT, dtype="float64")),
            "longitude": ("longitude", np.arange(N_LON, dtype="float64")),
            "valid_time": (
                ("init_time", "lead_time"),
                init_times.values[:, None] + LEAD_TIMES.values[None, :],
            ),
            # Like prod spatial_ref: value equals fill_value, so any writer using
            # write_empty_chunks=False deletes the chunk instead of writing it.
            "spatial_ref": ((), np.int64(0)),
        },
        attrs={"dataset_id": DATASET_ID, "dataset_version": "v1.0"},
    )
    # Explicit time encodings so dimension-appends re-encode consistently.
    time_encoding = {
        "dtype": "int64",
        "fill_value": -1,
        "units": "seconds since 1970-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
    }
    # Fixed append-dim coord chunk sizes (like prod) so growth never changes the
    # chunk grid and a metadata refresh render is byte-identical to the store.
    ds["init_time"].encoding.update({**time_encoding, "chunks": (8,)})
    ds["valid_time"].encoding.update({**time_encoding, "chunks": (8, N_LEADS)})
    ds["lead_time"].encoding.update(
        {"dtype": "int64", "fill_value": -1, "units": "seconds"}
    )
    ds["latitude"].encoding["fill_value"] = np.nan
    ds["longitude"].encoding["fill_value"] = np.nan
    ds["spatial_ref"].encoding.update({"dtype": "int64", "fill_value": 0})
    return xr.DataTree.from_dict({"/": ds})


# --- synthetic virtual forecast dataset ---


class VirtualTestDataVar(DataVar[BaseInternalAttrs]):
    encoding: Encoding = Encoding(
        dtype="float64",
        fill_value=np.nan,
        chunks=(1, 1, N_LAT, N_LON),  # one chunk per (init, lead) message
        shards=None,
        compressors=(),
        filters=None,
    )
    attrs: DataVarAttrs = DataVarAttrs(
        units="K",
        long_name="Temperature 2m",
        short_name="2t",
        step_type="instant",
    )
    internal_attrs: BaseInternalAttrs = BaseInternalAttrs(
        keep_mantissa_bits="no-rounding"
    )


class VirtualTestSourceFileCoord(SourceFileCoord):
    init_time: Timestamp
    lead_time: Timedelta

    def get_url(self) -> str:
        return VirtualTestRegionJob.messages_url


class VirtualTestRegionJob(
    VirtualRegionJob[VirtualTestDataVar, VirtualTestSourceFileCoord]
):
    # Set per test by _make_dataset; the generator points every ref at this file.
    messages_url: ClassVar[str] = ""
    # Whole files per yielded batch during backfill (operational yields 1 at a time).
    backfill_batch_files: ClassVar[int] = 2

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[VirtualTestDataVar],  # noqa: ARG002
    ) -> Sequence[VirtualTestSourceFileCoord]:
        return [
            VirtualTestSourceFileCoord(
                init_time=pd.Timestamp(init_time), lead_time=lead
            )
            for init_time in processing_region_ds["init_time"].values
            for lead in LEAD_TIMES
        ]

    def process_virtual_refs(
        self,
        remaining: Sequence[VirtualTestSourceFileCoord],
    ) -> Iterator[Sequence[tuple[VirtualTestSourceFileCoord, Sequence[VirtualRef]]]]:
        data_var = self.data_vars[0]
        template_root = self.template_ds.to_dataset()
        init_index = template_root.get_index("init_time")
        lead_index = template_root.get_index("lead_time")
        for group in batched(remaining, self.backfill_batch_files, strict=False):
            batch: list[tuple[VirtualTestSourceFileCoord, Sequence[VirtualRef]]] = []
            for coord in group:
                init_idx = int(init_index.get_indexer(pd.Index([coord.init_time]))[0])
                lead_idx = int(lead_index.get_indexer(pd.Index([coord.lead_time]))[0])
                offset, length = _message_offset_length(init_idx, lead_idx)
                ref = VirtualRef(
                    data_var=data_var,
                    out_loc=coord.out_loc(),
                    location=self.messages_url,
                    offset=offset,
                    length=length,
                )
                batch.append((coord, [ref]))
            yield batch


class VirtualTestTemplateConfig(TemplateConfig[VirtualTestDataVar]):
    dims: dict[Group, tuple[Dim, ...]] = {
        ROOT: ("init_time", "lead_time", "latitude", "longitude")
    }
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = APPEND_DIM_START
    append_dim_frequency: Timedelta = APPEND_DIM_FREQ

    @computed_field
    @property
    def dataset_id(self) -> str:
        return DATASET_ID

    @computed_field
    @property
    def version(self) -> str:
        return "v1.0"

    @computed_field
    @property
    def data_vars(self) -> Sequence[VirtualTestDataVar]:
        return [VirtualTestDataVar(name="temperature_2m")]


class VirtualTestDataset(
    DynamicalDataset[VirtualTestDataVar, VirtualTestSourceFileCoord]
):
    template_config: VirtualTestTemplateConfig = VirtualTestTemplateConfig()
    region_job_class: type[VirtualTestRegionJob] = VirtualTestRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        return [
            ReformatCronJob(
                name=f"{self.dataset_id}-update",
                schedule="0 0 * * *",
                pod_active_deadline=timedelta(minutes=5),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1",
                memory="1G",
                shared_memory="1G",
                ephemeral_storage="1G",
                secret_names=self.store_factory.k8s_secret_names(),
            ),
            ValidationCronJob(
                name=f"{self.dataset_id}-validate",
                schedule="0 1 * * *",
                pod_active_deadline=timedelta(minutes=5),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1",
                memory="1G",
                shared_memory="1G",
                ephemeral_storage="1G",
                secret_names=self.store_factory.k8s_secret_names(),
            ),
        ]

    def validators(self) -> Sequence[validation.DataValidator]:
        return ()


def _make_dataset(tmp_path: Path, *, n_inits: int = 4) -> VirtualTestDataset:
    """Build the dataset, write the messages file, and point the region job at it.

    The virtual chunk container is a local-filesystem store rooted at tmp_path;
    refs point at tmp_path/messages.bin.
    """
    messages_path = tmp_path / "messages.bin"
    _write_messages_file(messages_path, n_inits=max(n_inits, 4))
    VirtualTestRegionJob.messages_url = f"file://{messages_path}"
    VirtualTestRegionJob.backfill_batch_files = 2

    container = icechunk.VirtualChunkContainer(
        f"file://{tmp_path}/", icechunk.local_filesystem_store(str(tmp_path))
    )
    return VirtualTestDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
        icechunk_virtual_config=IcechunkVirtualConfig(
            containers=(container,),
            manifest_split=manifest_append_dim_split(split_size=2, dim="init_time"),
        ),
    )


def _make_region_job(
    template_ds: xr.DataTree,
    *,
    region: slice,
    processing_mode: Literal["backfill", "update"] = "backfill",
) -> VirtualTestRegionJob:
    return VirtualTestRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=[VirtualTestDataVar(name="temperature_2m")],
        append_dim="init_time",
        region=region,
        reformat_job_name="test",
        processing_mode=processing_mode,
    )


def _primary_repo(factory: StoreFactory) -> icechunk.Repository:
    return factory.icechunk_repos(sort="primary-first")[0][1]


def _snapshot_count(repo: icechunk.Repository, branch: str = "main") -> int:
    return sum(1 for _ in repo.ancestry(branch=branch))


def _main_store_bytes(factory: StoreFactory, key: str) -> bytes | None:
    store = _primary_repo(factory).readonly_session("main").store
    buffer = asyncio.run(store.get(key, prototype=default_buffer_prototype()))
    return None if buffer is None else bytes(buffer.to_bytes())


def _process_virtual(
    job: VirtualTestRegionJob,
    primary_repo: icechunk.Repository,
    replica_repos: Sequence[icechunk.Repository] = (),
    branch: str = "main",
) -> None:
    """Gather remaining coords (as process_worker_jobs does), then drive the write loop."""
    readonly = primary_repo.readonly_session(branch).store
    remaining = job.filter_already_present(job.source_file_coords(), readonly)
    job.process_virtual(primary_repo, list(replica_repos), branch, remaining)


def _assert_store_values(store: object, n_inits: int) -> None:
    result = xr.open_zarr(store, decode_timedelta=True)
    assert result.sizes["init_time"] == n_inits
    for init_idx in range(n_inits):
        for lead_idx in range(N_LEADS):
            np.testing.assert_array_equal(
                result["temperature_2m"]
                .isel(init_time=init_idx, lead_time=lead_idx)
                .values,
                _block_values(init_idx, lead_idx),
            )


def _assert_all_values(dataset: VirtualTestDataset, n_inits: int) -> None:
    _assert_store_values(dataset.store_factory.primary_store(), n_inits)


def _make_replica_repo(
    tmp_path: Path, dataset: VirtualTestDataset
) -> icechunk.Repository:
    """An independent icechunk repo (the test harness collapses a dataset's own
    primary+replica to one path) registered with the same virtual container."""
    repo_config, credentials = _virtual_repository_config_and_credentials(
        dataset.icechunk_virtual_config
    )
    return icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "replica.icechunk")),
        config=repo_config,
        authorize_virtual_chunk_access=credentials,
    )


# --- unit tests (chunk_key, sizing, commit guard) ---


def test_chunk_key_maps_labels_to_index() -> None:
    job = _make_region_job(_create_template_ds(4), region=slice(0, 4))
    var = job.data_vars[0]
    out_loc: Mapping[Dim, CoordinateValue] = {
        "init_time": APPEND_DIM_START + 2 * APPEND_DIM_FREQ,
        "lead_time": LEAD_TIMES[1],
    }
    # init index 2, lead index 1; lat/lon are single full-width chunks at 0.
    assert job.chunk_key(out_loc, var) == (2, 1, 0, 0)


def test_chunk_key_returns_none_for_unknown_label() -> None:
    job = _make_region_job(_create_template_ds(4), region=slice(0, 4))
    out_loc: Mapping[Dim, CoordinateValue] = {
        "init_time": pd.Timestamp("2030-01-01"),  # not in the template's coords
        "lead_time": LEAD_TIMES[0],
    }
    assert job.chunk_key(out_loc, job.data_vars[0]) is None


def test_chunk_key_asserts_on_unaligned_position() -> None:
    # Template whose lead_time chunk size is 2; lead index 1 falls mid-chunk.
    job = _make_region_job(
        _create_template_ds(4, chunks=(1, 2, N_LAT, N_LON)), region=slice(0, 4)
    )
    out_loc: Mapping[Dim, CoordinateValue] = {
        "init_time": APPEND_DIM_START,
        "lead_time": LEAD_TIMES[1],
    }
    with pytest.raises(AssertionError, match="chunk boundary"):
        job.chunk_key(out_loc, job.data_vars[0])


def test_chunk_key_uses_template_geometry_not_datavar_encoding() -> None:
    # The checked-in template is authoritative: a DataVar.encoding that disagrees
    # with it must be ignored so filtering and emission can't drift from readers.
    job = _make_region_job(
        _create_template_ds(4, chunks=(1, 1, N_LAT, N_LON)), region=slice(0, 4)
    )
    misconfigured_var = VirtualTestDataVar(
        name="temperature_2m",
        encoding=Encoding(
            dtype="float64",
            fill_value=np.nan,
            chunks=(1, 2, N_LAT, N_LON),  # disagrees with the template's (1, 1, ...)
            shards=None,
            compressors=None,
            filters=None,
        ),
    )
    out_loc: Mapping[Dim, CoordinateValue] = {
        "init_time": APPEND_DIM_START,
        "lead_time": LEAD_TIMES[1],
    }
    # Uses the template's chunk size of 1 (lead index 1 -> chunk 1), not the
    # DataVar's 2 (which would assert on the mid-chunk position).
    assert job.chunk_key(out_loc, misconfigured_var) == (0, 1, 0, 0)


def _create_ensemble_template_ds(
    n_inits: int, *, ensemble_members: tuple[int, ...] = (0, 1, 2)
) -> xr.DataTree:
    """Like _create_template_ds, but with a plain-int labeled dim (ensemble_member)
    in place of lead_time, to exercise non-Timestamp/Timedelta label lookups."""
    init_times = pd.date_range(APPEND_DIM_START, periods=n_inits, freq=APPEND_DIM_FREQ)
    encoding: dict[str, Any] = {
        "dtype": "float64",
        "chunks": (1, 1, N_LAT, N_LON),
        "fill_value": np.nan,
        "compressors": None,
        "filters": None,
    }
    ds = xr.Dataset(
        {
            "temperature_2m": xr.Variable(
                ("init_time", "ensemble_member", "latitude", "longitude"),
                dask.array.full(
                    (n_inits, len(ensemble_members), N_LAT, N_LON),
                    np.nan,
                    dtype="float64",
                    chunks=-1,
                ),
                encoding=encoding,
            )
        },
        coords={
            "init_time": ("init_time", init_times),
            "ensemble_member": ("ensemble_member", np.array(ensemble_members)),
            "latitude": ("latitude", np.arange(N_LAT, dtype="float64")),
            "longitude": ("longitude", np.arange(N_LON, dtype="float64")),
        },
        attrs={"dataset_id": DATASET_ID, "dataset_version": "v1.0"},
    )
    ds["init_time"].encoding.update(
        {
            "dtype": "int64",
            "fill_value": -1,
            "units": "seconds since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        }
    )
    ds["ensemble_member"].encoding["fill_value"] = -1
    ds["latitude"].encoding["fill_value"] = np.nan
    ds["longitude"].encoding["fill_value"] = np.nan
    return xr.DataTree.from_dict({"/": ds})


def test_chunk_key_int_labeled_dim_resolves_and_handles_absent_label() -> None:
    # Non-Timestamp/Timedelta labeled dim: dict-based lookups must hash/compare
    # int coordinate labels consistently, not just datetime-like ones.
    template_ds = _create_ensemble_template_ds(4, ensemble_members=(0, 1, 2))
    job = _make_region_job(template_ds, region=slice(0, 4))
    var = job.data_vars[0]

    present: Mapping[Dim, CoordinateValue] = {
        "init_time": APPEND_DIM_START + APPEND_DIM_FREQ,
        "ensemble_member": 2,
    }
    assert job.chunk_key(present, var) == (1, 2, 0, 0)

    absent: Mapping[Dim, CoordinateValue] = {
        "init_time": APPEND_DIM_START,
        "ensemble_member": 99,  # not in the template's ensemble_member coord
    }
    assert job.chunk_key(absent, var) is None

    # Repeat the present lookup to prove any cache built on first use is stable.
    assert job.chunk_key(present, var) == (1, 2, 0, 0)


def test_chunk_key_repeated_calls_same_var_different_out_loc() -> None:
    # A cache keyed on var.path alone (ignoring out_loc) would return stale
    # results; each distinct out_loc must resolve independently and correctly.
    job = _make_region_job(_create_template_ds(4), region=slice(0, 4))
    var = job.data_vars[0]

    for init_idx in range(4):
        for lead_idx, lead in enumerate(LEAD_TIMES):
            out_loc: Mapping[Dim, CoordinateValue] = {
                "init_time": APPEND_DIM_START + init_idx * APPEND_DIM_FREQ,
                "lead_time": lead,
            }
            assert job.chunk_key(out_loc, var) == (init_idx, lead_idx, 0, 0)

    # And querying an already-seen out_loc again still gives the same answer.
    out_loc = {
        "init_time": APPEND_DIM_START + 2 * APPEND_DIM_FREQ,
        "lead_time": LEAD_TIMES[1],
    }
    assert job.chunk_key(out_loc, var) == (2, 1, 0, 0)
    assert job.chunk_key(out_loc, var) == (2, 1, 0, 0)


def test_chunk_key_two_datavars_same_path_different_encoding_in_sequence() -> None:
    # Any per-var.path geometry cache must key strictly off the template, not
    # off whichever DataVar object happened to populate it first.
    job = _make_region_job(
        _create_template_ds(4, chunks=(1, 1, N_LAT, N_LON)), region=slice(0, 4)
    )
    correct_var = job.data_vars[0]
    misconfigured_var = VirtualTestDataVar(
        name="temperature_2m",
        encoding=Encoding(
            dtype="float64",
            fill_value=np.nan,
            chunks=(1, 2, N_LAT, N_LON),  # disagrees with the template's (1, 1, ...)
            shards=None,
            compressors=None,
            filters=None,
        ),
    )
    out_loc: Mapping[Dim, CoordinateValue] = {
        "init_time": APPEND_DIM_START,
        "lead_time": LEAD_TIMES[1],
    }
    # correct_var first (populates any cache), then misconfigured_var: both must
    # resolve against the template's chunk size of 1, not either var's own encoding.
    assert job.chunk_key(out_loc, correct_var) == (0, 1, 0, 0)
    assert job.chunk_key(out_loc, misconfigured_var) == (0, 1, 0, 0)
    # And again in the opposite order, to rule out order-dependent cache seeding.
    job2 = _make_region_job(
        _create_template_ds(4, chunks=(1, 1, N_LAT, N_LON)), region=slice(0, 4)
    )
    assert job2.chunk_key(out_loc, misconfigured_var) == (0, 1, 0, 0)
    assert job2.chunk_key(out_loc, correct_var) == (0, 1, 0, 0)


def test_chunk_key_separate_job_instances_do_not_share_cache() -> None:
    # Two jobs over different templates (different init_time chunk sizes) must
    # not cross-contaminate any var.path-keyed cache.
    job_chunk1 = _make_region_job(
        _create_template_ds(4, chunks=(1, 1, N_LAT, N_LON)), region=slice(0, 4)
    )
    job_chunk2 = _make_region_job(
        _create_template_ds(4, chunks=(2, 1, N_LAT, N_LON)), region=slice(0, 4)
    )
    var = job_chunk1.data_vars[0]
    out_loc: Mapping[Dim, CoordinateValue] = {
        "init_time": APPEND_DIM_START,
        "lead_time": LEAD_TIMES[0],
    }
    # Warm job_chunk1's cache first.
    assert job_chunk1.chunk_key(out_loc, var) == (0, 0, 0, 0)
    # job_chunk2's init_time chunk size is 2: init index 2 lands at the start of
    # chunk index 1, distinct from job_chunk1's chunk index 2 for the same position.
    out_loc_init2: Mapping[Dim, CoordinateValue] = {
        "init_time": APPEND_DIM_START + 2 * APPEND_DIM_FREQ,
        "lead_time": LEAD_TIMES[0],
    }
    assert job_chunk2.chunk_key(out_loc_init2, job_chunk2.data_vars[0]) == (1, 0, 0, 0)
    # Re-querying job_chunk1 for the same position must still use its own
    # (1, 1, ...) chunking, unaffected by job_chunk2's lookups in between.
    assert job_chunk1.chunk_key(out_loc_init2, var) == (2, 0, 0, 0)


def test_resolve_chunk_keys_matches_chunk_key_and_batches_lookups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # _resolve_chunk_keys (used by _emit_refs and filter_already_present) must
    # return exactly what calling chunk_key once per item would, while touching the
    # template only once per var per call regardless of how many items (or labeled
    # dims) share that var -- not once per item.
    job = _make_region_job(_create_template_ds(4), region=slice(0, 4))
    var = job.data_vars[0]
    out_locs: list[Mapping[Dim, CoordinateValue]] = [
        {
            "init_time": APPEND_DIM_START + init_idx * APPEND_DIM_FREQ,
            "lead_time": LEAD_TIMES[0],
        }
        for init_idx in range(4)
    ]
    expected = [job.chunk_key(out_loc, var) for out_loc in out_locs]

    call_count = 0
    original_getitem = xr.DataTree.__getitem__

    def counting_getitem(self: xr.DataTree, key: str) -> xr.DataTree | xr.DataArray:
        nonlocal call_count
        call_count += 1
        return original_getitem(self, key)

    monkeypatch.setattr(xr.DataTree, "__getitem__", counting_getitem)
    results = job._resolve_chunk_keys([(out_loc, var) for out_loc in out_locs])

    assert results == expected
    # One template lookup for the whole group (dims/chunks/sizes and both labeled
    # dims' positions all reuse it), not one per item (4).
    assert call_count == 1


def test_resolve_chunk_keys_rejects_inconsistent_labeled_dims() -> None:
    # Every ref for a given var must label the same set of dims -- if it didn't,
    # peeking at the first item's labels to decide which dims to vectorize over
    # would silently miscompute the rest. Fail loudly instead.
    job = _make_region_job(_create_template_ds(4), region=slice(0, 4))
    var = job.data_vars[0]
    items: list[tuple[Mapping[Dim, CoordinateValue], VirtualTestDataVar]] = [
        ({"init_time": APPEND_DIM_START, "lead_time": LEAD_TIMES[0]}, var),
        ({"init_time": APPEND_DIM_START + APPEND_DIM_FREQ}, var),
    ]
    with pytest.raises(AssertionError, match="must label the same set of dims"):
        job._resolve_chunk_keys(items)


def test_resolve_chunk_keys_multi_var_interleaved_preserves_order() -> None:
    # _resolve_chunk_keys groups items by var.path internally, then must scatter
    # results back into the caller's original order. With items for only one var, a
    # stable sort leaves order unchanged, so a bug that swapped which item's result
    # went where (e.g. writing to the group-local index instead of the original
    # item index) wouldn't be caught. Two vars with DIFFERENT init_time chunk sizes,
    # interleaved and queried at the SAME position, produce genuinely different
    # correct answers -- exactly what's needed to catch a reassembly bug.
    template_ds = _create_template_ds(4, chunks=(1, 1, N_LAT, N_LON))
    ds = template_ds.to_dataset()
    ds["dewpoint_2m"] = ds["temperature_2m"].copy(deep=False)
    ds["dewpoint_2m"].encoding = {
        **ds["temperature_2m"].encoding,
        "chunks": (2, 1, N_LAT, N_LON),
    }
    template_ds = xr.DataTree.from_dict({"/": ds})

    var_a = VirtualTestDataVar(name="temperature_2m")  # init_time chunk size 1
    var_b = VirtualTestDataVar(name="dewpoint_2m")  # init_time chunk size 2
    job = VirtualTestRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=[var_a, var_b],
        append_dim="init_time",
        region=slice(0, 4),
        reformat_job_name="test",
    )

    def out_loc(init_idx: int) -> Mapping[Dim, CoordinateValue]:
        return {
            "init_time": APPEND_DIM_START + init_idx * APPEND_DIM_FREQ,
            "lead_time": LEAD_TIMES[0],
        }

    # Interleaved, not grouped by var -- init index 2 lands on a chunk boundary for
    # both vars (chunk sizes 1 and 2), so it's safe to query for both while still
    # giving different chunk indices (2 vs. 1).
    items = [
        (out_loc(2), var_b),  # -> (1, 0, 0, 0)
        (out_loc(0), var_a),  # -> (0, 0, 0, 0)
        (out_loc(2), var_a),  # -> (2, 0, 0, 0)
        (out_loc(0), var_b),  # -> (0, 0, 0, 0)
    ]
    results = job._resolve_chunk_keys(items)

    assert results == [(1, 0, 0, 0), (0, 0, 0, 0), (2, 0, 0, 0), (0, 0, 0, 0)]
    assert results == [job.chunk_key(out_loc, var) for out_loc, var in items]


def test_resolve_chunk_keys_mixed_present_and_absent_in_one_group() -> None:
    # Within a single var's vectorized group, an absent label must not corrupt a
    # neighboring present item's resolved index (np.where clamps the absent
    # position to 0 before divmod so it never raises, and the boundary assert only
    # checks the present subset), and must land as None at exactly its own slot.
    job = _make_region_job(_create_template_ds(4), region=slice(0, 4))
    var = job.data_vars[0]
    items: list[tuple[Mapping[Dim, CoordinateValue], VirtualTestDataVar]] = [
        ({"init_time": APPEND_DIM_START, "lead_time": LEAD_TIMES[0]}, var),
        (
            {"init_time": pd.Timestamp("2030-01-01"), "lead_time": LEAD_TIMES[0]},
            var,
        ),  # not in the template's coords
        (
            {
                "init_time": APPEND_DIM_START + 2 * APPEND_DIM_FREQ,
                "lead_time": LEAD_TIMES[1],
            },
            var,
        ),
    ]
    results = job._resolve_chunk_keys(items)

    assert results == [(0, 0, 0, 0), None, (2, 1, 0, 0)]
    assert results == [job.chunk_key(out_loc, v) for out_loc, v in items]


def test_resolve_chunk_keys_asserts_absent_multi_chunk_dim() -> None:
    # lead_time spans multiple chunks (size 2, chunk size 1); an out_loc without it
    # would silently collapse every ref to lead chunk 0.
    job = _make_region_job(_create_template_ds(4), region=slice(0, 4))
    out_loc: Mapping[Dim, CoordinateValue] = {"init_time": APPEND_DIM_START}
    with pytest.raises(AssertionError, match="absent from out_loc"):
        job.chunk_key(out_loc, job.data_vars[0])


def test_needed_append_dim_size() -> None:
    job = _make_region_job(_create_template_ds(4), region=slice(0, 4))
    refs = [
        VirtualRef(
            VirtualTestDataVar(name="temperature_2m"),
            {
                "init_time": APPEND_DIM_START + 2 * APPEND_DIM_FREQ,
                "lead_time": LEAD_TIMES[0],
            },
            "file://x",
            0,
            BLOCK_NBYTES,
        )
    ]
    assert job._needed_append_dim_size(refs) == 3  # init index 2 -> size 3


def test_needed_append_dim_size_asserts_label_beyond_template() -> None:
    # Growing to fit such a ref would size the append dim past the labels the
    # template can supply coordinate values for.
    job = _make_region_job(_create_template_ds(4), region=slice(0, 4))
    refs = [
        VirtualRef(
            VirtualTestDataVar(name="temperature_2m"),
            {"init_time": pd.Timestamp("2030-01-01"), "lead_time": LEAD_TIMES[0]},
            "file://x",
            0,
            BLOCK_NBYTES,
        )
    ]
    with pytest.raises(AssertionError, match="not present in the template"):
        job._needed_append_dim_size(refs)


def test_virtual_get_jobs_regions_in_append_dim_order() -> None:
    # Virtual jobs are not spread: contiguous worker blocks keep each flush's
    # manifest-window rewrites bounded, see docs/parallel_processing.md.
    assert VirtualTestRegionJob.worker_assignment == "contiguous"
    jobs = VirtualTestRegionJob.get_jobs(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=_create_template_ds(8),
        append_dim="init_time",
        all_data_vars=[VirtualTestDataVar(name="temperature_2m")],
        reformat_job_name="test",
    )
    assert [j.region for j in jobs] == [slice(i, i + 1) for i in range(8)]


def test_virtual_worker_jobs_are_contiguous_along_append_dim() -> None:
    jobs = VirtualTestRegionJob.get_jobs(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=_create_template_ds(8),
        append_dim="init_time",
        all_data_vars=[VirtualTestDataVar(name="temperature_2m")],
        reformat_job_name="test",
    )
    worker_jobs = get_worker_jobs(
        jobs, worker_index=1, workers_total=3, worker_assignment="contiguous"
    )
    assert [j.region for j in worker_jobs] == [slice(3, 4), slice(4, 5), slice(5, 6)]


def test_processing_region_rejects_buffered_region() -> None:
    # Buffering is meaningless for virtual datasets and would make adjacent
    # backfill workers emit refs into each other's regions.
    class BufferedJob(VirtualTestRegionJob):
        def get_processing_region(self) -> slice:
            return slice(self.region.start - 1, self.region.stop + 1)

    job = BufferedJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=_create_template_ds(4),
        data_vars=[VirtualTestDataVar(name="temperature_2m")],
        append_dim="init_time",
        region=slice(1, 3),
        reformat_job_name="test",
    )
    with pytest.raises(AssertionError, match="get_processing_region must equal"):
        job._processing_region_ds()


def test_process_virtual_rejects_empty_batch(tmp_path: Path) -> None:
    # The generator contract is one commit's worth of whole files per yield, so an
    # empty batch is a bug (and an empty icechunk commit would raise). The loop
    # asserts it loudly rather than silently no-opping.
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    template_utils.write_metadata(template_ds, dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)

    class EmptyBatchJob(VirtualTestRegionJob):
        def process_virtual_refs(
            self,
            remaining: Sequence[VirtualTestSourceFileCoord],  # noqa: ARG002
        ) -> Iterator[
            Sequence[tuple[VirtualTestSourceFileCoord, Sequence[VirtualRef]]]
        ]:
            yield []

    job = EmptyBatchJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=[VirtualTestDataVar(name="temperature_2m")],
        append_dim="init_time",
        region=slice(0, 4),
        reformat_job_name="test",
    )
    with pytest.raises(AssertionError, match="empty batch"):
        _process_virtual(job, repo)


def test_process_virtual_rejects_refs_missing_probe_chunk(tmp_path: Path) -> None:
    # A file's refs must cover the chunk filter_already_present probes, or the
    # filter never sees the file land and re-ingests it forever.
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    template_utils.write_metadata(template_ds, dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)

    class WrongChunkJob(VirtualTestRegionJob):
        def process_virtual_refs(
            self,
            remaining: Sequence[VirtualTestSourceFileCoord],
        ) -> Iterator[
            Sequence[tuple[VirtualTestSourceFileCoord, Sequence[VirtualRef]]]
        ]:
            coord = remaining[0]
            ref = VirtualRef(
                data_var=self.data_vars[0],
                # A different lead than the coord's file covers.
                out_loc={"init_time": coord.init_time, "lead_time": LEAD_TIMES[1]},
                location=self.messages_url,
                offset=0,
                length=8,
            )
            yield [(coord, [ref])]

    job = WrongChunkJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=[VirtualTestDataVar(name="temperature_2m")],
        append_dim="init_time",
        region=slice(0, 4),
        reformat_job_name="test",
    )
    with pytest.raises(AssertionError, match="do not cover representative chunk"):
        _process_virtual(job, repo)


def test_emit_refs_rejects_unregistered_container_location(tmp_path: Path) -> None:
    # icechunk silently reports (rather than raises on) refs outside every registered
    # VirtualChunkContainer prefix; _emit_refs must fail loudly on them, or the
    # chunks would be unreadable.
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(1)
    template_utils.write_metadata(template_ds, dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)
    job = _make_region_job(template_ds, region=slice(0, 1))

    session = repo.writable_session("main")
    ref = VirtualRef(
        job.data_vars[0],
        {"init_time": APPEND_DIM_START, "lead_time": LEAD_TIMES[0]},
        "file:///not-a-registered-container/messages.bin",
        0,
        BLOCK_NBYTES,
    )
    with pytest.raises(AssertionError, match="registered container"):
        job._emit_refs([session.store], [ref])


def test_virtual_operational_rejects_backfill_mode_job(tmp_path: Path) -> None:
    # Without "update" the job sweeps once instead of polling; the driver
    # asserts operational jobs are constructed to poll.
    dataset = _make_dataset(tmp_path)
    template_utils.write_metadata(_create_template_ds(0), dataset.store_factory)
    job = _make_region_job(_create_template_ds(4), region=slice(0, 4))

    with pytest.raises(AssertionError, match="processing_mode='update'"):
        dataset._run_virtual_operational_update([job], worker_index=0, workers_total=1)


def _construct_dataset(
    tmp_path: Path, dataset_cls: type[VirtualTestDataset]
) -> VirtualTestDataset:
    container = icechunk.VirtualChunkContainer(
        f"file://{tmp_path}/", icechunk.local_filesystem_store(str(tmp_path))
    )
    return dataset_cls(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
        icechunk_virtual_config=IcechunkVirtualConfig(
            containers=(container,),
            manifest_split=manifest_append_dim_split(split_size=2, dim="init_time"),
        ),
    )


def _encoding(**overrides: Any) -> Encoding:  # noqa: ANN401 - encoding field passthrough
    defaults: dict[str, Any] = {
        "dtype": "float64",
        "fill_value": np.nan,
        "chunks": (1, 1, N_LAT, N_LON),
        "shards": None,
        "compressors": (),
        "filters": None,
    }
    return Encoding(**{**defaults, **overrides})


def test_virtual_dataset_rejects_sharded_or_compressed_encodings(
    tmp_path: Path,
) -> None:
    class ShardedTemplateConfig(VirtualTestTemplateConfig):
        @computed_field  # type: ignore[prop-decorator]
        @property
        def data_vars(self) -> Sequence[VirtualTestDataVar]:
            return [
                VirtualTestDataVar(
                    name="temperature_2m",
                    encoding=_encoding(shards=(2, 2, N_LAT, N_LON)),
                )
            ]

    class ShardedDataset(VirtualTestDataset):
        template_config: ShardedTemplateConfig = ShardedTemplateConfig()

    with pytest.raises(ValidationError, match="must not declare shards"):
        _construct_dataset(tmp_path, ShardedDataset)

    class CompressedTemplateConfig(VirtualTestTemplateConfig):
        @computed_field  # type: ignore[prop-decorator]
        @property
        def data_vars(self) -> Sequence[VirtualTestDataVar]:
            return [
                VirtualTestDataVar(
                    name="temperature_2m", encoding=_encoding(compressors=None)
                )
            ]

    class CompressedDataset(VirtualTestDataset):
        template_config: CompressedTemplateConfig = CompressedTemplateConfig()

    with pytest.raises(ValidationError, match="must declare compressors="):
        _construct_dataset(tmp_path, CompressedDataset)


# --- process_virtual integration (real value read-back) ---


def test_backfill_emits_refs_and_reads_back_values(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    # Pre-size main with full metadata (as a backfill's parallel_setup would).
    template_utils.write_metadata(template_ds, dataset.store_factory)

    repo = _primary_repo(dataset.store_factory)
    job = _make_region_job(template_ds, region=slice(0, 4))
    _process_virtual(job, repo)

    _assert_all_values(dataset, n_inits=4)


def test_filter_skips_already_present_refs(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    template_utils.write_metadata(template_ds, dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)
    job = _make_region_job(template_ds, region=slice(0, 4))

    _process_virtual(job, repo)
    snapshots_after_first = _snapshot_count(repo)

    # Second run: every candidate is already present, so the filter drops them
    # all, the generator yields nothing, and no new commit is made.
    _process_virtual(job, repo)
    assert _snapshot_count(repo) == snapshots_after_first

    candidates = job.generate_source_file_coords(
        job._processing_region_ds(), job.data_vars
    )
    readonly = repo.readonly_session("main").store
    assert job.filter_already_present(candidates, readonly) == []


def test_filter_already_present_mixed_candidates(tmp_path: Path) -> None:
    # One filter call over a mix: already-emitted (dropped), not-yet-emitted (kept),
    # and an out-of-template label whose chunk_key is None (kept as remaining).
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    template_utils.write_metadata(template_ds, dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)
    # Emit only init 0's files.
    _process_virtual(_make_region_job(template_ds, region=slice(0, 1)), repo)

    present = VirtualTestSourceFileCoord(
        init_time=APPEND_DIM_START, lead_time=LEAD_TIMES[0]
    )
    absent = VirtualTestSourceFileCoord(
        init_time=APPEND_DIM_START + APPEND_DIM_FREQ, lead_time=LEAD_TIMES[0]
    )
    out_of_coords = VirtualTestSourceFileCoord(
        init_time=pd.Timestamp("2031-01-01"), lead_time=LEAD_TIMES[0]
    )

    job = _make_region_job(template_ds, region=slice(0, 4))
    readonly = repo.readonly_session("main").store
    remaining = job.filter_already_present([present, absent, out_of_coords], readonly)
    assert remaining == [absent, out_of_coords]


def test_sync_dims_to_grows_then_is_noop(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    # Initialize main empty (0 inits), as a virtual operational store starts.
    template_utils.write_metadata(_create_template_ds(0), dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)
    full_template = _create_template_ds(4)
    job = _make_region_job(full_template, region=slice(0, 4))

    session = repo.writable_session("main")
    assert job._append_dim_size(session.store) == 0
    job.sync_dims_to([session.store], 3)
    session.commit("grow to 3")

    grown = xr.open_zarr(dataset.store_factory.primary_store(), decode_timedelta=True)
    assert grown.sizes["init_time"] == 3
    # Derived coord (dims init_time x lead_time) expanded alongside the append dim.
    assert grown["valid_time"].sizes["init_time"] == 3
    np.testing.assert_array_equal(
        grown["valid_time"].values, full_template.to_dataset()["valid_time"].values[:3]
    )
    np.testing.assert_array_equal(
        grown["init_time"].values, full_template.to_dataset()["init_time"].values[:3]
    )

    # Already covered -> no-op (no write, session stays clean).
    session = repo.writable_session("main")
    job.sync_dims_to([session.store], 3)
    assert not session.has_uncommitted_changes


def test_sync_dims_to_resizes_each_store_from_its_own_size(tmp_path: Path) -> None:
    """Replicas commit before the primary, so a partial commit can leave a replica
    ahead. sync_dims_to must append each store's own missing slice; appending the
    primary's slice to an already-grown store would duplicate positions.

    The test harness maps every store of one dataset to a single path, so we stand
    in for the ahead replica with a second, independent repo.
    """
    dataset = _make_dataset(tmp_path)
    template_utils.write_metadata(_create_template_ds(0), dataset.store_factory)
    primary_repo = _primary_repo(dataset.store_factory)

    ahead_repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "ahead.icechunk"))
    )
    template_utils.write_metadata(
        _create_template_ds(0), ahead_repo.writable_session("main").store, mode="w-"
    )

    full_template = _create_template_ds(4)
    job = _make_region_job(full_template, region=slice(0, 4))

    # The ahead store committed a grow to 3 while the primary stayed at 0.
    ahead_session = ahead_repo.writable_session("main")
    job.sync_dims_to([ahead_session.store], 3)
    ahead_session.commit("ahead store grew before a partial commit")

    primary_session = primary_repo.writable_session("main")
    ahead_session = ahead_repo.writable_session("main")
    job.sync_dims_to([primary_session.store, ahead_session.store], 3)
    # The ahead store already covers 3 -> untouched; only the primary grows.
    assert not ahead_session.has_uncommitted_changes
    primary_session.commit("primary catches up")

    for repo in (primary_repo, ahead_repo):
        ds = xr.open_zarr(repo.readonly_session("main").store, decode_timedelta=True)
        assert ds.sizes["init_time"] == 3
        np.testing.assert_array_equal(
            ds["init_time"].values, full_template.to_dataset()["init_time"].values[:3]
        )


def test_yield_is_the_commit_unit(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    template_utils.write_metadata(template_ds, dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)

    # 4 inits x 2 leads = 8 candidate files, batched 2 at a time -> 4 yields.
    VirtualTestRegionJob.backfill_batch_files = 2
    job = _make_region_job(template_ds, region=slice(0, 4))
    before = _snapshot_count(repo)
    _process_virtual(job, repo)
    assert _snapshot_count(repo) - before == 4


def test_backfill_worker_commits_once_across_multiple_jobs(tmp_path: Path) -> None:
    # The core of the batching change: a worker handed several region jobs (one per
    # init, as get_jobs partitions a backfill) makes exactly ONE icechunk commit for
    # the whole worker, not one per job, and every init reads back.
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    template_utils.write_metadata(template_ds, dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)

    # A single sweep over the union of all jobs' coords (as the base generator does
    # for a backfill), so the whole worker is one commit.
    VirtualTestRegionJob.backfill_batch_files = 4 * N_LEADS
    worker_jobs = [
        _make_region_job(template_ds, region=slice(i, i + 1)) for i in range(4)
    ]

    before = _snapshot_count(repo)
    results = VirtualTestRegionJob.process_worker_jobs(
        worker_jobs, dataset.store_factory, "main", worker_index=0
    )
    assert results == {}
    assert _snapshot_count(repo) - before == 1
    _assert_all_values(dataset, n_inits=4)


def test_all_present_worker_makes_no_commit(tmp_path: Path) -> None:
    # A worker whose jobs are all already present yields no remaining coords; it must
    # skip the write loop (an empty icechunk commit would raise).
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    template_utils.write_metadata(template_ds, dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)

    VirtualTestRegionJob.backfill_batch_files = 4 * N_LEADS
    worker_jobs = [
        _make_region_job(template_ds, region=slice(i, i + 1)) for i in range(4)
    ]
    VirtualTestRegionJob.process_worker_jobs(
        worker_jobs, dataset.store_factory, "main", worker_index=0
    )
    after_first = _snapshot_count(repo)

    VirtualTestRegionJob.process_worker_jobs(
        worker_jobs, dataset.store_factory, "main", worker_index=0
    )
    assert _snapshot_count(repo) == after_first


def test_batched_driver_region_is_poisoned(tmp_path: Path) -> None:
    # The batched write loop spans every job's region, so a method that reads
    # self.region would silently narrow the batch to jobs[0]. The driver's region is
    # poisoned, so such a leak raises loudly instead of corrupting the write set.
    class RegionReadingJob(VirtualTestRegionJob):
        def process_virtual_refs(
            self,
            remaining: Sequence[VirtualTestSourceFileCoord],
        ) -> Iterator[
            Sequence[tuple[VirtualTestSourceFileCoord, Sequence[VirtualRef]]]
        ]:
            _ = self.region.start  # the leak the poison guards against
            yield from super().process_virtual_refs(remaining)

    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    template_utils.write_metadata(template_ds, dataset.store_factory)

    RegionReadingJob.backfill_batch_files = 4 * N_LEADS
    worker_jobs = [
        RegionReadingJob(
            tmp_store=Path("unused-tmp.zarr"),
            template_ds=template_ds,
            data_vars=[VirtualTestDataVar(name="temperature_2m")],
            append_dim="init_time",
            region=slice(i, i + 1),
            reformat_job_name="test",
            processing_mode="backfill",
        )
        for i in range(4)
    ]
    with pytest.raises(AssertionError, match=r"self\.region"):
        RegionReadingJob.process_worker_jobs(
            worker_jobs, dataset.store_factory, "main", worker_index=0
        )


# --- driver fork integration (operational + backfill routing) ---


def test_virtual_region_job_requires_virtual_config(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="icechunk_virtual_config"):
        VirtualTestDataset(
            primary_storage_config=StorageConfig(
                base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
            ),
        )


def test_two_worker_backfill_disjoint(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    template_utils.write_metadata(template_ds, dataset.store_factory)

    all_jobs = VirtualTestRegionJob.get_jobs(
        tmp_store=dataset._tmp_store(),
        template_ds=template_ds,
        append_dim="init_time",
        all_data_vars=dataset.template_config.data_vars,
        reformat_job_name="test",
    )
    # shards=None -> partition by chunks -> one region (one init) per job.
    assert len(all_jobs) == 4

    for worker_index in range(2):
        dataset._process_region_jobs(
            all_jobs=all_jobs,
            worker_index=worker_index,
            workers_total=2,
            reformat_job_name="test",
            template_ds=template_ds,
            # Distinct per worker, simulating separate pods
            tmp_store=tmp_path / f"worker-{worker_index}-tmp.zarr",
            update_template_with_results=False,
        )

    _assert_all_values(dataset, n_inits=4)
    # Temp branch cleaned up after finalize.
    assert list(_primary_repo(dataset.store_factory).list_branches()) == ["main"]


def test_virtual_operational_single_writer_expands_main(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    # Start with an empty main (no future NaNs), then operationally expand it.
    template_utils.write_metadata(_create_template_ds(0), dataset.store_factory)
    full_template = _create_template_ds(4)
    job = _make_region_job(full_template, region=slice(0, 4), processing_mode="update")

    dataset._run_virtual_operational_update([job], worker_index=0, workers_total=1)

    # Single writer committed straight to main (no temp branch ever created).
    assert list(_primary_repo(dataset.store_factory).list_branches()) == ["main"]
    _assert_all_values(dataset, n_inits=4)


def test_update_still_commits_per_tick(tmp_path: Path) -> None:
    # Batching commits per worker for backfill must not collapse the update cadence:
    # the operational update still commits once per yielded tick, so readers see each
    # tick's files within seconds rather than waiting for the whole window.
    dataset = _make_dataset(tmp_path)
    template_utils.write_metadata(_create_template_ds(0), dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)
    full_template = _create_template_ds(4)

    # 4 inits x 2 leads = 8 files, yielded 2 at a time -> 4 ticks -> 4 commits.
    VirtualTestRegionJob.backfill_batch_files = 2
    job = _make_region_job(full_template, region=slice(0, 4), processing_mode="update")

    before = _snapshot_count(repo)
    dataset._run_virtual_operational_update([job], worker_index=0, workers_total=1)
    assert _snapshot_count(repo) - before == 4
    _assert_all_values(dataset, n_inits=4)


def test_update_routes_virtual_to_single_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercise the production entry point: update() must detect virtual via
    # issubclass and route to the single-writer path (not _process_region_jobs).
    dataset = _make_dataset(tmp_path)
    template_utils.write_metadata(_create_template_ds(0), dataset.store_factory)
    full_template = _create_template_ds(4)
    job = _make_region_job(full_template, region=slice(0, 4), processing_mode="update")

    monkeypatch.setattr(
        VirtualTestRegionJob,
        "operational_update_jobs",
        classmethod(lambda cls, **kwargs: ([job], full_template)),
    )
    dataset.update("test")

    assert list(_primary_repo(dataset.store_factory).list_branches()) == ["main"]
    _assert_all_values(dataset, n_inits=4)


def test_virtual_operational_rejects_multiple_jobs(tmp_path: Path) -> None:
    # One active-window job whose generator polls all still-missing files; a
    # second job would run sequentially and could be starved by the first's poll.
    dataset = _make_dataset(tmp_path)
    template_utils.write_metadata(_create_template_ds(0), dataset.store_factory)
    full_template = _create_template_ds(4)
    jobs = [
        _make_region_job(full_template, region=slice(0, 2)),
        _make_region_job(full_template, region=slice(2, 4)),
    ]
    with pytest.raises(AssertionError, match="single active-window job"):
        dataset._run_virtual_operational_update(jobs, worker_index=0, workers_total=1)


def test_validate_dataset_on_virtual_skips_shard_check(tmp_path: Path) -> None:
    # Virtual stores have shards=None and intentionally-missing chunks for
    # partially-published inits, so check_for_expected_shards must be skipped.
    dataset = _make_dataset(tmp_path)
    template_utils.write_metadata(_create_template_ds(4), dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)
    # Emit only inits 0-1; inits 2-3 stay missing (a partially-published state).
    _process_virtual(_make_region_job(_create_template_ds(4), region=slice(0, 2)), repo)
    dataset.validate_dataset("test")  # must not raise


# --- virtual operational validators (completeness + decode health) ---


def _backfilled_store(
    dataset: VirtualTestDataset, template_ds: xr.DataTree, *, emit: slice
) -> icechunk.IcechunkStore:
    """Pre-size main to the template, emit refs for `emit`, return the readonly store."""
    template_utils.write_metadata(template_ds, dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)
    _process_virtual(_make_region_job(template_ds, region=emit), repo)
    return repo.readonly_session("main").store


def test_check_virtual_manifest_completeness_passes(tmp_path: Path) -> None:
    # Default (1.0,): every position in the window must be fully present.
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    store = _backfilled_store(dataset, template_ds, emit=slice(0, 4))
    job = _make_region_job(template_ds, region=slice(0, 4))

    result = validation.CheckVirtualManifestCompleteness()(
        job, store, xr.open_zarr(store, decode_timedelta=True)
    )
    assert result.passed, result.message


def test_check_virtual_manifest_completeness_detects_hole(tmp_path: Path) -> None:
    # Emit inits 0-1 of a 4-init window; with the default (1.0,) the missing inits 2-3
    # are real holes.
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    store = _backfilled_store(dataset, template_ds, emit=slice(0, 2))
    job = _make_region_job(template_ds, region=slice(0, 4))

    result = validation.CheckVirtualManifestCompleteness()(
        job, store, xr.open_zarr(store, decode_timedelta=True)
    )
    assert not result.passed
    assert "Incomplete" in result.message


def test_check_virtual_manifest_completeness_tolerates_partial_newest(
    tmp_path: Path,
) -> None:
    # Emit inits 0-2 but not the newest (init 3). (0.0, 1.0): the newest may be absent,
    # every older position must be complete -> passes.
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    store = _backfilled_store(dataset, template_ds, emit=slice(0, 3))
    job = _make_region_job(template_ds, region=slice(0, 4))
    ds = xr.open_zarr(store, decode_timedelta=True)

    tolerant = validation.CheckVirtualManifestCompleteness(
        min_present_fraction=(0.0, 1.0)
    )
    assert tolerant(job, store, ds).passed
    # But a fraction floor the absent newest can't meet still fails on it.
    strict = validation.CheckVirtualManifestCompleteness(
        min_present_fraction=(0.5, 1.0)
    )
    result = strict(job, store, ds)
    assert not result.passed
    assert "Incomplete" in result.message


def test_check_virtual_manifest_completeness_fails_when_window_too_short(
    tmp_path: Path,
) -> None:
    # A 1-position window can't satisfy a 2-tier threshold: fail loudly, never silently pass.
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    store = _backfilled_store(dataset, template_ds, emit=slice(0, 1))
    job = _make_region_job(template_ds, region=slice(0, 1))

    result = validation.CheckVirtualManifestCompleteness(
        min_present_fraction=(0.5, 1.0)
    )(job, store, xr.open_zarr(store, decode_timedelta=True))
    assert not result.passed
    assert "need at least 2" in result.message


def test_check_virtual_decode_health_passes(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    store = _backfilled_store(dataset, template_ds, emit=slice(0, 4))
    job = _make_region_job(template_ds, region=slice(0, 4))
    ds = xr.open_zarr(store, decode_timedelta=True)

    result = validation.CheckVirtualDecodeHealth()(job, store, ds)
    assert result.passed, result.message
    assert "all readable" in result.message
    # "latest": targets the newest present position, not an older one.
    assert str(ds.get_index("init_time")[-1]) in result.message


def test_check_virtual_decode_health_only_decodes_present_refs(tmp_path: Path) -> None:
    # Emit inits 0-1 of a 4-init window. The absent inits 2-3 must NOT be decoded (which
    # would read fill-value NaN and false-fail); decode-health checks the latest *present*
    # init (1), whose refs decode to real data -> passes.
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    store = _backfilled_store(dataset, template_ds, emit=slice(0, 2))
    job = _make_region_job(template_ds, region=slice(0, 4))
    ds = xr.open_zarr(store, decode_timedelta=True)

    result = validation.CheckVirtualDecodeHealth()(job, store, ds)
    assert result.passed, result.message
    assert str(ds.get_index("init_time")[1]) in result.message


def test_check_virtual_decode_health_fails_when_no_present_refs(tmp_path: Path) -> None:
    # Pre-sized but empty store: source files are expected but none are ingested, so there
    # is nothing to decode -> fail loudly rather than silently pass.
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    template_utils.write_metadata(template_ds, dataset.store_factory)
    store = _primary_repo(dataset.store_factory).readonly_session("main").store
    job = _make_region_job(template_ds, region=slice(0, 4))

    result = validation.CheckVirtualDecodeHealth()(
        job, store, xr.open_zarr(store, decode_timedelta=True)
    )
    assert not result.passed
    assert "No present references" in result.message


def test_check_virtual_decode_health_detects_unreadable_ref(tmp_path: Path) -> None:
    # A present ref whose bytes decode to all-NaN is unreadable data. Overwrite init 0's
    # message blocks with NaN, emit only init 0, and assert decode-health flags the var.
    dataset = _make_dataset(tmp_path)
    messages = tmp_path / "messages.bin"
    data = bytearray(messages.read_bytes())
    nan_block = np.full(N_LAT * N_LON, np.nan, dtype="<f8").tobytes()
    for lead_idx in range(N_LEADS):
        offset, length = _message_offset_length(0, lead_idx)
        data[offset : offset + length] = nan_block
    messages.write_bytes(bytes(data))

    template_ds = _create_template_ds(1)
    store = _backfilled_store(dataset, template_ds, emit=slice(0, 1))
    job = _make_region_job(template_ds, region=slice(0, 1))

    result = validation.CheckVirtualDecodeHealth()(
        job, store, xr.open_zarr(store, decode_timedelta=True)
    )
    assert not result.passed
    assert "entirely NaN" in result.message


def test_check_virtual_decode_health_skips_vars_without_reference(
    tmp_path: Path,
) -> None:
    # Same all-NaN unreadable ref as detects_unreadable_ref, but the offline opt-in
    # reference_exists oracle reports the var has no reference at the sampled position:
    # it is skipped (an availability matter) so decode-health PASSES and reports the
    # no-reference var. With the oracle reporting the ref present (like the operational
    # default reference_exists=None) the same all-NaN var still FAILS.
    dataset = _make_dataset(tmp_path)
    messages = tmp_path / "messages.bin"
    data = bytearray(messages.read_bytes())
    nan_block = np.full(N_LAT * N_LON, np.nan, dtype="<f8").tobytes()
    for lead_idx in range(N_LEADS):
        offset, length = _message_offset_length(0, lead_idx)
        data[offset : offset + length] = nan_block
    messages.write_bytes(bytes(data))

    template_ds = _create_template_ds(1)
    store = _backfilled_store(dataset, template_ds, emit=slice(0, 1))
    job = _make_region_job(template_ds, region=slice(0, 1))
    ds = xr.open_zarr(store, decode_timedelta=True)

    skipped = validation.CheckVirtualDecodeHealth(
        reference_exists=lambda var_path, out_loc: False
    )(job, store, ds)
    assert skipped.passed, skipped.message
    assert "no reference" in skipped.message

    present = validation.CheckVirtualDecodeHealth(
        reference_exists=lambda var_path, out_loc: True
    )(job, store, ds)
    assert not present.passed
    assert "entirely NaN" in present.message


def test_validate_dataset_requires_region_job_for_virtual_validator(
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(tmp_path)
    store = _backfilled_store(dataset, _create_template_ds(4), emit=slice(0, 4))
    with pytest.raises(AssertionError, match="needs a region_job"):
        validation.validate_dataset(
            store, [validation.CheckVirtualManifestCompleteness()]
        )


def test_validate_dataset_wires_virtual_validators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The dataset lists the virtual validators in validators(); validate_dataset must
    # build the operational-window region job and dispatch the context to them.
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    _backfilled_store(dataset, template_ds, emit=slice(0, 4))
    job = _make_region_job(template_ds, region=slice(0, 4), processing_mode="update")

    monkeypatch.setattr(
        VirtualTestRegionJob,
        "operational_update_jobs",
        classmethod(lambda cls, **kwargs: ([job], template_ds)),
    )
    monkeypatch.setattr(
        VirtualTestDataset,
        "validators",
        lambda self: (
            validation.CheckVirtualManifestCompleteness(),
            validation.CheckVirtualDecodeHealth(),
        ),
    )
    dataset.validate_dataset("test")  # must not raise


def test_process_virtual_writes_refs_to_replica(tmp_path: Path) -> None:
    # Replica emit path + replicas-then-primary commit: both stores get every ref.
    dataset = _make_dataset(tmp_path)
    template_utils.write_metadata(_create_template_ds(0), dataset.store_factory)
    primary_repo = _primary_repo(dataset.store_factory)
    replica_repo = _make_replica_repo(tmp_path, dataset)
    template_utils.write_metadata(
        _create_template_ds(0), replica_repo.writable_session("main").store, mode="w-"
    )

    job = _make_region_job(_create_template_ds(4), region=slice(0, 4))
    _process_virtual(job, primary_repo, [replica_repo])

    for repo in (primary_repo, replica_repo):
        _assert_store_values(repo.readonly_session("main").store, n_inits=4)


def test_process_virtual_recovers_when_replica_ahead(tmp_path: Path) -> None:
    # Replicas commit before the primary, so a partial commit can leave a replica
    # ahead. The next fire derives work from the (behind) primary and replays on
    # both stores; the replica replay is idempotent (no duplicate positions).
    dataset = _make_dataset(tmp_path)
    template_utils.write_metadata(_create_template_ds(0), dataset.store_factory)
    primary_repo = _primary_repo(dataset.store_factory)
    replica_repo = _make_replica_repo(tmp_path, dataset)
    template_utils.write_metadata(
        _create_template_ds(0), replica_repo.writable_session("main").store, mode="w-"
    )

    # Partial commit: the replica committed (grew to 4 + refs); the primary did not.
    _process_virtual(
        _make_region_job(_create_template_ds(4), region=slice(0, 4)), replica_repo
    )
    primary_now = xr.open_zarr(
        primary_repo.readonly_session("main").store, decode_timedelta=True
    )
    assert primary_now.sizes["init_time"] == 0
    _assert_store_values(replica_repo.readonly_session("main").store, n_inits=4)

    # Next fire catches the primary up and idempotently replays on the replica.
    _process_virtual(
        _make_region_job(_create_template_ds(4), region=slice(0, 4)),
        primary_repo,
        [replica_repo],
    )
    for repo in (primary_repo, replica_repo):
        _assert_store_values(repo.readonly_session("main").store, n_inits=4)


def test_refresh_metadata_applies_to_replica_repos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A deployed template fix must reach every repo, not just the primary; a
    # replica left stale would drift from what primary readers see.
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(2)
    template_utils.write_metadata(template_ds, dataset.store_factory)
    primary_repo = _primary_repo(dataset.store_factory)
    replica_repo = _make_replica_repo(tmp_path, dataset)
    template_utils.write_metadata(
        _create_template_ds(2), replica_repo.writable_session("main").store, mode="w-"
    )
    _process_virtual(
        _make_region_job(template_ds, region=slice(0, 2)), primary_repo, [replica_repo]
    )

    # The test harness collapses a dataset's replicas onto the primary path, so
    # route refresh_store_metadata's store enumeration to the independent replica.
    def fake_replica_stores(
        _self: StoreFactory, writable: bool = False, branch: str = "main"
    ) -> list[Any]:
        assert writable
        return [replica_repo.writable_session(branch).store]

    monkeypatch.setattr(StoreFactory, "replica_stores", fake_replica_stores)

    fixed = _create_template_ds(2)
    fixed["temperature_2m"].attrs["comment"] = "deployed metadata fix"
    job = _make_region_job(fixed, region=slice(0, 2), processing_mode="update")
    replica_snapshots_before = _snapshot_count(replica_repo)
    job.refresh_metadata(dataset.store_factory, tmp_path / "refresh_tmp.zarr")

    assert _snapshot_count(replica_repo) == replica_snapshots_before + 1
    for repo in (primary_repo, replica_repo):
        ds = xr.open_zarr(repo.readonly_session("main").store, decode_timedelta=True)
        assert ds["temperature_2m"].attrs["comment"] == "deployed metadata fix"


def test_refresh_metadata_empty_store_returns_without_commit(tmp_path: Path) -> None:
    # Initial sizing is the backfill's job: a refresh on a store whose append dim is
    # size 0 must write and commit nothing, even when the template carries a fix.
    dataset = _make_dataset(tmp_path)
    template_utils.write_metadata(_create_template_ds(0), dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)

    fixed = _create_template_ds(4)
    fixed["temperature_2m"].attrs["comment"] = "deployed metadata fix"
    job = _make_region_job(fixed, region=slice(0, 4), processing_mode="update")
    snapshots_before = _snapshot_count(repo)
    job.refresh_metadata(dataset.store_factory, tmp_path / "refresh_tmp.zarr")
    assert _snapshot_count(repo) == snapshots_before


def test_virtual_operational_second_fire_sees_no_new_work(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    template_utils.write_metadata(_create_template_ds(0), dataset.store_factory)
    full_template = _create_template_ds(4)

    def fire() -> None:
        job = _make_region_job(
            full_template, region=slice(0, 4), processing_mode="update"
        )
        dataset._run_virtual_operational_update([job], worker_index=0, workers_total=1)

    fire()
    repo = _primary_repo(dataset.store_factory)
    snapshots_after_first = _snapshot_count(repo)
    # The first fire's append must not delete spatial_ref (value == fill_value),
    # or the next fire's refresh would restore it and commit every fire.
    spatial_ref_chunk = _main_store_bytes(dataset.store_factory, "spatial_ref/c")
    assert spatial_ref_chunk is not None

    # No new data and no template drift -> no new commits: the metadata refresh's
    # unconsolidated render is byte-identical to what appends leave in the store.
    fire()
    assert _snapshot_count(repo) == snapshots_after_first
    assert (
        _main_store_bytes(dataset.store_factory, "spatial_ref/c") == spatial_ref_chunk
    )


def test_virtual_backfill_then_fire_leaves_metadata_stable(tmp_path: Path) -> None:
    # The prod sequence that exposed writer misalignment: a backfill through
    # parallel_setup/finalize, then an operational fire whose refresh should
    # find nothing to fix.
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    # Seed with consolidated metadata (as older code wrote) to prove the
    # backfill's own metadata writes remove it.
    template_utils.write_metadata(template_ds, dataset.store_factory)

    all_jobs = VirtualTestRegionJob.get_jobs(
        tmp_store=dataset._tmp_store(),
        template_ds=template_ds,
        append_dim="init_time",
        all_data_vars=dataset.template_config.data_vars,
        reformat_job_name="test",
    )
    dataset._process_region_jobs(
        all_jobs=all_jobs,
        worker_index=0,
        workers_total=1,
        reformat_job_name="test",
        template_ds=template_ds,
        tmp_store=tmp_path / "worker-tmp.zarr",
        update_template_with_results=False,
    )

    root_metadata = _main_store_bytes(dataset.store_factory, "zarr.json")
    assert root_metadata is not None
    assert json.loads(root_metadata).get("consolidated_metadata") is None
    assert _main_store_bytes(dataset.store_factory, "spatial_ref/c") is not None

    repo = _primary_repo(dataset.store_factory)
    snapshots_after_backfill = _snapshot_count(repo)
    job = _make_region_job(template_ds, region=slice(0, 4), processing_mode="update")
    dataset._run_virtual_operational_update([job], worker_index=0, workers_total=1)
    assert _snapshot_count(repo) == snapshots_after_backfill


def test_serializer_threads_through_expansion_without_decoding(tmp_path: Path) -> None:
    """A GribberishCodec serializer (decode-only) must survive dimension expansion
    and ref emit without ever being invoked. We never decode here (no real GRIB) -
    we assert the codec persists in metadata and the ref lands in the manifest."""
    dataset = _make_dataset(tmp_path)
    serializer = GribberishCodec(var="TMP").to_dict()
    template_utils.write_metadata(
        _create_template_ds(1, serializer=serializer), dataset.store_factory
    )
    repo = _primary_repo(dataset.store_factory)

    full_template = _create_template_ds(2, serializer=serializer)
    job = _make_region_job(full_template, region=slice(0, 2))

    session = repo.writable_session("main")
    job.sync_dims_to([session.store], 2)
    new_init = APPEND_DIM_START + APPEND_DIM_FREQ
    job._emit_refs(
        [session.store],
        [
            VirtualRef(
                job.data_vars[0],
                {"init_time": new_init, "lead_time": LEAD_TIMES[0]},
                VirtualTestRegionJob.messages_url,
                *_message_offset_length(1, 0),
            )
        ],
    )
    session.commit("expand + ref")

    readonly = repo.readonly_session("main").store
    array = zarr.open_group(readonly, mode="r")["temperature_2m"]
    assert isinstance(array, zarr.Array)
    assert isinstance(array.metadata, ArrayV3Metadata)
    codec_dicts = [
        codec.to_dict() if hasattr(codec, "to_dict") else codec
        for codec in array.metadata.codecs
    ]
    assert {"name": "gribberish", "configuration": {"var": "TMP"}} in codec_dicts
    assert asyncio.run(readonly.exists("temperature_2m/c/1/0/0/0")) is True


def test_exists_many_retries_only_failed_keys() -> None:
    """A transient error on one key re-probes only that key, not the whole batch."""
    calls: dict[str, int] = {}

    class FlakyStore:
        async def exists(self, key: str) -> bool:
            calls[key] = calls.get(key, 0) + 1
            if key == "b" and calls[key] == 1:
                raise RuntimeError("transient")
            return key != "c"  # "c" is genuinely absent

    result = _exists_many(cast("icechunk.IcechunkStore", FlakyStore()), ["a", "b", "c"])
    assert result == {"a": True, "b": True, "c": False}
    # "a"/"c" succeeded on the first attempt and are not re-probed; only "b" retries.
    assert calls == {"a": 1, "b": 2, "c": 1}


def test_exists_many_raises_after_exhausting_attempts() -> None:
    class DeadStore:
        async def exists(self, _key: str) -> bool:
            raise RuntimeError("object store down")

    with pytest.raises(RuntimeError, match="object store down"):
        _exists_many(cast("icechunk.IcechunkStore", DeadStore()), ["a"], max_attempts=2)
