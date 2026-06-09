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
from collections.abc import Iterator, Mapping, Sequence
from datetime import timedelta
from itertools import batched
from pathlib import Path
from typing import Any, ClassVar

import dask.array
import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from gribberish.zarr import GribberishCodec
from pydantic import ValidationError, computed_field
from zarr.core.metadata import ArrayV3Metadata

from reformatters.common import template_utils, validation
from reformatters.common.config_models import (
    BaseInternalAttrs,
    DataVar,
    DataVarAttrs,
    Encoding,
)
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.region_job import (
    CoordinateValue,
    SourceFileCoord,
    VirtualRef,
    VirtualRegionJob,
)
from reformatters.common.storage import (
    DatasetFormat,
    IcechunkVirtualConfig,
    StorageConfig,
    StoreFactory,
    manifest_append_dim_split,
)
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp

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
) -> xr.Dataset:
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
    ds["init_time"].encoding.update(time_encoding)
    ds["valid_time"].encoding.update(time_encoding)
    ds["lead_time"].encoding.update(
        {"dtype": "int64", "fill_value": -1, "units": "seconds"}
    )
    ds["latitude"].encoding["fill_value"] = np.nan
    ds["longitude"].encoding["fill_value"] = np.nan
    return ds


# --- synthetic virtual forecast dataset ---


class VirtualTestDataVar(DataVar[BaseInternalAttrs]):
    encoding: Encoding = Encoding(
        dtype="float64",
        fill_value=np.nan,
        chunks=(1, 1, N_LAT, N_LON),  # one chunk per (init, lead) message
        shards=None,
        compressors=None,
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
    ) -> Iterator[Sequence[VirtualRef]]:
        data_var = self.data_vars[0]
        init_index = self.template_ds.get_index("init_time")
        lead_index = self.template_ds.get_index("lead_time")
        for group in batched(remaining, self.backfill_batch_files, strict=False):
            refs: list[VirtualRef] = []
            for coord in group:
                init_idx = int(init_index.get_indexer(pd.Index([coord.init_time]))[0])
                lead_idx = int(lead_index.get_indexer(pd.Index([coord.lead_time]))[0])
                offset, length = _message_offset_length(init_idx, lead_idx)
                refs.append(
                    VirtualRef(
                        data_var=data_var,
                        out_loc=coord.out_loc(),
                        location=self.messages_url,
                        offset=offset,
                        length=length,
                    )
                )
            yield refs


class VirtualTestTemplateConfig(TemplateConfig[VirtualTestDataVar]):
    dims: tuple[Dim, ...] = ("init_time", "lead_time", "latitude", "longitude")
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
    template_ds: xr.Dataset,
    *,
    region: slice,
) -> VirtualTestRegionJob:
    return VirtualTestRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=[VirtualTestDataVar(name="temperature_2m")],
        append_dim="init_time",
        region=region,
        reformat_job_name="test",
    )


def _primary_repo(factory: StoreFactory) -> icechunk.Repository:
    return factory.all_icechunk_repos(sort="primary-first")[0][1]


def _snapshot_count(repo: icechunk.Repository, branch: str = "main") -> int:
    return sum(1 for _ in repo.ancestry(branch=branch))


def _assert_all_values(dataset: VirtualTestDataset, n_inits: int) -> None:
    result = xr.open_zarr(dataset.store_factory.primary_store(), decode_timedelta=True)
    assert result.sizes["init_time"] == n_inits
    for init_idx in range(n_inits):
        for lead_idx in range(N_LEADS):
            np.testing.assert_array_equal(
                result["temperature_2m"]
                .isel(init_time=init_idx, lead_time=lead_idx)
                .values,
                _block_values(init_idx, lead_idx),
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
        ) -> Iterator[Sequence[VirtualRef]]:
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
        job.process_virtual(repo, [], "main")


# --- process_virtual integration (real value read-back) ---


def test_backfill_emits_refs_and_reads_back_values(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    # Pre-size main with full metadata (as a backfill's parallel_setup would).
    template_utils.write_metadata(template_ds, dataset.store_factory)

    repo = _primary_repo(dataset.store_factory)
    job = _make_region_job(template_ds, region=slice(0, 4))
    job.process_virtual(repo, [], "main")

    _assert_all_values(dataset, n_inits=4)


def test_filter_skips_already_present_refs(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    template_ds = _create_template_ds(4)
    template_utils.write_metadata(template_ds, dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)
    job = _make_region_job(template_ds, region=slice(0, 4))

    job.process_virtual(repo, [], "main")
    snapshots_after_first = _snapshot_count(repo)

    # Second run: every candidate is already present, so the filter drops them
    # all, the generator yields nothing, and no new commit is made.
    job.process_virtual(repo, [], "main")
    assert _snapshot_count(repo) == snapshots_after_first

    candidates = job.generate_source_file_coords(
        job._processing_region_ds(), job.data_vars
    )
    readonly = repo.readonly_session("main").store
    assert job.filter_already_present(candidates, readonly) == []


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
        grown["valid_time"].values, full_template["valid_time"].values[:3]
    )
    np.testing.assert_array_equal(
        grown["init_time"].values, full_template["init_time"].values[:3]
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
            ds["init_time"].values, full_template["init_time"].values[:3]
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
    job.process_virtual(repo, [], "main")
    assert _snapshot_count(repo) - before == 4


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
            tmp_store=dataset._tmp_store(),
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
    job = _make_region_job(full_template, region=slice(0, 4))

    dataset._run_virtual_operational_update([job], workers_total=1)

    # Single writer committed straight to main (no temp branch ever created).
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
        dataset._run_virtual_operational_update(jobs, workers_total=1)


def test_validate_dataset_on_virtual_skips_shard_check(tmp_path: Path) -> None:
    # Virtual stores have shards=None and intentionally-missing chunks for
    # partially-published inits, so check_for_expected_shards must be skipped.
    dataset = _make_dataset(tmp_path)
    template_utils.write_metadata(_create_template_ds(4), dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)
    # Emit only inits 0-1; inits 2-3 stay missing (a partially-published state).
    _make_region_job(_create_template_ds(4), region=slice(0, 2)).process_virtual(
        repo, [], "main"
    )
    dataset.validate_dataset("test")  # must not raise


def test_virtual_operational_second_fire_sees_no_new_work(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    template_utils.write_metadata(_create_template_ds(0), dataset.store_factory)
    full_template = _create_template_ds(4)

    def fire() -> None:
        job = _make_region_job(full_template, region=slice(0, 4))
        dataset._run_virtual_operational_update([job], workers_total=1)

    fire()
    repo = _primary_repo(dataset.store_factory)
    snapshots_after_first = _snapshot_count(repo)

    fire()  # everything already present -> no new commits
    assert _snapshot_count(repo) == snapshots_after_first


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
