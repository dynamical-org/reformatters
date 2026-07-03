"""End-to-end test for a VIRTUAL icechunk dataset with a vertical-dimension group.

This mirrors the single-group harness in ``virtual_region_job_test.py`` but adds a
``pressure_level`` group so the multi-group virtual write loop is exercised: refs
landing at a group-qualified array path (``pressure_level/temperature``), per-file
commit atomicity *across* groups (one file's root + pressure refs commit together),
per-group append-dim growth (each node grown on its own via ``sync_dims_to``), and a
chunk_key that resolves a vertical-dimension position.

As in the single-group file the "GRIB messages" are raw little-endian float64 blocks
in a local file and refs point at byte ranges via a local-filesystem virtual chunk
container (no real GRIB / no GribberishCodec), so a value round-trip catches bad chunk
keys *and* bad byte ranges.
"""

from collections.abc import Iterator, Mapping, Sequence
from datetime import timedelta
from itertools import batched
from pathlib import Path
from typing import Any, ClassVar, Literal

import dask.array
import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr

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
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.region_job import CoordinateValue, SourceFileCoord
from reformatters.common.storage import (
    DatasetFormat,
    IcechunkVirtualConfig,
    StorageConfig,
    StoreFactory,
    manifest_append_dim_split,
)
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob

pytestmark = pytest.mark.slow

N_LAT = 2
N_LON = 3
LEAD_TIMES = pd.timedelta_range("0h", periods=1, freq="6h")
N_LEADS = len(LEAD_TIMES)
PRESSURE_LEVELS = [1000, 850]
N_LEVELS = len(PRESSURE_LEVELS)
BLOCK_NBYTES = N_LAT * N_LON * 8  # one float64 message
APPEND_DIM_START = pd.Timestamp("2024-01-01")
APPEND_DIM_FREQ = pd.Timedelta("6h")
DATASET_ID = "test-virtual-multi-group-dataset"

# Per-file message layout: a root temperature_2m block followed by one block per
# pressure level. Byte offsets are deterministic and distinct so a value round-trip
# proves each ref landed at the right (group/name, chunk key, byte range).
BLOCKS_PER_FILE = 1 + N_LEVELS
FILE_NBYTES = BLOCKS_PER_FILE * BLOCK_NBYTES


def _root_values(init_idx: int, lead_idx: int) -> np.ndarray:
    """Deterministic root (temperature_2m) message values; distinct per (init, lead)."""
    base = 1000.0 * init_idx + 10.0 * lead_idx
    return (base + np.arange(N_LAT * N_LON)).reshape(N_LAT, N_LON).astype("<f8")


def _pressure_values(init_idx: int, lead_idx: int, level_idx: int) -> np.ndarray:
    """Deterministic pressure-level message values; distinct per (init, lead, level)."""
    base = 1000.0 * init_idx + 10.0 * lead_idx + 100000.0 * (level_idx + 1)
    return (base + np.arange(N_LAT * N_LON)).reshape(N_LAT, N_LON).astype("<f8")


def _file_offset(init_idx: int, lead_idx: int) -> int:
    return (init_idx * N_LEADS + lead_idx) * FILE_NBYTES


def _root_offset_length(init_idx: int, lead_idx: int) -> tuple[int, int]:
    return _file_offset(init_idx, lead_idx), BLOCK_NBYTES


def _pressure_offset_length(
    init_idx: int, lead_idx: int, level_idx: int
) -> tuple[int, int]:
    # Root block first, then one block per level.
    block = 1 + level_idx
    return _file_offset(init_idx, lead_idx) + block * BLOCK_NBYTES, BLOCK_NBYTES


def _write_messages_file(path: Path, n_inits: int) -> None:
    chunks: list[bytes] = []
    for init_idx in range(n_inits):
        for lead_idx in range(N_LEADS):
            chunks.append(_root_values(init_idx, lead_idx).tobytes())
            chunks.extend(
                _pressure_values(init_idx, lead_idx, level_idx).tobytes()
                for level_idx in range(N_LEVELS)
            )
    path.write_bytes(b"".join(chunks))


def _create_template_ds(n_inits: int) -> xr.DataTree:
    """Forecast-shaped virtual template with a root var plus a pressure_level group.

    One chunk per message: chunk size 1 along init_time, lead_time, and pressure_level
    so each (init, lead[, level]) is its own chunk. No shards; no compressors.
    """
    init_times = pd.date_range(APPEND_DIM_START, periods=n_inits, freq=APPEND_DIM_FREQ)
    shared_coords = {
        "init_time": ("init_time", init_times),
        "lead_time": ("lead_time", LEAD_TIMES),
        "latitude": ("latitude", np.arange(N_LAT, dtype="float64")),
        "longitude": ("longitude", np.arange(N_LON, dtype="float64")),
        "valid_time": (
            ("init_time", "lead_time"),
            init_times.values[:, None] + LEAD_TIMES.values[None, :],
        ),
    }
    root = xr.Dataset(
        {
            "temperature_2m": xr.Variable(
                ("init_time", "lead_time", "latitude", "longitude"),
                dask.array.full(
                    (n_inits, N_LEADS, N_LAT, N_LON),
                    np.nan,
                    dtype="float64",
                    chunks=-1,
                ),
                encoding={
                    "dtype": "float64",
                    "chunks": (1, 1, N_LAT, N_LON),
                    "fill_value": np.nan,
                    "compressors": None,
                    "filters": None,
                },
            )
        },
        coords=shared_coords,
        attrs={"dataset_id": DATASET_ID, "dataset_version": "v1.0"},
    )
    pressure = xr.Dataset(
        {
            "temperature": xr.Variable(
                ("init_time", "lead_time", "latitude", "longitude", "pressure_level"),
                dask.array.full(
                    (n_inits, N_LEADS, N_LAT, N_LON, N_LEVELS),
                    np.nan,
                    dtype="float64",
                    chunks=-1,
                ),
                encoding={
                    "dtype": "float64",
                    "chunks": (1, 1, N_LAT, N_LON, 1),  # one chunk per level
                    "fill_value": np.nan,
                    "compressors": None,
                    "filters": None,
                },
            )
        },
        # Shared coords are duplicated into the group so it can be opened on its own.
        coords={**shared_coords, "pressure_level": ("pressure_level", PRESSURE_LEVELS)},
    )

    # Explicit time encodings so dimension-appends re-encode consistently.
    time_encoding = {
        "dtype": "int64",
        "fill_value": -1,
        "units": "seconds since 1970-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
    }
    lead_encoding = {"dtype": "int64", "fill_value": -1, "units": "seconds"}
    for ds in (root, pressure):
        ds["init_time"].encoding.update(time_encoding)
        ds["valid_time"].encoding.update(time_encoding)
        ds["lead_time"].encoding.update(lead_encoding)
        ds["latitude"].encoding["fill_value"] = np.nan
        ds["longitude"].encoding["fill_value"] = np.nan
    pressure["pressure_level"].encoding["fill_value"] = -1
    return xr.DataTree.from_dict({"/": root, "/pressure_level": pressure})


# --- synthetic virtual multi-group forecast dataset ---


class RootDataVar(DataVar[BaseInternalAttrs]):
    encoding: Encoding = Encoding(
        dtype="float64",
        fill_value=np.nan,
        chunks=(1, 1, N_LAT, N_LON),
        shards=None,
        compressors=(),
        filters=None,
    )
    attrs: DataVarAttrs = DataVarAttrs(
        units="K",
        long_name="2 metre temperature",
        short_name="2t",
        step_type="instant",
    )
    internal_attrs: BaseInternalAttrs = BaseInternalAttrs(
        keep_mantissa_bits="no-rounding"
    )


class PressureDataVar(DataVar[BaseInternalAttrs]):
    group: Group = "pressure_level"
    encoding: Encoding = Encoding(
        dtype="float64",
        fill_value=np.nan,
        chunks=(1, 1, N_LAT, N_LON, 1),
        shards=None,
        compressors=(),
        filters=None,
    )
    attrs: DataVarAttrs = DataVarAttrs(
        units="K",
        long_name="Temperature",
        short_name="t",
        step_type="instant",
    )
    internal_attrs: BaseInternalAttrs = BaseInternalAttrs(
        keep_mantissa_bits="no-rounding"
    )


def _data_vars() -> list[DataVar[BaseInternalAttrs]]:
    return [
        RootDataVar(name="temperature_2m"),
        PressureDataVar(name="temperature", group="pressure_level"),
    ]


class MultiGroupSourceFileCoord(SourceFileCoord):
    init_time: Timestamp
    lead_time: Timedelta

    def get_url(self) -> str:
        return MultiGroupRegionJob.messages_url


class MultiGroupRegionJob(
    VirtualRegionJob[DataVar[BaseInternalAttrs], MultiGroupSourceFileCoord]
):
    # Set per test by _make_dataset; every ref points at this file.
    messages_url: ClassVar[str] = ""
    # Whole files per yielded batch (one commit per batch).
    backfill_batch_files: ClassVar[int] = 1

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[DataVar[BaseInternalAttrs]],  # noqa: ARG002
    ) -> Sequence[MultiGroupSourceFileCoord]:
        return [
            MultiGroupSourceFileCoord(init_time=pd.Timestamp(init_time), lead_time=lead)
            for init_time in processing_region_ds["init_time"].values
            for lead in LEAD_TIMES
        ]

    def process_virtual_refs(
        self,
        remaining: Sequence[MultiGroupSourceFileCoord],
    ) -> Iterator[Sequence[tuple[MultiGroupSourceFileCoord, Sequence[VirtualRef]]]]:
        # One source file contributes refs for BOTH the root chunk and each
        # pressure-level chunk, all in one (coord, refs) entry so they commit
        # together (per-file commit atomicity across groups).
        root_var, pressure_var = _data_vars()
        init_index = self.template_ds.to_dataset().get_index("init_time")
        lead_index = self.template_ds.to_dataset().get_index("lead_time")
        for group in batched(remaining, self.backfill_batch_files, strict=False):
            batch: list[tuple[MultiGroupSourceFileCoord, Sequence[VirtualRef]]] = []
            for coord in group:
                init_idx = int(init_index.get_indexer(pd.Index([coord.init_time]))[0])
                lead_idx = int(lead_index.get_indexer(pd.Index([coord.lead_time]))[0])

                root_offset, root_length = _root_offset_length(init_idx, lead_idx)
                refs: list[VirtualRef] = [
                    VirtualRef(
                        data_var=root_var,
                        out_loc=coord.out_loc(),  # {init_time, lead_time}
                        location=self.messages_url,
                        offset=root_offset,
                        length=root_length,
                    )
                ]
                for level_idx, level in enumerate(PRESSURE_LEVELS):
                    offset, length = _pressure_offset_length(
                        init_idx, lead_idx, level_idx
                    )
                    refs.append(
                        VirtualRef(
                            data_var=pressure_var,
                            # var.path routes set_virtual_refs to the group array;
                            # the level position comes from this out_loc.
                            out_loc={**coord.out_loc(), "pressure_level": level},
                            location=self.messages_url,
                            offset=offset,
                            length=length,
                        )
                    )
                batch.append((coord, refs))
            yield batch


class MultiGroupTemplateConfig(TemplateConfig[DataVar[BaseInternalAttrs]]):
    dims: dict[Group, tuple[Dim, ...]] = {
        ROOT: ("init_time", "lead_time", "latitude", "longitude"),
        "pressure_level": (
            "init_time",
            "lead_time",
            "latitude",
            "longitude",
            "pressure_level",
        ),
    }
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = APPEND_DIM_START
    append_dim_frequency: Timedelta = APPEND_DIM_FREQ

    @property
    def dataset_id(self) -> str:
        return DATASET_ID

    @property
    def version(self) -> str:
        return "v1.0"

    @property
    def data_vars(self) -> Sequence[DataVar[BaseInternalAttrs]]:
        return _data_vars()


class MultiGroupDataset(
    DynamicalDataset[DataVar[BaseInternalAttrs], MultiGroupSourceFileCoord]
):
    template_config: MultiGroupTemplateConfig = MultiGroupTemplateConfig()
    region_job_class: type[MultiGroupRegionJob] = MultiGroupRegionJob

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


def _make_dataset(tmp_path: Path, *, n_inits: int = 2) -> MultiGroupDataset:
    """Build the dataset, write the messages file, and point the region job at it.

    The virtual chunk container is a local-filesystem store rooted at tmp_path;
    refs point at tmp_path/messages.bin.
    """
    messages_path = tmp_path / "messages.bin"
    _write_messages_file(messages_path, n_inits=max(n_inits, 2))
    MultiGroupRegionJob.messages_url = f"file://{messages_path}"
    MultiGroupRegionJob.backfill_batch_files = 1

    container = icechunk.VirtualChunkContainer(
        f"file://{tmp_path}/", icechunk.local_filesystem_store(str(tmp_path))
    )
    return MultiGroupDataset(
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
) -> MultiGroupRegionJob:
    return MultiGroupRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=_data_vars(),
        append_dim="init_time",
        region=region,
        reformat_job_name="test",
        processing_mode=processing_mode,
    )


def _primary_repo(factory: StoreFactory) -> icechunk.Repository:
    return factory.icechunk_repos(sort="primary-first")[0][1]


def _process_virtual(
    job: MultiGroupRegionJob,
    primary_repo: icechunk.Repository,
    replica_repos: Sequence[icechunk.Repository] = (),
    branch: str = "main",
) -> None:
    """Gather remaining coords (as process_worker_jobs does), then drive the write loop."""
    readonly = primary_repo.readonly_session(branch).store
    remaining = job.filter_already_present(job.source_file_coords(), readonly)
    job.process_virtual(primary_repo, list(replica_repos), branch, remaining)


def _assert_all_values(dataset: MultiGroupDataset, n_inits: int) -> None:
    """Read back via DataTree and assert both groups carry their deterministic values
    at the correct (init, lead[, level]) positions."""
    store: Any = dataset.store_factory.primary_store()
    tree = xr.open_datatree(store, engine="zarr", decode_timedelta=True)

    root = tree.to_dataset()
    assert root.sizes["init_time"] == n_inits
    pressure_ds = tree["pressure_level"].to_dataset()
    assert pressure_ds.sizes["init_time"] == n_inits

    for init_idx in range(n_inits):
        for lead_idx in range(N_LEADS):
            np.testing.assert_array_equal(
                root["temperature_2m"]
                .isel(init_time=init_idx, lead_time=lead_idx)
                .values,
                _root_values(init_idx, lead_idx),
            )
            for level_idx in range(N_LEVELS):
                np.testing.assert_array_equal(
                    pressure_ds["temperature"]
                    .isel(
                        init_time=init_idx,
                        lead_time=lead_idx,
                        pressure_level=level_idx,
                    )
                    .values,
                    _pressure_values(init_idx, lead_idx, level_idx),
                )


# --- chunk_key resolves a vertical-dimension position ---


def test_chunk_key_resolves_pressure_level_position() -> None:
    job = _make_region_job(_create_template_ds(2), region=slice(0, 2))
    pressure_var = next(v for v in job.data_vars if v.group == "pressure_level")
    # init index 1, lead index 0, lat/lon single full-width chunks, level index 1.
    out_loc: Mapping[Dim, CoordinateValue] = {
        "init_time": APPEND_DIM_START + APPEND_DIM_FREQ,
        "lead_time": LEAD_TIMES[0],
        "pressure_level": PRESSURE_LEVELS[1],
    }
    assert job.chunk_key(out_loc, pressure_var) == (1, 0, 0, 0, 1)


# --- backfill: value round-trip + per-group growth + per-file atomicity ---


def test_backfill_emits_refs_to_both_groups_and_reads_back(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path, n_inits=2)
    template_ds = _create_template_ds(2)
    # Pre-size main with full metadata (as a backfill's parallel_setup would).
    template_utils.write_metadata(template_ds, dataset.store_factory)

    repo = _primary_repo(dataset.store_factory)
    job = _make_region_job(template_ds, region=slice(0, 2))
    _process_virtual(job, repo)

    # (1) Refs landed at the right group/name chunk keys and byte ranges; (2) both
    # the root and pressure_level groups grew to the expected init_time length.
    _assert_all_values(dataset, n_inits=2)


def test_open_flattened_dataset_includes_group_vars(tmp_path: Path) -> None:
    # The validation gap: xr.open_zarr reads only the root group, hiding group vars.
    # open_flattened_dataset must surface pressure_level/temperature (path-keyed)
    # alongside the root temperature_2m, with the group var carrying its vertical dim.
    dataset = _make_dataset(tmp_path, n_inits=2)
    template_ds = _create_template_ds(2)
    template_utils.write_metadata(template_ds, dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)
    _process_virtual(_make_region_job(template_ds, region=slice(0, 2)), repo)

    store = repo.readonly_session("main").store
    root_only = xr.open_zarr(store, consolidated=False, decode_timedelta=True)
    assert "pressure_level/temperature" not in root_only.data_vars

    flat = validation.open_flattened_dataset(store, consolidated=False)
    assert "temperature_2m" in flat.data_vars
    assert "pressure_level/temperature" in flat.data_vars
    assert "pressure_level" in flat["pressure_level/temperature"].dims


def test_nan_check_covers_group_vars(tmp_path: Path) -> None:
    # The most safety-critical case: a NaN/coverage validator must evaluate group vars,
    # not silently drop to root-only. Running it over the flat dataset with the group var
    # explicitly included reports "1 variable" (it was found and checked), not "0".
    dataset = _make_dataset(tmp_path, n_inits=2)
    template_ds = _create_template_ds(2)
    template_utils.write_metadata(template_ds, dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)
    _process_virtual(_make_region_job(template_ds, region=slice(0, 2)), repo)
    store = repo.readonly_session("main").store

    flat = validation.open_flattened_dataset(store, consolidated=False)
    result = validation.check_forecast_recent_nans(
        flat,
        include_vars=["pressure_level/temperature"],
        spatial_sampling="all",
    )
    assert result.passed, result.message
    assert "All 1 variables" in result.message


def test_decode_health_covers_group_vars(tmp_path: Path) -> None:
    # Decode health over a flattened multi-group store: the group var carries an extra
    # vertical dim (pressure_level) absent from the per-position selection, so per-var
    # dim handling must decode it without error rather than crashing on a missing/extra dim.
    dataset = _make_dataset(tmp_path, n_inits=2)
    template_ds = _create_template_ds(2)
    template_utils.write_metadata(template_ds, dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)
    _process_virtual(_make_region_job(template_ds, region=slice(0, 2)), repo)
    store = repo.readonly_session("main").store
    job = _make_region_job(template_ds, region=slice(0, 2))

    ds = validation.open_flattened_dataset(store, consolidated=False)
    assert "pressure_level/temperature" in ds.data_vars
    result = validation.CheckVirtualDecodeHealth()(job, store, ds)
    assert result.passed, result.message


def test_decode_health_samples_levels_and_positions(tmp_path: Path) -> None:
    # The offline decode scan tunes these knobs: sampling a subset of the group var's
    # levels and capping how many positions get decoded keep a whole-archive sweep bounded.
    dataset = _make_dataset(tmp_path, n_inits=2)
    template_ds = _create_template_ds(2)
    template_utils.write_metadata(template_ds, dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)
    _make_region_job(template_ds, region=slice(0, 2)).process_virtual(repo, [], "main")
    store = repo.readonly_session("main").store
    job = _make_region_job(template_ds, region=slice(0, 2))
    ds = validation.open_flattened_dataset(store, consolidated=False)

    # Sampling one of the two pressure levels still exercises the group var and passes.
    sampled = validation.CheckVirtualDecodeHealth(sampled_levels=1)(job, store, ds)
    assert sampled.passed, sampled.message

    # positions="all" capped to a single position decode-checks just one init.
    capped = validation.CheckVirtualDecodeHealth(positions="all", max_positions=1)(
        job, store, ds
    )
    assert capped.passed, capped.message
    assert capped.message.count("init_time=") == 1


def test_per_file_commit_contains_both_groups(tmp_path: Path) -> None:
    # Per-file atomicity across groups: each source file is one commit, and that
    # commit writes both a root chunk and a pressure_level chunk for the same file.
    dataset = _make_dataset(tmp_path, n_inits=2)
    template_ds = _create_template_ds(2)
    template_utils.write_metadata(template_ds, dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)

    job = _make_region_job(template_ds, region=slice(0, 1))
    _process_virtual(job, repo)

    # One file (init 0, lead 0) -> exactly one new commit holding both groups' chunks.
    snapshots = list(repo.ancestry(branch="main"))
    latest = snapshots[0]
    changed = repo.diff(from_snapshot_id=snapshots[1].id, to_snapshot_id=latest.id)
    written = {path.removeprefix("/") for path in changed.updated_chunks}
    assert "temperature_2m" in written
    assert "pressure_level/temperature" in written


def test_two_worker_backfill_disjoint(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path, n_inits=2)
    template_ds = _create_template_ds(2)
    template_utils.write_metadata(template_ds, dataset.store_factory)

    all_jobs = MultiGroupRegionJob.get_jobs(
        tmp_store=dataset._tmp_store(),
        template_ds=template_ds,
        append_dim="init_time",
        all_data_vars=dataset.template_config.data_vars,
        reformat_job_name="test",
    )
    # shards=None -> partition by chunks -> one region (one init) per job.
    assert len(all_jobs) == 2

    for worker_index in range(2):
        dataset._process_region_jobs(
            all_jobs=all_jobs,
            worker_index=worker_index,
            workers_total=2,
            reformat_job_name="test",
            template_ds=template_ds,
            tmp_store=tmp_path / f"worker-{worker_index}-tmp.zarr",
            update_template_with_results=False,
        )

    _assert_all_values(dataset, n_inits=2)
    assert list(_primary_repo(dataset.store_factory).list_branches()) == ["main"]


class _PressureProbeCoord(MultiGroupSourceFileCoord):
    pressure_level: float

    def get_url(self) -> str:
        return MultiGroupRegionJob.messages_url


class _PressureProbeRegionJob(MultiGroupRegionJob):
    def representative_var(
        self,
        coord: MultiGroupSourceFileCoord,  # noqa: ARG002
    ) -> DataVar[BaseInternalAttrs]:
        # Probe the pressure_level group array, not the root var.
        return next(v for v in self.data_vars if v.group == "pressure_level")


def test_filter_already_present_probes_group_array(tmp_path: Path) -> None:
    # A backfill over init 0 only writes pressure_level/temperature chunks for init 0.
    dataset = _make_dataset(tmp_path, n_inits=2)
    template_ds = _create_template_ds(2)
    template_utils.write_metadata(template_ds, dataset.store_factory)
    repo = _primary_repo(dataset.store_factory)
    job = _make_region_job(template_ds, region=slice(0, 1))
    _process_virtual(job, repo)

    probe_job = _PressureProbeRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=_data_vars(),
        append_dim="init_time",
        region=slice(0, 2),
        reformat_job_name="test",
        processing_mode="backfill",
    )
    present = _PressureProbeCoord(
        init_time=APPEND_DIM_START,
        lead_time=LEAD_TIMES[0],
        pressure_level=float(PRESSURE_LEVELS[0]),  # written by the init-0 backfill
    )
    absent = _PressureProbeCoord(
        init_time=APPEND_DIM_START + APPEND_DIM_FREQ,  # init 1, not backfilled
        lead_time=LEAD_TIMES[0],
        pressure_level=float(PRESSURE_LEVELS[0]),
    )
    out_of_coords = _PressureProbeCoord(
        init_time=APPEND_DIM_START,
        lead_time=LEAD_TIMES[0],
        pressure_level=123.0,  # not in PRESSURE_LEVELS -> chunk_key None -> remaining
    )

    readonly_store = repo.readonly_session("main").store
    remaining = probe_job.filter_already_present(
        [present, absent, out_of_coords], readonly_store
    )

    assert present not in remaining  # already in the manifest
    assert absent in remaining  # not yet written
    assert out_of_coords in remaining  # not a position in the dataset


def test_virtual_operational_expands_both_groups(tmp_path: Path) -> None:
    # Per-group append-dim growth via sync_dims_to: start with empty main (0 inits),
    # then operationally expand both groups to 2 inits.
    dataset = _make_dataset(tmp_path, n_inits=2)
    template_utils.write_metadata(_create_template_ds(0), dataset.store_factory)
    full_template = _create_template_ds(2)
    job = _make_region_job(full_template, region=slice(0, 2), processing_mode="update")

    dataset._run_virtual_operational_update([job], worker_index=0, workers_total=1)

    assert list(_primary_repo(dataset.store_factory).list_branches()) == ["main"]
    _assert_all_values(dataset, n_inits=2)
