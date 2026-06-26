"""End-to-end tests pinning what happens to a dataset when source files fail
mid-update: where NaN holes appear, whether trimming and resume remove them, and
which validators do (and do not) catch them.

These are the production failure modes the review flagged: a transient download
failure or a lagging variable can leave a permanent NaN hole that survives the
normal resume-from-latest update, and shard-presence validation does not notice
because the (NaN-filled) shard is still written.

Marked `slow` because each test runs `_process_region_jobs` end-to-end.
"""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import computed_field
from zarr.abc.store import Store

from reformatters.common import (
    materialized_region_job as materialized_region_job_module,
)
from reformatters.common import template_utils, validation
from reformatters.common.config_models import (
    BaseInternalAttrs,
    DataVar,
    DataVarAttrs,
    Encoding,
)
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.materialized_region_job import MaterializedRegionJob
from reformatters.common.region_job import RegionJob, SourceFileCoord
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import (
    AppendDim,
    ArrayFloat32,
    DatetimeLike,
    Dim,
    Timedelta,
    Timestamp,
)

pytestmark = pytest.mark.slow

_LAT_SIZE = 10
_LON_SIZE = 25
_START = pd.Timestamp("2025-01-01")
_FREQ = pd.Timedelta("1h")


@pytest.fixture(autouse=True)
def _use_thread_pool_for_shard_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip ProcessPoolExecutor spawn overhead; SharedMemory works in-process."""
    monkeypatch.setattr(
        materialized_region_job_module, "ProcessPoolExecutor", ThreadPoolExecutor
    )


@pytest.fixture(autouse=True)
def _reset_failure_injection() -> None:
    """Clear the injected failures between tests so class state never leaks."""
    FailRegionJob.fail_download_times = frozenset()
    FailRegionJob.fail_read_var_times = frozenset()
    FailRegionJob.update_end_time = _START


def _time(step: int) -> pd.Timestamp:
    return _START + step * _FREQ


class FailDataVar(DataVar[BaseInternalAttrs]):
    encoding: Encoding = Encoding(
        dtype="float32",
        fill_value=np.nan,
        chunks=(1, _LAT_SIZE, _LON_SIZE),
        shards=(2, _LAT_SIZE, _LON_SIZE),
        compressors=[],
    )
    attrs: DataVarAttrs = DataVarAttrs(
        units="C",
        long_name="Test variable",
        short_name="test",
        step_type="instant",
    )
    internal_attrs: BaseInternalAttrs = BaseInternalAttrs(keep_mantissa_bits=10)


class FailSourceFileCoord(SourceFileCoord):
    time: Timestamp

    def get_url(self) -> str:
        return f"https://test.org/{self.time.isoformat()}"


class FailRegionJob(MaterializedRegionJob[FailDataVar, FailSourceFileCoord]):
    # Times whose download raises (a transient 5xx / 404 — both become NaN).
    fail_download_times: ClassVar[frozenset[pd.Timestamp]] = frozenset()
    # (variable name, time) pairs whose read raises (one variable lags behind).
    fail_read_var_times: ClassVar[frozenset[tuple[str, pd.Timestamp]]] = frozenset()
    # End (exclusive) the next operational_update_jobs call expands the template to.
    update_end_time: ClassVar[pd.Timestamp] = _START

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[FailDataVar],  # noqa: ARG002
    ) -> Sequence[FailSourceFileCoord]:
        return [
            FailSourceFileCoord(time=pd.Timestamp(t))
            for t in processing_region_ds[self.append_dim].values
        ]

    def download_file(self, coord: FailSourceFileCoord) -> Path:
        if coord.time in type(self).fail_download_times:
            raise RuntimeError(f"simulated transient download failure at {coord.time}")
        return Path("testfile")

    def read_data(
        self,
        coord: FailSourceFileCoord,
        data_var: FailDataVar,
    ) -> ArrayFloat32:
        if (data_var.name, coord.time) in type(self).fail_read_var_times:
            raise RuntimeError(
                f"simulated read failure for {data_var.name} {coord.time}"
            )
        return np.ones((_LAT_SIZE, _LON_SIZE), dtype=np.float32)

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[FailDataVar],
        reformat_job_name: str,
    ) -> tuple[Sequence[RegionJob[FailDataVar, FailSourceFileCoord]], xr.Dataset]:
        # Mirror the standard resume pattern: reprocess from the latest time already
        # in the store through the (injected) end time.
        existing_ds = xr.open_zarr(primary_store, decode_timedelta=True, chunks=None)
        append_dim_start = pd.Timestamp(existing_ds[append_dim].max().values)
        template_ds = get_template_fn(cls.update_end_time)
        jobs = cls.get_jobs(
            tmp_store=tmp_store,
            template_ds=template_ds,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            filter_start=append_dim_start,
        )
        return jobs, template_ds


class FailTemplateConfig(TemplateConfig[FailDataVar]):
    dims: tuple[Dim, ...] = ("time", "latitude", "longitude")
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = _START
    append_dim_frequency: Timedelta = _FREQ

    @computed_field
    @property
    def dataset_id(self) -> str:
        return "test-failure-injection-dataset"

    @computed_field
    @property
    def version(self) -> str:
        return "v1.0"

    @computed_field
    @property
    def data_vars(self) -> list[FailDataVar]:
        return [FailDataVar(name="var0"), FailDataVar(name="var1")]


class FailDataset(DynamicalDataset[FailDataVar, FailSourceFileCoord]):
    template_config: FailTemplateConfig = FailTemplateConfig()
    region_job_class: type[FailRegionJob] = FailRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        return [
            ReformatCronJob(
                name=f"{self.dataset_id}-update",
                schedule="0 0 * * *",
                pod_active_deadline=timedelta(minutes=30),
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
                pod_active_deadline=timedelta(minutes=30),
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


def _make_template_ds(end_time: DatetimeLike) -> xr.Dataset:
    times = pd.date_range(_START, end_time, freq=_FREQ, inclusive="left")
    ds = xr.Dataset(
        {
            name: xr.Variable(
                data=np.full(
                    (len(times), _LAT_SIZE, _LON_SIZE), np.nan, dtype=np.float32
                ),
                dims=["time", "latitude", "longitude"],
                encoding={
                    "dtype": "float32",
                    "chunks": (1, _LAT_SIZE, _LON_SIZE),
                    "shards": (2, _LAT_SIZE, _LON_SIZE),
                    "fill_value": np.nan,
                },
            )
            for name in ("var0", "var1")
        },
        coords={
            "time": times,
            "latitude": np.linspace(0, 90, _LAT_SIZE),
            "longitude": np.linspace(0, 240, _LON_SIZE),
        },
        attrs={
            "dataset_id": "test-failure-injection-dataset",
            "dataset_version": "v1.0",
        },
    )
    ds["time"].encoding["fill_value"] = -1
    ds["latitude"].encoding["fill_value"] = np.nan
    ds["longitude"].encoding["fill_value"] = np.nan
    return ds


def _make_dataset(tmp_path: Path) -> FailDataset:
    return FailDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ZARR3
        ),
    )


def _run_jobs(
    dataset: FailDataset,
    all_jobs: Sequence[RegionJob[FailDataVar, FailSourceFileCoord]],
    template_ds: xr.Dataset,
    reformat_job_name: str,
    *,
    update_template_with_results: bool,
) -> None:
    dataset._process_region_jobs(
        all_jobs=all_jobs,
        worker_index=0,
        workers_total=1,
        reformat_job_name=reformat_job_name,
        template_ds=template_ds,
        tmp_store=dataset._tmp_store(),
        update_template_with_results=update_template_with_results,
    )


def _backfill_through(dataset: FailDataset, end_step: int) -> None:
    """Establish a clean store with steps [0, end_step) all successfully written."""
    template_ds = _make_template_ds(_time(end_step))
    template_utils.write_metadata(template_ds, dataset.store_factory)
    jobs = FailRegionJob.get_jobs(
        tmp_store=dataset._tmp_store(),
        template_ds=template_ds,
        append_dim="time",
        all_data_vars=FailTemplateConfig().data_vars,
        reformat_job_name="backfill",
    )
    _run_jobs(
        dataset, jobs, template_ds, "backfill", update_template_with_results=False
    )


def _run_update(dataset: FailDataset, end_step: int, job_name: str) -> None:
    """Run an operational update (resume-from-latest) extending through end_step."""
    FailRegionJob.update_end_time = _time(end_step)
    jobs, template_ds = FailRegionJob.operational_update_jobs(
        primary_store=dataset.store_factory.primary_store(),
        tmp_store=dataset._tmp_store(),
        get_template_fn=_make_template_ds,
        append_dim="time",
        all_data_vars=FailTemplateConfig().data_vars,
        reformat_job_name=job_name,
    )
    _run_jobs(dataset, jobs, template_ds, job_name, update_template_with_results=True)


def test_transient_mid_region_failure_leaves_permanent_hole(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)

    # Backfill steps 0..3 cleanly.
    _backfill_through(dataset, 4)
    backfilled = xr.open_zarr(dataset.store_factory.primary_store())
    assert backfilled.sizes["time"] == 4
    assert not np.isnan(backfilled["var0"].values).any()

    # Update through step 8 with a transient download failure at step 5.
    FailRegionJob.fail_download_times = frozenset({_time(5)})
    _run_update(dataset, 8, "update1")

    updated = xr.open_zarr(dataset.store_factory.primary_store())
    # Trimming keeps everything through the max *successful* step (7), so the failed
    # step 5 is NOT trimmed away — it is published as an all-NaN hole.
    assert updated.sizes["time"] == 8
    assert bool(np.isnan(updated["var0"].sel(time=_time(5)).values).all())
    # Its neighbors, including later steps, are fine.
    assert not np.isnan(updated["var0"].sel(time=_time(4)).values).any()
    assert not np.isnan(updated["var0"].sel(time=_time(6)).values).any()
    assert not np.isnan(updated["var0"].sel(time=_time(7)).values).any()

    # A normal resume-from-latest update does NOT reprocess step 5 (it is older than
    # the store's max time), so the hole is permanent without manual intervention.
    _run_update(dataset, 8, "update2")
    resumed = xr.open_zarr(dataset.store_factory.primary_store())
    assert bool(np.isnan(resumed["var0"].sel(time=_time(5)).values).all())

    # Shard-presence validation does NOT catch the hole: the shard covering step 5
    # was written (step 4 succeeded in the same shard), it is just NaN-filled.
    shard_result = validation.check_for_expected_shards(
        dataset.store_factory.primary_store(), resumed
    )
    assert shard_result.passed, shard_result.message

    # A NaN-fraction validator, by contrast, would catch it.
    assert float(resumed["var0"].isnull().mean()) > 0.0


def test_lagging_variable_publishes_ragged_nan_hole(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)

    _backfill_through(dataset, 4)

    # Update through step 8, but var1's reads fail for the last two steps while var0
    # fully succeeds. Trimming uses the max successful step across ALL variables (var0
    # reaches step 7), so var1 is published with a ragged NaN hole at steps 6 and 7.
    FailRegionJob.fail_read_var_times = frozenset(
        {("var1", _time(6)), ("var1", _time(7))}
    )
    _run_update(dataset, 8, "update1")

    updated = xr.open_zarr(dataset.store_factory.primary_store())
    assert updated.sizes["time"] == 8
    assert not np.isnan(updated["var0"].sel(time=_time(7)).values).any()
    assert bool(np.isnan(updated["var1"].sel(time=_time(6)).values).all())
    assert bool(np.isnan(updated["var1"].sel(time=_time(7)).values).all())

    # Again, shard-presence validation passes despite the ragged hole.
    shard_result = validation.check_for_expected_shards(
        dataset.store_factory.primary_store(), updated
    )
    assert shard_result.passed, shard_result.message
