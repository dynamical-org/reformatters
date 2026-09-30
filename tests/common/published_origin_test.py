import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import icechunk
import pandas as pd
import pytest
import xarray as xr
from zarr.abc.store import Store

from reformatters.common import materialized_region_job, template_utils
from reformatters.common import parallel_coordination as pc
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import AppendDim, DatetimeLike
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day025DegreeDataset,
)
from tests.common.parallel_writes_test import (
    ParallelDataset,
    ParallelDataVar,
    ParallelRegionJob,
    ParallelTemplateConfig,
    _create_template_ds,
)


def factory() -> StoreFactory:
    return StoreFactory(
        primary_storage_config=StorageConfig(
            base_path="unused", format=DatasetFormat.ICECHUNK
        ),
        dataset_id="pin-test",
        template_config_version="v1",
    )


def write_origin(repo: icechunk.Repository, start: str) -> str:
    session = repo.writable_session("main")
    xr.Dataset(coords={"time": pd.date_range(start, periods=2)}).to_zarr(
        session.store, mode="w", consolidated=False
    )
    return session.commit("origin")


def test_pin_survives_crash_before_setup_and_single_worker(tmp_path: Path) -> None:
    sf = factory()
    repo = sf.icechunk_repos(sort="primary-first")[0][1]
    first = write_origin(repo, "2024-04-01")
    info = pc.pin_operational_update(
        sf,
        is_first=True,
        reformat_job_name="daily",
        append_dim="time",
        template_identity="image-and-layout",
    )
    assert info["repo_snapshots"]["primary"] == first
    write_origin(repo, "2020-11-29")
    retry = pc.pin_operational_update(
        sf,
        is_first=True,
        reformat_job_name="daily",
        append_dim="time",
        template_identity="image-and-layout",
    )
    follower = pc.pin_operational_update(
        sf,
        is_first=False,
        reformat_job_name="daily",
        append_dim="time",
        template_identity="image-and-layout",
    )
    assert retry == follower == info
    assert info["origin"] == "2024-04-01T00:00:00"
    assert info["branch_name"] == "_job_daily"
    assert json.loads(sf.read_all_coordination_files("daily", "pin")[0]) == info


def test_pin_rejects_changed_template_identity() -> None:
    sf = factory()
    repo = sf.icechunk_repos(sort="primary-first")[0][1]
    write_origin(repo, "2024-04-01")
    pc.pin_operational_update(
        sf,
        is_first=True,
        reformat_job_name="daily",
        append_dim="time",
        template_identity="one",
    )
    with pytest.raises(AssertionError, match="identity"):
        pc.pin_operational_update(
            sf,
            is_first=True,
            reformat_job_name="daily",
            append_dim="time",
            template_identity="two",
        )


def test_stale_worker_labels_rejected_before_write() -> None:
    old = xr.DataTree(
        xr.Dataset(coords={"time": pd.date_range("2024-04-01", periods=3)})
    )
    shifted = xr.DataTree(
        xr.Dataset(coords={"time": pd.date_range("2024-03-30", periods=5)})
    )
    with pytest.raises(AssertionError, match="labels"):
        template_utils.assert_append_labels_match(old, shifted, "time")
    template_utils.assert_append_labels_match(
        shifted, shifted.isel(time=slice(0, 3)), "time"
    )


def test_atomic_pin_does_not_overwrite() -> None:
    sf = factory()
    sf.create_coordination_file("daily", "pin/source.json", b'{"snapshot":"first"}')
    with pytest.raises(FileExistsError):
        sf.create_coordination_file(
            "daily", "pin/source.json", b'{"snapshot":"second"}'
        )
    assert sf.read_all_coordination_files("daily", "pin") == [b'{"snapshot":"first"}']


def test_worker_zero_after_reset_rejects_nonzero_workers_old_template(
    tmp_path: Path,
) -> None:
    sf = factory()
    repo = sf.icechunk_repos(sort="primary-first")[0][1]
    old = _create_template_ds(4)
    template_utils.write_metadata(old, sf)
    stale_jobs = ParallelRegionJob.get_jobs(
        tmp_store=tmp_path / "stale",
        template_ds=old,
        append_dim="time",
        all_data_vars=ParallelTemplateConfig().data_vars,
        reformat_job_name="daily",
    )
    shifted = xr.DataTree(
        _create_template_ds(6)
        .to_dataset()
        .assign_coords(time=pd.date_range("2024-12-31T22:00", periods=6, freq="h"))
    )
    shifted.coords["time"].encoding = old.coords["time"].encoding.copy()
    session = repo.writable_session("main")
    shifted.load().to_zarr(session.store, mode="w", consolidated=False)
    published = session.commit("layout reset")
    info = pc.pin_operational_update(
        sf,
        is_first=True,
        reformat_job_name="daily",
        append_dim="time",
        template_identity="same-image",
    )
    pc.parallel_setup(
        sf,
        is_first=True,
        workers_total=2,
        reformat_job_name="daily",
        branch_name=info["branch_name"],
        template_ds=shifted,
        tmp_store=tmp_path / "ready",
        icechunk_repos=sf.icechunk_repos(sort="primary-first"),
        consolidated=False,
        pinned_setup=info,
    )
    with pytest.raises(AssertionError, match="labels"):
        ParallelRegionJob.process_worker_jobs(
            stale_jobs, sf, info["branch_name"], 1, overwrite_chunks=True
        )
    assert repo.lookup_branch("main") == published


def test_daily_pin_before_reset_fails_cas_before_replica_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sf = factory().model_copy(
        update={
            "replica_storage_configs": (
                StorageConfig(base_path="replica", format=DatasetFormat.ZARR3),
            )
        }
    )
    repo = sf.icechunk_repos(sort="primary-first")[0][1]
    template = _create_template_ds(4)
    template_utils.write_metadata(template, sf)
    info = pc.pin_operational_update(
        sf,
        is_first=True,
        reformat_job_name="daily",
        append_dim="time",
        template_identity="same-image",
    )
    jobs = ParallelRegionJob.get_jobs(
        tmp_store=tmp_path,
        template_ds=template,
        append_dim="time",
        all_data_vars=ParallelTemplateConfig().data_vars,
        reformat_job_name="daily",
    )
    pc.parallel_setup(
        sf,
        is_first=True,
        workers_total=1,
        reformat_job_name="daily",
        branch_name=info["branch_name"],
        template_ds=template,
        tmp_store=tmp_path / "template",
        icechunk_repos=sf.icechunk_repos(sort="primary-first"),
        consolidated=False,
        pinned_setup=info,
    )
    published = write_origin(repo, "2024-12-31")

    def no_replica_metadata(*args: object, **kwargs: object) -> None:
        if kwargs.get("zarr3_only"):
            pytest.fail("replica metadata copied before successful primary CAS")

    monkeypatch.setattr(pc, "copy_zarr_metadata", no_replica_metadata)
    with pytest.raises(RuntimeError, match="main moved"):
        pc.finalize(
            sf,
            all_jobs=jobs,
            merged_results={},
            reformat_job_name="daily",
            branch_name=info["branch_name"],
            template_ds=template,
            tmp_store=tmp_path / "final",
            setup_info=info,
            workers_total=1,
            update_template_with_results=False,
            consolidated=False,
            publish_zarr3_metadata=True,
        )
    assert repo.lookup_branch("main") == published


def test_ens_origin_policy_and_temporal_attributes() -> None:
    dataset = EcmwfIfsEnsForecast15Day025DegreeDataset(
        primary_storage_config=StorageConfig(
            base_path="test", format=DatasetFormat.ICECHUNK
        )
    )
    assert dataset.approved_origins == (pd.Timestamp("2024-04-01"),)
    with pytest.raises(AssertionError, match="Unapproved"):
        dataset._template_config_for_origin(pd.Timestamp("2024-03-30"))
    approved = dataset.model_copy(
        update={
            "approved_origins": (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-03-30"))
        }
    )
    config = approved._template_config_for_origin(pd.Timestamp("2024-03-30"))
    template = config.get_template(pd.Timestamp("2024-04-02"))
    assert template.coords["init_time"].values[0] == pd.Timestamp("2024-03-30")
    assert "2024-03-30" in template.attrs["time_domain"]
    for name in ("init_time", "valid_time"):
        assert (
            template.coords[name].attrs["statistics_approximate"]["min"]
            == "2024-03-30T00:00:00"
        )
    assert config.append_dim_coordinate_chunk_size() == 5840
    assert dataset.template_config.append_dim_start == pd.Timestamp("2024-04-01")


def test_non_ens_operational_setup_preserves_jobs_and_commit_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        materialized_region_job, "ProcessPoolExecutor", ThreadPoolExecutor
    )
    dataset = ParallelDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    initial = _create_template_ds(2)
    template_utils.write_metadata(initial, dataset.store_factory)
    monkeypatch.setattr(
        ParallelTemplateConfig,
        "get_template",
        lambda self, end_time: _create_template_ds(4),
    )

    def jobs(
        cls: type[ParallelRegionJob],
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[ParallelDataVar],
        reformat_job_name: str,
    ) -> tuple[Sequence[ParallelRegionJob], xr.DataTree]:
        template = get_template_fn(pd.Timestamp("2025-01-01T04:00"))
        return cls.get_jobs(
            tmp_store=tmp_store,
            template_ds=template,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
        ), template

    monkeypatch.setattr(ParallelRegionJob, "operational_update_jobs", classmethod(jobs))
    old_jobs, old_template = dataset._operational_update_jobs(
        "daily", tmp_path / "before"
    )
    original = dataset._operational_update_jobs

    def check_pinned(
        self: ParallelDataset,
        reformat_job_name: str,
        tmp_store: Path,
        *,
        primary_store: Store | None = None,
        template_config: TemplateConfig[ParallelDataVar] | None = None,
    ) -> object:
        assert (
            self.store_factory.read_coordination_file("daily", "pin/source.json")
            is not None
        )
        current_jobs, current_template = original(
            reformat_job_name,
            tmp_store,
            primary_store=primary_store,
            template_config=template_config,
        )
        xr.testing.assert_identical(current_template, old_template)
        assert [(job.region, job.data_vars) for job in current_jobs] == [
            (job.region, job.data_vars) for job in old_jobs
        ]
        return current_jobs, current_template

    monkeypatch.setattr(ParallelDataset, "_operational_update_jobs", check_pinned)
    repo = dict(dataset.store_factory.icechunk_repos(sort="primary-first"))["primary"]
    baseline = repo.lookup_branch("main")
    dataset.update("daily")
    history = list(repo.ancestry(branch="main"))
    assert history[3].id == baseline
    assert history[2].message == "Expand dataset"
    assert history[1].message.startswith("Update worker 0")
    assert history[0].message.startswith("Update at")
    assert (
        dataset.store_factory.read_coordination_file(
            "daily", "publication/complete.json"
        )
        is not None
    )
