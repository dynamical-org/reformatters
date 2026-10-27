import json
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import computed_field
from zarr.storage import MemoryStore

from reformatters.common import template_utils, validation
from reformatters.common.storage import (
    DatasetFormat,
    IcechunkVirtualConfig,
    StorageConfig,
    manifest_append_dim_split,
)
from reformatters.common.types import DatetimeLike
from reformatters.common.virtual_region_job import VirtualRef
from reformatters.noaa.rtma.analysis_virtual.template_config import (
    NoaaRtmaAnalysisVirtualTemplateConfig,
)
from reformatters.noaa.rtma.rtma_config_models import RtmaDataVar
from reformatters.noaa.urma.analysis_virtual.dynamical_dataset import (
    NoaaUrmaAnalysisVirtualDataset,
)
from reformatters.noaa.urma.analysis_virtual.region_job import (
    SOURCE_PREFIX,
    NoaaUrmaAnalysisVirtualRegionJob,
    NoaaUrmaAnalysisVirtualSourceFileCoord,
)
from reformatters.noaa.urma.analysis_virtual.template_config import (
    NoaaUrmaAnalysisVirtualTemplateConfig,
)

START = pd.Timestamp("2026-10-12T11:00")
FILE_TIME = pd.Timestamp("2026-10-12T12:00")
FIRE = pd.Timestamp("2026-10-12T18:45")
CUTOFF = FIRE + pd.Timedelta("57min")


class TimingTemplateConfig(NoaaUrmaAnalysisVirtualTemplateConfig):
    def get_template(self, end_time: DatetimeLike) -> xr.DataTree:
        template = NoaaRtmaAnalysisVirtualTemplateConfig(
            append_dim_start=self.append_dim_start
        ).get_template(end_time)
        ds = template.dataset[["temperature_2m"]]
        ds.attrs = self.dataset_attributes.model_dump(exclude_none=True)
        return xr.DataTree(ds)

    @computed_field
    @property
    def data_vars(self) -> Sequence[RtmaDataVar]:
        return [v for v in super().data_vars if v.name == "temperature_2m"]


def make_dataset(tmp_path: Path) -> NoaaUrmaAnalysisVirtualDataset:
    return NoaaUrmaAnalysisVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
        template_config=TimingTemplateConfig(append_dim_start=START),
        region_job_class=NoaaUrmaAnalysisVirtualRegionJob,
        icechunk_virtual_config=IcechunkVirtualConfig(
            containers=(
                icechunk.VirtualChunkContainer(
                    url_prefix=SOURCE_PREFIX,
                    store=icechunk.s3_store(region="us-east-1", anonymous=True),
                ),
            ),
            manifest_split=manifest_append_dim_split(split_size=30000, dim="time"),
        ),
    )


@pytest.mark.parametrize("minute", [42, 44])
@pytest.mark.parametrize("missing_due_hour", [False, True])
def test_validation_checks_newest_due_hour_at_poll_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    minute: int,
    missing_due_hour: bool,
) -> None:
    now = pd.Timestamp("2026-10-12T19:00") + pd.Timedelta(minutes=minute)
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda cls, *args, **kwargs: now)
    )
    current = make_dataset(tmp_path).validators()[0]
    assert isinstance(current, validation.CheckCurrentData)
    context = validation.ValidationContext(
        store=MemoryStore(),
        ds=xr.Dataset(
            coords={"time": [START] if missing_due_hour else [START, FILE_TIME]}
        ),
        append_dim="time",
        append_dim_frequency=pd.Timedelta("1h"),
    )
    result = current.check(context)
    assert result.passed == (not missing_due_hour), result.message
    assert f"time={FILE_TIME.isoformat()}" in result.message
    assert current.max_delay == timedelta(hours=7, minutes=40)
    if missing_due_hour:
        assert "Missing time=" in result.message
        assert f"latest present is {START.isoformat()}" in result.message


def test_unshifted_window_and_actual_fire_poll_deadline(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    assert NoaaUrmaAnalysisVirtualRegionJob.operational_update_window == pd.Timedelta(
        "48h"
    )
    assert NoaaUrmaAnalysisVirtualRegionJob.tick_interval == pd.Timedelta("1s")
    (cron,) = dataset.operational_kubernetes_resources("test-image")
    assert cron.schedule == "45 * * * *"
    assert not cron.suspend
    assert cron.as_kubernetes_object()["spec"]["suspend"] is False
    assert (cron.workers_total, cron.parallelism) == (1, 1)
    assert cron.pod_active_deadline == timedelta(minutes=59)
    for now in (FIRE, FIRE + pd.Timedelta("25min"), CUTOFF):
        assert dataset._virtual_poll_deadline(now) == CUTOFF
    template_utils.write_metadata(
        dataset.template_config.get_template(START),
        dataset.store_factory,
        consolidated=False,
    )
    config = TimingTemplateConfig(
        append_dim_start=(FIRE - pd.Timedelta("72h")).floor("h"),
    )
    jobs, template = NoaaUrmaAnalysisVirtualRegionJob.operational_update_jobs(
        primary_store=dataset.store_factory.primary_store(),
        tmp_store=tmp_path,
        get_template_fn=config.get_template,
        append_dim="time",
        all_data_vars=config.data_vars,
        reformat_job_name="test-window",
        job_fire_time=FIRE,
    )
    (job,) = jobs
    assert isinstance(job, NoaaUrmaAnalysisVirtualRegionJob)
    times = [c.time for c in job.source_file_coords()]
    assert len(times) == 48
    assert times[0] == pd.Timestamp("2026-10-10T19:00")
    assert times[-1] == FIRE.floor("h")
    assert template.dataset.time.values[-1] == FIRE.floor("h").to_datetime64()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("publication_time", "early"),
    [
        ("2026-10-12T18:45", True),
        ("2026-10-12T18:58", False),
        ("2026-10-12T19:35:36", False),
    ],
)
def test_available_file_commits_before_missing_hours_or_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publication_time: str, early: bool
) -> None:
    dataset = make_dataset(tmp_path)
    template_utils.write_metadata(
        dataset.template_config.get_template(START),
        dataset.store_factory,
        consolidated=False,
    )
    (job,), _template = NoaaUrmaAnalysisVirtualRegionJob.operational_update_jobs(
        primary_store=dataset.store_factory.primary_store(),
        tmp_store=tmp_path,
        get_template_fn=dataset.template_config.get_template,
        append_dim="time",
        all_data_vars=dataset.template_config.data_vars,
        reformat_job_name="test-publication",
        job_fire_time=FIRE,
    )
    assert isinstance(job, NoaaUrmaAnalysisVirtualRegionJob)
    job = job.model_copy(update={"poll_deadline": CUTOFF})
    candidates = list(job.source_file_coords())
    sizes = {coord.time: size for coord, size in job.discover_available(candidates)}
    publication = pd.Timestamp(publication_time)
    now = FIRE if early else publication - pd.Timedelta("1s")
    late_publication = publication + pd.Timedelta("2s")
    events: list[tuple[pd.Timestamp, tuple[pd.Timestamp, ...]]] = []

    def discover(
        self: NoaaUrmaAnalysisVirtualRegionJob,
        pending: list[NoaaUrmaAnalysisVirtualSourceFileCoord],
    ) -> list[tuple[NoaaUrmaAnalysisVirtualSourceFileCoord, int]]:
        visible = xr.open_zarr(
            dataset.store_factory.primary_store(), consolidated=False, chunks=None
        )
        times = tuple(pd.Timestamp(t) for t in visible.time.values)
        events.append((now, times))
        if now > publication:
            assert FILE_TIME in times, (
                "The published file must commit before the next source poll"
            )
            assert times[-1] == FILE_TIME
            if now < late_publication:
                assert any(coord.time == START for coord in pending)
                assert all(coord.time != FILE_TIME for coord in pending)
        return [
            (coord, sizes[coord.time])
            for coord in pending
            if (coord.time == FILE_TIME and now >= publication)
            or (coord.time == START and now >= late_publication)
        ]

    def sleep(seconds: float) -> None:
        nonlocal now
        assert 0 <= seconds <= 1
        now += pd.Timedelta("1s")
        if now > late_publication + pd.Timedelta("1s"):
            now = CUTOFF

    monkeypatch.setattr(
        NoaaUrmaAnalysisVirtualRegionJob, "discover_available", discover
    )
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda cls, *args, **kwargs: now)
    )
    monkeypatch.setattr("reformatters.common.virtual_region_job.time.sleep", sleep)
    repo = dataset.store_factory.icechunk_repos(sort="primary-first")[0][1]
    job.process_virtual(repo, (), "main", candidates)
    assert now == CUTOFF
    assert events[1 if early else 2][1] == (START, FILE_TIME)
    assert events[1 if early else 2][0] == publication + pd.Timedelta("1s")
    assert len(events) >= 3
    store = dataset.store_factory.primary_store()
    assert isinstance(store, icechunk.IcechunkStore)
    ds = xr.open_zarr(store, consolidated=False, chunks=None)
    np.testing.assert_array_equal(
        ds.time.values, [START.to_datetime64(), FILE_TIME.to_datetime64()]
    )
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures/urma_20261012T12.json").read_text()
    )
    temperature = next(m for m in fixture["messages"] if m["element"] == "TMP")
    point = fixture["points"][0]
    np.testing.assert_allclose(
        ds.temperature_2m.sel(time=FILE_TIME).isel(y=point["y"], x=point["x"]).item(),
        temperature["values"][0] - 273.15,
        rtol=0,
        atol=1e-12,
    )
    context = validation.ValidationContext(
        store=store,
        ds=ds,
        append_dim="time",
        append_dim_frequency=pd.Timedelta("1h"),
        data_vars=dataset.template_config.data_vars,
        region_job=job,
    )
    for check in dataset.validators():
        result = check.check(context)
        assert result.passed, result.message
    assert job.filter_already_present(candidates, store) == candidates[2:]


def test_arrival_after_cutoff_retried_without_holding_other_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    variables = TimingTemplateConfig().data_vars
    job = NoaaUrmaAnalysisVirtualRegionJob(
        tmp_store=tmp_path,
        template_ds=TimingTemplateConfig(append_dim_start=START).get_template(
            START + pd.Timedelta("3h")
        ),
        data_vars=variables,
        append_dim="time",
        region=slice(0, 3),
        reformat_job_name="test-retry",
        processing_mode="update",
        poll_deadline=CUTOFF,
    )
    coords = list(job.source_file_coords())
    now = FIRE
    poll_cutoff = CUTOFF

    def discover(
        self: NoaaUrmaAnalysisVirtualRegionJob,
        pending: list[NoaaUrmaAnalysisVirtualSourceFileCoord],
    ) -> list[tuple[NoaaUrmaAnalysisVirtualSourceFileCoord, int]]:
        return [
            (coord, 1)
            for coord in pending
            if coord.time == FILE_TIME or (coord.time == START and now > CUTOFF)
        ]

    def refs(
        self: NoaaUrmaAnalysisVirtualRegionJob,
        coord: NoaaUrmaAnalysisVirtualSourceFileCoord,
        file_size: int,
    ) -> list[VirtualRef]:
        assert file_size == 1
        return [VirtualRef(variables[0], coord.out_loc(), coord.get_url(), 0, 1)]

    def sleep(seconds: float) -> None:
        nonlocal now
        assert 0 <= seconds <= 1
        now = poll_cutoff

    monkeypatch.setattr(
        NoaaUrmaAnalysisVirtualRegionJob, "discover_available", discover
    )
    monkeypatch.setattr(NoaaUrmaAnalysisVirtualRegionJob, "file_refs", refs)
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda cls, *args, **kwargs: now)
    )
    monkeypatch.setattr("reformatters.common.virtual_region_job.time.sleep", sleep)
    batches = job.process_virtual_refs(coords)
    assert [coord.time for coord, _refs in next(batches)] == [FILE_TIME]
    assert now == FIRE
    assert list(batches) == []
    assert now == CUTOFF
    now = FIRE + pd.Timedelta("1h")
    poll_cutoff = CUTOFF + pd.Timedelta("1h")
    next_job = job.model_copy(update={"poll_deadline": CUTOFF + pd.Timedelta("1h")})
    retry = next_job.process_virtual_refs([coords[0], coords[2]])
    assert [coord.time for coord, _refs in next(retry)] == [START]
    assert list(retry) == []
    assert now == poll_cutoff
