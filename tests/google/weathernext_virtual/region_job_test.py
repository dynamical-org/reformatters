from collections.abc import Iterator, Sequence
from typing import Any, Literal
from unittest.mock import Mock

import pandas as pd
import pytest

from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.google.weathernext2.forecast_virtual.region_job import (
    GoogleWeathernext2ForecastHistoricalVirtualRegionJob,
    GoogleWeathernext2ForecastOperationalVirtualRegionJob,
    GoogleWeathernext2ForecastVirtualSourceFileCoord,
)
from reformatters.google.weathernext_virtual import region_job
from reformatters.google.weathernext_virtual.holdback import is_publishable
from reformatters.google.weathernext_virtual.listing import NativeObjectMetadata
from reformatters.google.weathernext_virtual.region_job import (
    WeatherNextVirtualRegionJob,
)
from reformatters.google.weathernext_virtual.validation import CheckNoRefsInsideHoldback
from tests.google.weathernext2.forecast_virtual.native_region_job_test import (
    HISTORICAL,
    OPERATIONAL,
    _coord,
    _job,
    _var,
)
from tests.google.weathernext3.forecast_virtual.datasets_test import DATASETS, job_for


@pytest.fixture(
    params=[None, *DATASETS], ids=["wn2", *[ds.dataset_id for ds in DATASETS]]
)
def unrestricted_update_job(
    request: pytest.FixtureRequest,
) -> WeatherNextVirtualRegionJob[Any, Any]:
    if request.param is None:
        job = _job(
            GoogleWeathernext2ForecastOperationalVirtualRegionJob,
            OPERATIONAL,
            OPERATIONAL.get_template(pd.Timestamp("2025-01-01T18:00")),
            OPERATIONAL.data_vars,
        )
    else:
        dataset = request.param
        job = job_for(dataset).model_copy(
            update={"data_vars": dataset.template_config.data_vars}
        )
    return job.model_copy(
        update={
            "processing_mode": "update",
            "publication_cutoff": pd.Timestamp.max,
            "poll_deadline": pd.Timestamp("2000-01-01"),
            "region": slice(0, 3),
        }
    )


def test_unrestricted_update_rejects_mixed_pressure_backlog_before_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(
        GoogleWeathernext2ForecastOperationalVirtualRegionJob,
        OPERATIONAL,
        OPERATIONAL.get_template(pd.Timestamp("2025-01-03")),
        OPERATIONAL.data_vars,
    ).model_copy(update={"processing_mode": "update", "region": slice(0, 8)})
    coords = job.source_file_coords()
    newest = max(coord.init_time for coord in coords)
    pending = [
        coord
        for coord in coords
        if coord.init_time == newest or coord.data_vars[0].group == "pressure_level"
    ]
    assert len(pending) == 3_360
    discovery = Mock(side_effect=AssertionError("discovery must not run"))
    monkeypatch.setattr(type(job), "discover_available", discovery)
    with pytest.raises(
        AssertionError,
        match=r"2426880.*1320960.*docs/weathernext2_holdback.md#unrestricted-rollout",
    ):
        list(job.process_virtual_refs(pending))
    discovery.assert_not_called()


@pytest.mark.parametrize("extra_ref", [False, True])
def test_unrestricted_update_reference_budget_boundary(
    unrestricted_update_job: WeatherNextVirtualRegionJob[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    extra_ref: bool,
) -> None:
    job = unrestricted_update_job
    coords = job.source_file_coords()
    oldest = min(coord.init_time for coord in coords)
    pending = [coord for coord in coords if coord.init_time == oldest]
    refs_per_init = sum(len(job._source_chunks(coord)) for coord in pending)
    bound = refs_per_init * job.manifest_init_split
    pending *= job.manifest_init_split
    if extra_ref:
        extra = coords[0].model_copy()
        count = type(job)._source_chunk_count
        monkeypatch.setattr(
            type(job),
            "_source_chunk_count",
            lambda self, coord: 1 if coord is extra else count(self, coord),
        )
        pending.append(extra)
    discovery = Mock(return_value=[])
    monkeypatch.setattr(type(job), "discover_available", discovery)
    if extra_ref:
        with pytest.raises(
            AssertionError,
            match=rf"{bound + 1}.*{bound}.*docs/weathernext2_holdback.md#unrestricted-rollout",
        ):
            list(job.process_virtual_refs(pending))
        discovery.assert_not_called()
    else:
        assert list(job.process_virtual_refs(pending)) == []
        discovery.assert_called_once_with(pending)


def test_unrestricted_update_admits_steady_state(
    unrestricted_update_job: WeatherNextVirtualRegionJob[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = unrestricted_update_job
    coords = job.source_file_coords()
    oldest = min(coord.init_time for coord in coords)
    first_lead = min(coord.lead_time for coord in coords)
    pending = [
        coord
        for coord in coords
        if coord.init_time != oldest or coord.lead_time == first_lead
    ]
    per_init = len(job.template_ds.to_dataset().lead_time) * len(job.data_vars)
    assert len(pending) == 2 * per_init + len(job.data_vars)
    discovery = Mock(return_value=[])
    monkeypatch.setattr(type(job), "discover_available", discovery)
    assert list(job.process_virtual_refs(pending)) == []
    discovery.assert_called_once_with(pending)


@pytest.mark.parametrize("historical", [False, True])
def test_wn2_source_chunk_count_matches_refs(historical: bool) -> None:
    config = HISTORICAL if historical else OPERATIONAL
    cls = (
        GoogleWeathernext2ForecastHistoricalVirtualRegionJob
        if historical
        else GoogleWeathernext2ForecastOperationalVirtualRegionJob
    )
    init = pd.Timestamp("2022-01-02T12:00" if historical else "2025-03-01T06:00")
    job = _job(
        cls, config, config.get_template(init + pd.Timedelta("6h")), config.data_vars
    )
    for var in config.data_vars:
        coord = _coord(config, [var], init)
        assert job._source_chunk_count(coord) == len(job._source_chunks(coord))
    coord = _coord(config, config.data_vars, init)
    assert job._source_chunk_count(coord) == len(job._source_chunks(coord))


def test_source_chunk_count_matches_refs(
    unrestricted_update_job: WeatherNextVirtualRegionJob[Any, Any],
) -> None:
    job = unrestricted_update_job
    init = pd.Timestamp(job.template_ds.to_dataset().init_time.values[0])
    lead = pd.Timedelta(job.template_ds.to_dataset().lead_time.values[0])
    for coord in job._coords_for_step(init, lead, job.data_vars):
        assert job._source_chunk_count(coord) == len(job._source_chunks(coord))


def test_unrestricted_eligibility_does_not_compare_valid_time() -> None:
    assert is_publishable(pd.Timestamp.max, pd.Timedelta("360h"), pd.Timestamp.max)


def test_unrestricted_step_enumeration_does_not_compare_valid_time(
    unrestricted_update_job: WeatherNextVirtualRegionJob[Any, Any],
) -> None:
    job = unrestricted_update_job
    processing_ds = job._processing_region_ds().isel(init_time=slice(0, 1))
    processing_ds = processing_ds.assign_coords(init_time=[pd.Timestamp.max])
    coords = job.generate_source_file_coords(processing_ds, job.data_vars)
    assert coords
    assert all(coord.init_time == pd.Timestamp.max for coord in coords)
    assert not job._step_coords(processing_ds, job.data_vars, publishable=False)


def test_unrestricted_holdback_validation_skips_manifest() -> None:
    job = _job(
        GoogleWeathernext2ForecastOperationalVirtualRegionJob,
        OPERATIONAL,
        OPERATIONAL.get_template(pd.Timestamp("2025-01-01T06:00")),
        OPERATIONAL.data_vars[:1],
    )
    context = Mock()
    context.virtual_region_job.return_value = job
    assert CheckNoRefsInsideHoldback().check(context).passed
    context.virtual_region_job.assert_called_once_with()


@pytest.mark.parametrize("historical", [True, False])
def test_discovery_deduplicates_queries_and_requires_every_object(
    monkeypatch: pytest.MonkeyPatch, historical: bool
) -> None:
    config = HISTORICAL if historical else OPERATIONAL
    cls = (
        GoogleWeathernext2ForecastHistoricalVirtualRegionJob
        if historical
        else GoogleWeathernext2ForecastOperationalVirtualRegionJob
    )
    init = pd.Timestamp("2022-01-02T12:00" if historical else "2025-03-01T06:00")
    var = _var(config, "pressure_level/temperature")
    job = _job(cls, config, config.get_template(init + pd.Timedelta("6h")), [var])
    first = _coord(config, [var], init)
    second = first.model_copy(
        update={"lead_time": pd.Timedelta("18h"), "chunk_metadata": {}}
    )
    objects = {
        chunk.location: NativeObjectMetadata(100, '"etag"')
        for coord in (first, second)
        for chunk in job._source_chunks(coord)
    }
    missing = job._source_chunks(second)[-1].location
    del objects[missing]
    objects["irrelevant"] = NativeObjectMetadata(20, '"ignored"')
    listing = Mock(return_value=objects)
    monkeypatch.setattr(region_job, "list_objects", listing)

    assert job.discover_available([first, second]) == [(first, 0)]
    assert listing.call_count == (1 if historical else 2)
    assert "irrelevant" not in first.chunk_metadata
    assert not second.chunk_metadata

    objects[missing] = NativeObjectMetadata(101, '"new"')
    listing.reset_mock()
    assert job.discover_available([first, second]) == [(first, 0), (second, 0)]
    assert listing.call_count == (1 if historical else 2)
    assert second.chunk_metadata[missing] == NativeObjectMetadata(101, '"new"')


@pytest.mark.parametrize("mode", ["backfill", "update"])
def test_manifest_batches_preserve_window_and_step_order(
    monkeypatch: pytest.MonkeyPatch, mode: Literal["backfill", "update"]
) -> None:
    template = HISTORICAL.get_template(pd.Timestamp("2022-03-01"))
    var = _var(HISTORICAL, "temperature_2m")
    job = _job(
        GoogleWeathernext2ForecastHistoricalVirtualRegionJob,
        HISTORICAL,
        template,
        [var],
    ).model_copy(update={"processing_mode": mode})
    inits = template.to_dataset().get_index("init_time")
    coords = [_coord(HISTORICAL, [var], inits[i]) for i in (129, 0, 128, 127)]
    calls = []

    def process(
        self: VirtualRegionJob[Any, Any],
        remaining: Sequence[GoogleWeathernext2ForecastVirtualSourceFileCoord],
    ) -> Iterator[
        Sequence[
            tuple[
                GoogleWeathernext2ForecastVirtualSourceFileCoord, Sequence[VirtualRef]
            ]
        ]
    ]:
        calls.append(list(remaining))
        yield [(coord, []) for coord in remaining]

    monkeypatch.setattr(VirtualRegionJob, "process_virtual_refs", process)
    list(job.process_virtual_refs(coords))

    assert calls == (
        [[coords[2], coords[0]], [coords[1], coords[3]]]
        if mode == "backfill"
        else [coords]
    )
