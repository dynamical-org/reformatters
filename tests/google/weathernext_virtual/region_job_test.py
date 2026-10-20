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


def test_unrestricted_update_rejects_backlog_before_discovery(
    unrestricted_update_job: WeatherNextVirtualRegionJob[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = unrestricted_update_job
    bound = (
        job.manifest_init_split
        * len(job.template_ds.to_dataset().lead_time)
        * len(job.data_vars)
    )
    coord = job.source_file_coords()[0]
    discovery = Mock(side_effect=AssertionError("discovery must not run"))
    monkeypatch.setattr(type(job), "discover_available", discovery)
    with pytest.raises(
        AssertionError,
        match=rf"{bound + 1}.*{bound}.*docs/weathernext2_holdback.md#unrestricted-rollout",
    ):
        list(job.process_virtual_refs([coord] * (bound + 1)))
    discovery.assert_not_called()


@pytest.mark.parametrize("at_bound", [False, True])
def test_unrestricted_update_admits_steady_state_and_batch_boundary(
    unrestricted_update_job: WeatherNextVirtualRegionJob[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    at_bound: bool,
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
    if at_bound:
        pending = [coords[0]] * (job.manifest_init_split * per_init)
    discovery = Mock(return_value=[])
    monkeypatch.setattr(type(job), "discover_available", discovery)
    assert list(job.process_virtual_refs(pending)) == []
    discovery.assert_called_once_with(pending)


def test_unrestricted_eligibility_does_not_compare_valid_time() -> None:
    assert is_publishable(pd.Timestamp.max, pd.Timedelta("360h"), pd.Timestamp.max)


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
