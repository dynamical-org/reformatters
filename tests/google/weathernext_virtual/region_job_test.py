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
from reformatters.google.weathernext_virtual.validation import CheckNoRefsInsideHoldback
from tests.google.weathernext2.forecast_virtual.native_region_job_test import (
    HISTORICAL,
    OPERATIONAL,
    _coord,
    _job,
    _var,
)


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
