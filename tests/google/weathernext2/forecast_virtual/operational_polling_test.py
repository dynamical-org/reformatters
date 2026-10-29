import re
from collections.abc import Sequence
from pathlib import Path

import httpx
import pandas as pd
import pytest

from reformatters.google.weathernext2.forecast_virtual import (
    region_job as region_job_module,
)
from reformatters.google.weathernext2.forecast_virtual.region_job import (
    GoogleWeathernext2ForecastOperationalVirtualRegionJob,
    GoogleWeathernext2ForecastVirtualSourceFileCoord,
)
from reformatters.google.weathernext2.forecast_virtual.template_config import (
    GoogleWeathernext2DataVar,
)
from reformatters.google.weathernext_virtual import region_job
from reformatters.google.weathernext_virtual.listing import (
    PROXY_LOCATION_PREFIX,
    NativeObjectMetadata,
    ObjectListingQuery,
)
from tests.google.weathernext2.forecast_virtual.native_region_job_test import (
    OPERATIONAL,
    _var,
)

INIT = pd.Timestamp("2025-03-01T06:00")
SURFACE = _var(OPERATIONAL, "temperature_2m")
PRESSURE = _var(OPERATIONAL, "pressure_level/temperature")
MEMBER_GLOB = region_job_module._OPERATIONAL_MEMBER_GLOB


class FakeSource:
    """Published source keys, listed with the proxy's prefix + matchGlob semantics."""

    def __init__(self) -> None:
        self.keys: set[str] = set()
        self.queries: list[ObjectListingQuery] = []

    def publish(
        self,
        job: GoogleWeathernext2ForecastOperationalVirtualRegionJob,
        coords: Sequence[GoogleWeathernext2ForecastVirtualSourceFileCoord],
        *,
        skip_last: bool = False,
    ) -> None:
        for coord in coords:
            chunks = job._source_chunks(coord)
            for chunk in chunks[:-1] if skip_last else chunks:
                self.keys.add(chunk.location.removeprefix(PROXY_LOCATION_PREFIX))

    def list_objects(
        self, client: httpx.Client, query: ObjectListingQuery
    ) -> dict[str, NativeObjectMetadata] | None:
        assert client is not None
        self.queries.append(query)
        assert query.match_glob is not None
        assert query.delimiter == "/"
        pattern = re.compile(
            re.escape(query.match_glob)
            .replace(r"\*", "[^/]*")
            .replace(r"\{", "(")
            .replace(r"\}", ")")
            .replace(",", "|")
        )
        return {
            f"{PROXY_LOCATION_PREFIX}{key}": NativeObjectMetadata(1, '"etag"')
            for key in self.keys
            if key.startswith(query.prefix) and pattern.fullmatch(key)
        }

    def take(self) -> list[ObjectListingQuery]:
        queries, self.queries = self.queries, []
        return queries


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch) -> FakeSource:
    fake = FakeSource()
    monkeypatch.setattr(region_job, "list_objects", fake.list_objects)
    return fake


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    fake = Clock()
    monkeypatch.setattr(region_job_module, "_monotonic", fake)
    return fake


def _update_job(
    data_vars: Sequence[GoogleWeathernext2DataVar],
) -> GoogleWeathernext2ForecastOperationalVirtualRegionJob:
    return GoogleWeathernext2ForecastOperationalVirtualRegionJob(
        tmp_store=Path("unused.zarr"),
        template_ds=OPERATIONAL.get_template(INIT + pd.Timedelta("6h")),
        data_vars=data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
        processing_mode="update",
    )


def _coords(
    data_vars: Sequence[GoogleWeathernext2DataVar],
    lead_indices: Sequence[int],
    init: pd.Timestamp = INIT,
) -> list[GoogleWeathernext2ForecastVirtualSourceFileCoord]:
    return [
        GoogleWeathernext2ForecastVirtualSourceFileCoord(
            source_layout="operational",
            init_time=init,
            lead_time=pd.Timedelta("6h") * (lead + 1),
            data_vars=(var,),
        )
        for lead in lead_indices
        for var in data_vars
    ]


def _all_leads(var: GoogleWeathernext2DataVar, init: pd.Timestamp = INIT) -> str:
    return f"{_prefix(var, init)}{MEMBER_GLOB}.*"


def _prefix(var: GoogleWeathernext2DataVar, init: pd.Timestamp = INIT) -> str:
    return (
        f"weathernext_2_0_0/zarr/2025_to_present/{init:%Y%m%d}_{init:%H}hr_01_preds/"
        f"predictions.zarr/{var.internal_attrs.source_name}/"
    )


def _probe(
    init: pd.Timestamp = INIT, var: GoogleWeathernext2DataVar = SURFACE
) -> ObjectListingQuery:
    prefix = f"{_prefix(var, init)}63."
    return ObjectListingQuery(prefix, f"{prefix}*", "/")


def _per_lead(
    var: GoogleWeathernext2DataVar, lead: int, init: pd.Timestamp = INIT
) -> ObjectListingQuery:
    prefix = _prefix(var, init)
    return ObjectListingQuery(prefix, f"{prefix}{MEMBER_GLOB}.{lead}.*", "/")


def _available(
    job: GoogleWeathernext2ForecastOperationalVirtualRegionJob,
    pending: list[GoogleWeathernext2ForecastVirtualSourceFileCoord],
) -> list[tuple[int, str]]:
    available = job.discover_available(pending)
    ready = {id(coord) for coord, _ in available}
    pending[:] = [coord for coord in pending if id(coord) not in ready]
    return sorted((coord.lead_index, coord.data_vars[0].path) for coord, _ in available)


def test_complete_init_batches_each_variable(source: FakeSource, clock: Clock) -> None:
    job = _update_job([SURFACE, PRESSURE])
    pending = _coords([SURFACE, PRESSURE], range(60))
    source.publish(job, pending)

    assert len(_available(job, pending)) == 120
    assert set(source.take()) == {
        _probe(),
        _probe(var=PRESSURE),
        ObjectListingQuery(_prefix(SURFACE), _all_leads(SURFACE), "/"),
        ObjectListingQuery(_prefix(PRESSURE), _all_leads(PRESSURE), "/"),
    }
    assert not pending


def test_waiting_probes_every_pending_variable(
    source: FakeSource, clock: Clock
) -> None:
    job = _update_job([SURFACE, PRESSURE])
    pending = _coords([SURFACE, PRESSURE], range(60))

    for _ in range(130):
        assert _available(job, pending) == []
        assert set(source.take()) == {_probe(), _probe(var=PRESSURE)}
        clock.now += 15


def test_variables_and_leads_are_discovered_independently(
    source: FakeSource, clock: Clock
) -> None:
    job = _update_job([SURFACE, PRESSURE])
    pending = _coords([SURFACE, PRESSURE], [0, 1, 59])
    source.publish(job, _coords([PRESSURE], [59]))

    assert _available(job, pending) == [(59, PRESSURE.path)]
    assert set(source.take()) == {
        _probe(),
        _probe(var=PRESSURE),
        _per_lead(PRESSURE, 59),
    }
    clock.now += 15
    assert _available(job, pending) == []
    assert set(source.take()) == {_probe(), _probe(var=PRESSURE)}


@pytest.mark.parametrize("var", [SURFACE, PRESSURE])
def test_probe_requires_every_level_of_its_member(
    source: FakeSource, clock: Clock, var: GoogleWeathernext2DataVar
) -> None:
    job = _update_job([var])
    pending = _coords([var], [0])
    source.publish(job, pending, skip_last=True)
    source.keys.add(f"{_prefix(var)}63.0.99.0.0")

    assert _available(job, pending) == []
    assert source.take() == [_probe(var=var)]
    source.publish(job, pending)
    clock.now += 15
    assert _available(job, pending) == [(0, var.path)]
    assert set(source.take()) == {_probe(var=var), _per_lead(var, 0)}


def test_failed_confirmation_retries_on_next_tick(
    source: FakeSource, clock: Clock
) -> None:
    job = _update_job([SURFACE, PRESSURE])
    pending = _coords([SURFACE, PRESSURE], [0, 1])
    first = _coords([PRESSURE], [0])[0]
    source.publish(job, [first])
    source.keys.remove(
        job._source_chunks(first)[0].location.removeprefix(PROXY_LOCATION_PREFIX)
    )
    assert _available(job, pending) == []
    assert _per_lead(PRESSURE, 0) in source.take()

    source.publish(job, pending)
    clock.now += 15
    assert _available(job, pending) == sorted(
        [
            (0, SURFACE.path),
            (0, PRESSURE.path),
            (1, SURFACE.path),
            (1, PRESSURE.path),
        ]
    )
    assert set(source.take()) == {
        _probe(),
        _probe(var=PRESSURE),
        _per_lead(SURFACE, 0),
        _per_lead(SURFACE, 1),
        _per_lead(PRESSURE, 0),
        _per_lead(PRESSURE, 1),
    }


def test_successful_confirmation_does_not_delay_next_lead(
    source: FakeSource, clock: Clock
) -> None:
    job = _update_job([SURFACE])
    pending = _coords([SURFACE], [0, 1])
    source.publish(job, _coords([SURFACE], [0]))
    assert _available(job, pending) == [(0, SURFACE.path)]
    source.take()
    source.publish(job, pending)
    clock.now += 15
    assert _available(job, pending) == [(1, SURFACE.path)]
    assert set(source.take()) == {_probe(), _per_lead(SURFACE, 1)}


def test_old_holes_poll_slowly_without_delaying_current_init(
    source: FakeSource, clock: Clock
) -> None:
    job = _update_job([SURFACE])
    old_init = INIT - pd.Timedelta("6h")
    pending = _coords([SURFACE], [0], old_init) + _coords([SURFACE], [0])
    assert _available(job, pending) == []
    assert set(source.take()) == {_probe(old_init), _probe()}
    source.publish(job, pending)

    clock.now += 15
    assert _available(job, pending) == [(0, SURFACE.path)]
    assert pending[0].init_time == old_init
    assert set(source.take()) == {_probe(), _per_lead(SURFACE, 0)}
    clock.now += 584
    assert _available(job, pending) == []
    assert source.take() == []
    clock.now += 1
    assert _available(job, pending) == [(0, SURFACE.path)]
    assert set(source.take()) == {_probe(old_init), _per_lead(SURFACE, 0, old_init)}


@pytest.mark.parametrize("lead_count", [4, 5])
def test_batching_depends_only_on_current_due_leads(
    source: FakeSource, clock: Clock, lead_count: int
) -> None:
    job = _update_job([PRESSURE])
    pending = _coords([PRESSURE], range(lead_count))
    source.publish(job, pending)
    assert len(_available(job, pending)) == lead_count
    expected = (
        {_per_lead(PRESSURE, lead) for lead in range(lead_count)}
        if lead_count < 5
        else {ObjectListingQuery(_prefix(PRESSURE), _all_leads(PRESSURE), "/")}
    )
    assert set(source.take()) == {_probe(var=PRESSURE), *expected}


def test_batched_result_also_ingests_lead_that_arrived_after_probe(
    source: FakeSource, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _update_job([SURFACE])
    pending = _coords([SURFACE], range(6))
    source.publish(job, pending[:5])
    listing = source.list_objects

    def publish_after_probe(
        client: httpx.Client, query: ObjectListingQuery
    ) -> dict[str, NativeObjectMetadata] | None:
        result = listing(client, query)
        if query == _probe():
            source.publish(job, pending[5:])
        return result

    monkeypatch.setattr(region_job, "list_objects", publish_after_probe)
    assert len(_available(job, pending)) == 6
    assert len(source.take()) == 2


def test_backfill_batches_without_probing_or_throttling(
    source: FakeSource, clock: Clock
) -> None:
    job = _update_job([SURFACE, PRESSURE]).model_copy(
        update={"processing_mode": "backfill"}
    )
    pending = _coords([SURFACE, PRESSURE], range(60))
    source.publish(job, pending)
    assert len(_available(job, pending)) == 120
    assert set(source.take()) == {
        ObjectListingQuery(_prefix(var), _all_leads(var), "/")
        for var in (SURFACE, PRESSURE)
    }


def test_refused_listing_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, clock: Clock
) -> None:
    monkeypatch.setattr(region_job, "list_objects", lambda client, query: None)
    job = _update_job([SURFACE])
    with pytest.raises(RuntimeError, match="refused listing"):
        job.discover_available(_coords([SURFACE], range(60)))
