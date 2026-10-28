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
OTHER_SURFACE = _var(OPERATIONAL, "wind_u_10m")
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


def _probe(init: pd.Timestamp = INIT) -> ObjectListingQuery:
    prefix = _prefix(SURFACE, init)
    return ObjectListingQuery(f"{prefix}63.", f"{prefix}63.*", "/")


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


def test_complete_init_is_listed_once_per_variable(
    source: FakeSource, clock: Clock
) -> None:
    job = _update_job([SURFACE, PRESSURE])
    pending = _coords([SURFACE, PRESSURE], range(60))
    source.publish(job, pending)

    assert len(_available(job, pending)) == 120
    assert {query.match_glob for query in source.take()} == {
        _all_leads(SURFACE),
        _all_leads(PRESSURE),
    }
    assert pending == []


def test_waiting_sends_one_probe_per_tick(source: FakeSource, clock: Clock) -> None:
    job = _update_job([SURFACE, PRESSURE])
    pending = _coords([SURFACE, PRESSURE], range(60))

    assert _available(job, pending) == []
    assert len(source.take()) == 2  # the first tick's full listing
    for _ in range(100):
        clock.now += 3
        assert _available(job, pending) == []
        assert source.take() == [_probe()]


def test_newly_published_lead_is_listed_once(source: FakeSource, clock: Clock) -> None:
    job = _update_job([SURFACE, PRESSURE])
    pending = _coords([SURFACE, PRESSURE], range(60))
    _available(job, pending)
    source.take()

    source.publish(job, _coords([SURFACE, PRESSURE], [0]))
    clock.now += 3
    assert _available(job, pending) == [(0, PRESSURE.path), (0, SURFACE.path)]
    assert sorted(source.take()) == sorted(
        [_probe(), _per_lead(SURFACE, 0), _per_lead(PRESSURE, 0)]
    )

    clock.now += 3
    assert _available(job, pending) == []
    assert source.take() == [_probe()]


def test_incomplete_lead_is_relisted_with_backoff(
    source: FakeSource, clock: Clock
) -> None:
    job = _update_job([SURFACE, PRESSURE])
    pending = _coords([SURFACE, PRESSURE], range(60))
    _available(job, pending)
    source.take()

    # The probe variable is complete for lead 0, the pressure variable not yet.
    source.publish(job, _coords([SURFACE], [0]))
    source.publish(job, _coords([PRESSURE], [0]), skip_last=True)
    clock.now += 3
    assert _available(job, pending) == [(0, SURFACE.path)]
    source.take()

    relisted_at = []
    for tick in range(1, 60):
        clock.now += 3
        _available(job, pending)
        if _per_lead(PRESSURE, 0) in source.take():
            relisted_at.append(tick * 3)
    # Backoff from 10 s doubling to 60 s, checked on 3 s ticks.
    assert relisted_at == [12, 33, 75, 135]

    source.publish(job, _coords([PRESSURE], [0]))
    while pending[0].lead_index == 0:
        clock.now += 3
        _available(job, pending)
    assert all(coord.lead_index > 0 for coord in pending)


def test_everything_is_relisted_periodically_without_the_probe(
    source: FakeSource, clock: Clock
) -> None:
    job = _update_job([SURFACE, OTHER_SURFACE])
    pending = _coords([SURFACE, OTHER_SURFACE], range(60))
    _available(job, pending)
    source.take()

    # The probe variable never appears, but another variable is complete.
    source.publish(job, _coords([OTHER_SURFACE], range(60)))
    clock.now += 599
    assert _available(job, pending) == []
    assert source.take() == [_probe()]
    clock.now += 1
    assert len(_available(job, pending)) == 60
    assert {query.match_glob for query in source.take()} == {
        _all_leads(SURFACE),
        _all_leads(OTHER_SURFACE),
    }


def test_each_pending_init_has_its_own_probe(source: FakeSource, clock: Clock) -> None:
    later = INIT + pd.Timedelta("6h")
    job = GoogleWeathernext2ForecastOperationalVirtualRegionJob(
        tmp_store=Path("unused.zarr"),
        template_ds=OPERATIONAL.get_template(later + pd.Timedelta("6h")),
        data_vars=[SURFACE],
        append_dim="init_time",
        region=slice(0, 2),
        reformat_job_name="test",
        processing_mode="update",
    )
    pending = _coords([SURFACE], range(60)) + _coords([SURFACE], range(60), later)
    _available(job, pending)
    source.take()

    clock.now += 3
    _available(job, pending)
    assert sorted(source.take()) == sorted([_probe(), _probe(later)])


def test_few_due_leads_use_per_lead_queries(source: FakeSource, clock: Clock) -> None:
    job = _update_job([PRESSURE])
    pending = _coords([PRESSURE], range(4))

    _available(job, pending)
    assert sorted(source.take()) == sorted(
        _per_lead(PRESSURE, lead) for lead in range(4)
    )


def test_lagging_variable_waits_for_its_batched_retry(
    source: FakeSource, clock: Clock
) -> None:
    job = _update_job([SURFACE, PRESSURE])
    pending = _coords([SURFACE, PRESSURE], range(60))
    _available(job, pending)
    source.take()

    # The pressure variable lags: its listing for lead 0 comes back empty.
    source.publish(job, _coords([SURFACE], [0]))
    clock.now += 3
    assert _available(job, pending) == [(0, SURFACE.path)]
    source.take()

    # Later leads of the lagging variable aren't listed as they appear...
    for lead in range(1, 6):
        source.publish(job, _coords([SURFACE], [lead]))
        clock.now += 1.5
        assert _available(job, pending) == [(lead, SURFACE.path)]
        assert sorted(source.take()) == sorted([_probe(), _per_lead(SURFACE, lead)])

    # ...but together at its retry, as one all-leads listing.
    source.publish(job, _coords([PRESSURE], range(6)))
    clock.now += 3
    assert _available(job, pending) == [(lead, PRESSURE.path) for lead in range(6)]
    assert sorted(source.take(), key=str) == sorted(
        [
            _probe(),
            ObjectListingQuery(_prefix(PRESSURE), _all_leads(PRESSURE), "/"),
        ],
        key=str,
    )
