import builtins
import json
import re
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import pairwise
from pathlib import Path
from threading import Barrier, Event, Lock, Thread
from typing import Any, cast
from unittest.mock import MagicMock

import obstore
import pandas as pd
import pytest
from obstore.exceptions import (
    GenericError,
    PermissionDeniedError,
    PreconditionError,
    UnauthenticatedError,
)
from obstore.store import ObjectStore

from reformatters.common.download import SourceAccessError
from reformatters.common.virtual_region_job import VirtualRef
from reformatters.ecmwf.aifs_ens.forecast_fast.region_job import (
    EcmwfAifsEnsForecastFastRegionJob,
)
from reformatters.ecmwf.fast import grib, region_job
from reformatters.ecmwf.fast.grib import (
    InterruptedGribStreamError,
    SourceContractError,
    StreamReader,
    inventory,
)
from reformatters.ecmwf.fast.region_job import EcmwfFastRegionJob
from reformatters.ecmwf.fast.source import EcmwfFastSourceFileCoord, Family
from reformatters.ecmwf.ifs_ens.forecast_0_25_degree_fast.region_job import (
    EcmwfIfsEnsForecast025DegreeFastRegionJob,
)
from tests.ecmwf.fast.grib_test import grib1, source_bytes
from tests.ecmwf.fast.region_job_test import LAST_MODIFIED, make_job, source_response

ETAG = '"opaque-multipart-17"'
STORE = cast(ObjectStore, object())


class Payload(bytearray):
    pass


class Response:
    def __init__(
        self,
        data: builtins.bytes,
        interval: tuple[int, int],
        owners: list[weakref.ReferenceType[Payload]],
    ) -> None:
        self.data = data
        self.range = interval
        self.meta: dict[str, Any] = {"size": len(data), "e_tag": ETAG}
        self.owners = owners
        self.failure: Exception | None = None
        self.length_change = 0

    def bytes(self) -> Payload:
        start, end = self.range
        payload = Payload(self.data[start:end])
        if self.length_change < 0:
            del payload[self.length_change :]
        elif self.length_change > 0:
            payload.extend(b"x" * self.length_change)
        self.owners.append(weakref.ref(payload))
        if self.failure is not None:
            raise self.failure
        return payload


@pytest.fixture
def ranged_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[EcmwfFastRegionJob, EcmwfFastSourceFileCoord, bytes]:
    job = make_job(tmp_path, EcmwfIfsEnsForecast025DegreeFastRegionJob).model_copy(
        update={"processing_mode": "update"}
    )
    coord = next(
        c
        for c in job.source_file_coords()
        if c.family == "cf" and c.lead_time == pd.Timedelta("6h")
    )
    data = source_bytes("ifs")
    job._last_modified[coord.key] = LAST_MODIFIED
    job._listed_etags[coord.key] = (len(data), ETAG)
    monkeypatch.setattr(region_job, "TARGET_RANGE_CHUNK", (len(data) + 3) // 4)
    monkeypatch.setattr(
        region_job, "s3_store_from_secret", lambda *args, **kwargs: STORE
    )
    monkeypatch.setattr("reformatters.common.retry.time.sleep", lambda _: None)
    return job, coord, data


def getter(
    data: bytes, owners: list[weakref.ReferenceType[Payload]]
) -> Callable[..., Response]:
    def get(store: ObjectStore, key: str, *, options: dict[str, Any]) -> Response:
        assert options["if_match"] == ETAG
        return Response(data, options["range"], owners)

    return get


@pytest.mark.parametrize(
    ("size", "expected_count"),
    [
        (1, 1),
        (64 * 1024**2, 1),
        (64 * 1024**2 + 1, 2),
        (620_000_003, 10),
        (640 * 1024**2, 10),
        (640 * 1024**2 + 1, 10),
        (3 * 1024**3, 10),
    ],
)
def test_balanced_range_partition_and_request_cap(
    size: int, expected_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class SizedPayload:
        def __len__(self) -> int:
            return self.size

        def __init__(self, size: int) -> None:
            self.size = size

    def get(store: ObjectStore, key: str, *, options: dict[str, Any]) -> MagicMock:
        start, end = options["range"]
        calls.append((start, end))
        assert options["if_match"] == ETAG
        return MagicMock(
            range=(start, end),
            meta={"size": size, "e_tag": ETAG},
            bytes=lambda: SizedPayload(end - start),
        )

    monkeypatch.setattr(region_job.obstore, "get", get)
    assert (
        len(list(region_job._ranged_chunks(STORE, "key", size, ETAG))) == expected_count
    )
    assert len(calls) == expected_count
    intervals = sorted(calls)
    assert intervals[0][0] == 0
    assert intervals[-1][1] == size
    assert all(a[1] == b[0] for a, b in pairwise(intervals))
    lengths = [end - start for start, end in intervals]
    if size <= 640 * 1024**2:
        assert max(lengths) <= 64 * 1024**2
    else:
        assert max(lengths) > 64 * 1024**2
    assert max(lengths) - min(lengths) <= 1


@pytest.mark.parametrize(
    "cls",
    [EcmwfIfsEnsForecast025DegreeFastRegionJob, EcmwfAifsEnsForecastFastRegionJob],
)
@pytest.mark.parametrize("family", ["cf", "pf"])
def test_all_update_files_use_conditional_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cls: type[EcmwfFastRegionJob],
    family: Family,
) -> None:
    job = make_job(tmp_path, cls).model_copy(update={"processing_mode": "update"})
    coord = next(
        c
        for c in job.source_file_coords()
        if c.family == family and c.lead_time == pd.Timedelta("6h")
    )
    data = source_bytes(cls.model, family=family)
    job._listed_etags = {coord.key: (len(data), ETAG)}
    owners: list[weakref.ReferenceType[Payload]] = []
    get = MagicMock(side_effect=getter(data, owners))
    monkeypatch.setattr(region_job, "s3_store_from_secret", lambda *a, **kw: STORE)
    monkeypatch.setattr(region_job.obstore, "get", get)
    assert job.file_refs(coord, len(data))
    get.assert_called_once_with(
        STORE, coord.key, options={"range": (0, len(data)), "if_match": ETAG}
    )
    assert all(owner() is None for owner in owners)


@pytest.mark.parametrize("damage", ["range", "size", "etag", "extra", "short"])
def test_response_contract_and_retry(
    ranged_job: tuple[EcmwfFastRegionJob, EcmwfFastSourceFileCoord, bytes],
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    job, coord, data = ranged_job
    owners: list[weakref.ReferenceType[Payload]] = []
    calls = []

    def get(store: ObjectStore, key: str, *, options: dict[str, Any]) -> Response:
        calls.append(options["range"])
        result = Response(data, options["range"], owners)
        if damage == "range":
            result.range = (0, 1)
        elif damage == "size":
            result.meta["size"] += 1
        elif damage == "etag":
            result.meta["e_tag"] = '"different"'
        else:
            result.length_change = -1 if damage == "short" else 1
        return result

    def backoff(_: float) -> None:
        assert len(calls) == 4
        assert all(owner() is None for owner in owners)

    monkeypatch.setattr(region_job.obstore, "get", get)
    monkeypatch.setattr("reformatters.common.retry.time.sleep", backoff)
    with pytest.raises(
        InterruptedGribStreamError if damage == "short" else SourceContractError
    ):
        job.file_refs(coord, len(data))
    assert len(calls) == (8 if damage == "short" else 4)
    assert all(owner() is None for owner in owners)


@pytest.mark.parametrize(
    "fatal",
    [PermissionDeniedError, UnauthenticatedError, SourceAccessError, AssertionError],
)
def test_transient_sibling_cannot_mask_part_three_fatal(
    ranged_job: tuple[EcmwfFastRegionJob, EcmwfFastSourceFileCoord, bytes],
    monkeypatch: pytest.MonkeyPatch,
    fatal: type[Exception],
) -> None:
    job, coord, data = ranged_job
    count = (
        len(data) + region_job.TARGET_RANGE_CHUNK - 1
    ) // region_job.TARGET_RANGE_CHUNK
    gate = Barrier(count, timeout=5)
    calls = []
    owners: list[weakref.ReferenceType[Payload]] = []

    def get(store: ObjectStore, key: str, *, options: dict[str, Any]) -> Response:
        start = options["range"][0]
        calls.append(start)
        gate.wait()
        if start == 0:
            raise GenericError("transient")
        if start == len(data) * 2 // count:
            raise fatal("fatal part three")
        return Response(data, options["range"], owners)

    monkeypatch.setattr(region_job.obstore, "get", get)
    with pytest.raises(
        AssertionError if fatal is AssertionError else SourceAccessError
    ):
        job._file_refs_or_skip(coord, len(data))
    assert len(calls) == count
    assert all(owner() is None for owner in owners)


@pytest.mark.parametrize("token", [None, "", 'W/"weak"'])
def test_missing_or_weak_listing_etag_skips_without_head(
    ranged_job: tuple[EcmwfFastRegionJob, EcmwfFastSourceFileCoord, bytes],
    monkeypatch: pytest.MonkeyPatch,
    token: str | None,
) -> None:
    job, coord, data = ranged_job
    job._listed_etags[coord.key] = (len(data), token)
    get, head = MagicMock(), MagicMock()
    monkeypatch.setattr(region_job.obstore, "get", get)
    monkeypatch.setattr(region_job.obstore, "head", head)
    assert job._file_refs_or_skip(coord, len(data)) == []
    get.assert_not_called()
    head.assert_not_called()


@pytest.mark.parametrize("changed", [False, True])
def test_412_nonretryable_last_modified_reoffer_and_copy_isolation(
    ranged_job: tuple[EcmwfFastRegionJob, EcmwfFastSourceFileCoord, bytes],
    monkeypatch: pytest.MonkeyPatch,
    changed: bool,
) -> None:
    parent, coord, data = ranged_job
    job = parent.model_copy()
    get = MagicMock(side_effect=PreconditionError("changed"))
    monkeypatch.setattr(region_job.obstore, "get", get)
    assert job._file_refs_or_skip(coord, len(data)) == []
    assert get.call_count == 4
    monkeypatch.setattr(region_job, "listing_client", object)
    modified = LAST_MODIFIED + pd.Timedelta(seconds=int(changed))
    monkeypatch.setattr(
        region_job,
        "list_prefix_or_defer",
        lambda prefix, client: [(coord.key, len(data), modified, '"new"')],
    )
    assert job.discover_available([coord]) == ([(coord, len(data))] if changed else [])
    assert parent._listed_etags == {coord.key: (len(data), ETAG)}
    assert parent._last_modified == {coord.key: LAST_MODIFIED}
    assert job._listed_etags is not parent._listed_etags
    if changed:
        assert job._listed_etags[coord.key][1] == '"new"'


@pytest.mark.parametrize("scanner_failure", [False, True])
def test_owned_payloads_released_before_ready(
    ranged_job: tuple[EcmwfFastRegionJob, EcmwfFastSourceFileCoord, bytes],
    monkeypatch: pytest.MonkeyPatch,
    scanner_failure: bool,
) -> None:
    job, coord, data = ranged_job
    owners: list[weakref.ReferenceType[Payload]] = []
    if scanner_failure:
        data = b"bad!" + data[4:]
    monkeypatch.setattr(region_job.obstore, "get", getter(data, owners))

    def event(name: str, **fields: object) -> None:
        if name == "ready":
            assert len(owners) == 4
            assert all(owner() is None for owner in owners)

    monkeypatch.setattr(region_job, "log_virtual_event", event)
    monkeypatch.setattr(type(job), "_refs_cover_probe_chunk", lambda *args: True)
    assert bool(job._ready_file_refs(coord, len(data))) is not scanner_failure


@pytest.mark.parametrize(
    "cls",
    [EcmwfIfsEnsForecast025DegreeFastRegionJob, EcmwfAifsEnsForecastFastRegionJob],
)
def test_backfill_streams_at_existing_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cls: type[EcmwfFastRegionJob]
) -> None:
    job = make_job(tmp_path, cls)
    coord = next(
        c
        for c in job.source_file_coords()
        if c.family == "cf" and c.lead_time == pd.Timedelta("6h")
    )
    data = source_bytes(cls.model)
    get = MagicMock(return_value=source_response(data))
    monkeypatch.setattr(region_job, "s3_store_from_secret", lambda *a, **kw: STORE)
    monkeypatch.setattr(region_job.obstore, "get", get)
    monkeypatch.setattr(
        region_job, "_ranged_chunks", lambda *a: pytest.fail("backfill ranged")
    )
    assert len(job.file_refs(coord, len(data))) == 7
    get.assert_called_once_with(STORE, coord.key)
    assert cls.download_concurrency == 64
    assert not hasattr(cls, "update_download_concurrency")


@pytest.mark.parametrize("model", ["ifs", "aifs"])
def test_native_local_fixture_refs_equal_serial_and_ranged(
    model: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fixtures = Path(__file__).parent / "fixtures"
    rows = json.loads((fixtures / "metadata.json").read_text())
    cls = (
        EcmwfIfsEnsForecast025DegreeFastRegionJob
        if model == "ifs"
        else EcmwfAifsEnsForecastFastRegionJob
    )
    job = make_job(tmp_path, cls)
    local = obstore.store.LocalStore(str(fixtures))
    monkeypatch.setattr(region_job, "s3_store_from_secret", lambda *a, **kw: local)
    for row in rows:
        if row["model"] != model:
            continue
        coord = EcmwfFastSourceFileCoord(
            model=cls.model,
            family=row["family"],
            init_time=pd.Timestamp(row["init_time"]),
            lead_time=pd.Timedelta(hours=row["hours"]),
            data_vars=job.data_vars,
            source_params=frozenset({"2t"}),
            source_members=tuple(row["members"]),
        )
        monkeypatch.setattr(region_job, "TARGET_RANGE_CHUNK", (row["bytes"] + 3) // 4)
        with caplog.at_level("WARNING"):
            serial = job.file_refs(coord, row["bytes"])
        update = job.model_copy(update={"processing_mode": "update"})
        update._listed_etags = {
            coord.key: (row["bytes"], obstore.head(local, coord.key)["e_tag"])
        }
        with caplog.at_level("INFO"):
            ranged = update.file_refs(coord, row["bytes"])
        assert ranged == serial
        assert len(ranged) == 1
        assert ranged[0].etag_checksum is None
    scans = [r.getMessage() for r in caplog.records if "Scanned " in r.getMessage()]
    assert len(scans) == 8
    assert all("request=0.000s" in line and line.endswith(") ranges") for line in scans)


def test_range_boundary_inside_grib_header(
    ranged_job: tuple[EcmwfFastRegionJob, EcmwfFastSourceFileCoord, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, coord, _ = ranged_job
    data = grib1("2t", 0) + grib1("10u", 0) + grib1("10v", 0)
    coord = coord.model_copy(update={"source_params": frozenset({"2t", "10u", "10v"})})
    owners: list[weakref.ReferenceType[Payload]] = []
    job._listed_etags[coord.key] = (len(data), ETAG)
    monkeypatch.setattr(region_job, "TARGET_RANGE_CHUNK", (len(data) + 3) // 4)
    monkeypatch.setattr(region_job.obstore, "get", getter(data, owners))
    messages = inventory(StreamReader(iter([data])), len(data), coord)
    assert any(
        m.offset < len(data) * i // 4 < m.offset + 36
        for m in messages
        for i in range(1, 4)
    )
    refs = job.file_refs(coord, len(data))
    assert [(r.offset, r.length) for r in refs] == [
        (m.offset, m.length) for m in messages
    ]


@pytest.mark.parametrize(
    "cls",
    [EcmwfIfsEnsForecast025DegreeFastRegionJob, EcmwfAifsEnsForecastFastRegionJob],
)
def test_five_file_slots_drain_queued_work_at_cutoff_without_queued_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cls: type[EcmwfFastRegionJob]
) -> None:
    job = make_job(tmp_path, cls).model_copy(
        update={
            "processing_mode": "update",
            "poll_deadline": pd.Timestamp.now() - pd.Timedelta("1s"),
        }
    )
    coords = list(job.source_file_coords())[:12]
    release = Event()
    full = Event()
    state_lock = Lock()
    admitted = []
    owners: list[weakref.ReferenceType[Payload]] = []
    peak = 0
    active = 0
    sweeps = []

    def discover(
        self: EcmwfFastRegionJob, pending: list[EcmwfFastSourceFileCoord]
    ) -> list[tuple[EcmwfFastSourceFileCoord, int]]:
        sweeps.append(list(pending))
        return [(coord, 1) for coord in pending]

    def refs(
        self: EcmwfFastRegionJob, coord: EcmwfFastSourceFileCoord, size: int
    ) -> list[VirtualRef]:
        nonlocal active, peak
        payload = Payload(b"x")
        with state_lock:
            owners.append(weakref.ref(payload))
            admitted.append(coord)
            active += 1
            peak = max(peak, active)
            if active == 5:
                full.set()
        try:
            assert release.wait(5)
            return [
                VirtualRef(self.data_vars[0], coord.out_loc(), coord.get_url(), 0, size)
            ]
        finally:
            payload = None
            with state_lock:
                active -= 1

    monkeypatch.setattr(type(job), "discover_available", discover)
    monkeypatch.setattr(type(job), "file_refs", refs)
    monkeypatch.setattr(type(job), "_refs_cover_probe_chunk", lambda *args: True)
    with ThreadPoolExecutor(1) as driver:
        result = driver.submit(lambda: list(job.process_virtual_refs(coords)))
        try:
            assert full.wait(5)
            assert len(admitted) == len(owners) == 5
            assert sum(owner() is not None for owner in owners) == 5
        finally:
            release.set()
        batches = result.result(timeout=5)
    assert len(batches) == 1
    assert {id(coord) for coord, _ in batches[0]} == {id(coord) for coord in coords}
    assert len(admitted) == 12
    assert peak == 5
    assert len(sweeps) == 1
    assert all(owner() is None for owner in owners)


def test_lazy_fetch_preserves_scanned_regex_and_phase_metrics(
    ranged_job: tuple[EcmwfFastRegionJob, EcmwfFastSourceFileCoord, bytes],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    job, coord, data = ranged_job
    clock = 0.0
    owners: list[weakref.ReferenceType[Payload]] = []

    def get(store: ObjectStore, key: str, *, options: dict[str, Any]) -> Response:
        nonlocal clock
        clock = 2.5
        return Response(data, options["range"], owners)

    original_inventory = inventory

    def scan(
        reader: StreamReader, size: int, coord: EcmwfFastSourceFileCoord
    ) -> list[grib.Message]:
        nonlocal clock
        messages = original_inventory(reader, size, coord)
        assert reader.position == size
        clock += 0.25
        return messages

    monkeypatch.setattr(region_job, "monotonic", lambda: clock)
    monkeypatch.setattr(grib, "monotonic", lambda: clock)
    monkeypatch.setattr(region_job.obstore, "get", get)
    monkeypatch.setattr(region_job, "inventory", scan)
    with caplog.at_level("INFO"):
        assert job.file_refs(coord, len(data))
    pattern = re.compile(
        r"Scanned (\S+): (\d+) fields, (\d+) bytes \(age_at_start=(?:([\d.]+)s|unknown), request=([\d.]+)s, stream_scan=([\d.]+)s, stream_wait=([\d.]+)s\)"
    )
    matches = [
        match for r in caplog.records if (match := pattern.search(r.getMessage()))
    ]
    assert len(matches) == 1
    match = matches[0]
    assert match.group(1) == coord.key
    assert int(match.group(3)) == len(data)
    assert tuple(float(match.group(i)) for i in (5, 6, 7)) == (0.0, 2.75, 2.5)
    assert all(owner() is None for owner in owners)


def test_single_range_conditional_206_covers_whole_file(
    ranged_job: tuple[EcmwfFastRegionJob, EcmwfFastSourceFileCoord, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, coord, data = ranged_job
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append((self.headers.get("Range"), self.headers.get("If-Match")))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes 0-{len(data) - 1}/{len(data)}")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("ETag", ETAG)
            self.send_header("Last-Modified", "Wed, 04 Nov 2026 00:00:00 GMT")
            self.end_headers()
            self.wfile.write(data)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            store = obstore.store.HTTPStore(
                f"http://127.0.0.1:{server.server_port}",
                client_options={"allow_http": True},
            )
            monkeypatch.setattr(
                region_job, "s3_store_from_secret", lambda *a, **kw: store
            )
            monkeypatch.setattr(region_job, "TARGET_RANGE_CHUNK", 64 * 1024**2)
            assert len(job.file_refs(coord, len(data))) == 7
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert requests == [(f"bytes=0-{len(data) - 1}", ETAG)]


def test_scanner_assertion_releases_resident_payloads(
    ranged_job: tuple[EcmwfFastRegionJob, EcmwfFastSourceFileCoord, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, coord, data = ranged_job
    owners: list[weakref.ReferenceType[Payload]] = []

    def fail(
        reader: StreamReader, size: int, coord: EcmwfFastSourceFileCoord
    ) -> list[grib.Message]:
        reader.read(1)
        assert len(owners) == 4
        raise AssertionError("scanner invariant")

    monkeypatch.setattr(region_job.obstore, "get", getter(data, owners))
    monkeypatch.setattr(region_job, "inventory", fail)
    with pytest.raises(AssertionError, match="scanner invariant"):
        job.file_refs(coord, len(data))
    assert all(owner() is None for owner in owners)
