import time
from collections.abc import Buffer, Generator, Iterator, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    FIRST_EXCEPTION,
    Future,
    ThreadPoolExecutor,
    wait,
)
from datetime import UTC, datetime
from functools import partial
from time import monotonic
from typing import ClassVar

import obstore
import pandas as pd
import xarray as xr
from obstore.exceptions import GenericError, PermissionDeniedError, UnauthenticatedError
from obstore.store import ObjectStore
from pydantic import PrivateAttr

from reformatters.common.download import (
    SourceAccessError,
    s3_store_from_secret,
    source_access,
)
from reformatters.common.logging import get_logger
from reformatters.common.retry import retry
from reformatters.common.types import Timedelta
from reformatters.common.virtual_region_job import (
    VirtualRef,
    VirtualRegionJob,
    log_virtual_event,
)
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar

from .grib import (
    InterruptedGribStreamError,
    SourceContractError,
    StreamReader,
    inventory,
)
from .source import (
    ENDPOINT,
    SOURCE_PREFIX,
    SOURCE_SECRET,
    EcmwfFastSourceFileCoord,
    list_prefix_or_defer,
    listing_client,
)
from .template_config import FastModel, lead_hours

log = get_logger(__name__)


TARGET_RANGE_CHUNK = 64 * 1024**2


class EcmwfFastRegionJob(VirtualRegionJob[EcmwfDataVar, EcmwfFastSourceFileCoord]):
    model: ClassVar[FastModel]
    retain_failed_files: ClassVar[bool] = True
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("24h")
    update_download_concurrency: ClassVar[int] = 5
    _listed_etags: dict[str, tuple[int, str | None]] = PrivateAttr(default_factory=dict)
    _last_modified: dict[str, datetime] = PrivateAttr(default_factory=dict)

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[EcmwfDataVar],
    ) -> Sequence[EcmwfFastSourceFileCoord]:
        leads = set(pd.to_timedelta(processing_region_ds.lead_time.values))
        members = set(processing_region_ds.ensemble_member.values)
        return [
            EcmwfFastSourceFileCoord(
                model=self.model,
                family=family,
                init_time=init,
                lead_time=lead,
                data_vars=data_var_group,
            )
            for init in pd.to_datetime(processing_region_ds.init_time.values)
            for hours in lead_hours(self.model, init.hour)
            if (lead := pd.Timedelta(hours=hours)) in leads
            for family in ("cf", "pf")
            if members.intersection((0,) if family == "cf" else range(1, 51))
        ]

    def discover_available(
        self, pending: list[EcmwfFastSourceFileCoord]
    ) -> list[tuple[EcmwfFastSourceFileCoord, int]]:
        by_key = {coord.key: coord for coord in pending}
        prefixes = sorted({coord.prefix for coord in pending})
        if not prefixes:
            return []
        client = listing_client()
        available: dict[str, tuple[EcmwfFastSourceFileCoord, int]] = {}
        # A model_copy shares this dict with its source job.
        self._last_modified = self._last_modified.copy()
        self._listed_etags = self._listed_etags.copy()
        with ThreadPoolExecutor(max_workers=len(prefixes)) as pool:
            for rows in pool.map(
                partial(list_prefix_or_defer, client=client), prefixes
            ):
                for key, size, last_modified, etag in rows:
                    if (
                        size > 0
                        and (coord := by_key.get(key)) is not None
                        and self._last_modified.get(key) != last_modified
                    ):
                        available[key] = (coord, size)
                        self._last_modified[key] = last_modified
                        self._listed_etags[key] = (size, etag)
        return list(available.values())

    def process_virtual_refs(  # noqa: PLR0912
        self, remaining: Sequence[EcmwfFastSourceFileCoord]
    ) -> Iterator[Sequence[tuple[EcmwfFastSourceFileCoord, Sequence[VirtualRef]]]]:
        """Yield finished whole files while polling; drain admitted work at cutoff."""
        if self.processing_mode == "backfill":
            yield from super().process_virtual_refs(remaining)
            return
        pending = list(remaining)
        futures: dict[Future[list[VirtualRef]], EcmwfFastSourceFileCoord] = {}
        next_sweep = time.monotonic()
        first_sweep = True
        pool = ThreadPoolExecutor(self.update_download_concurrency)
        try:
            while pending:
                for future in futures:
                    if future.done():
                        future.result()
                cutoff = not first_sweep and pd.Timestamp.now() >= self.poll_deadline
                if not cutoff and (first_sweep or time.monotonic() >= next_sweep):
                    sweep_start = time.monotonic()
                    in_flight = {id(coord) for coord in futures.values()}
                    available = self.discover_available(
                        [coord for coord in pending if id(coord) not in in_flight]
                    )
                    received_at = pd.Timestamp.now(tz="UTC").isoformat()
                    log_virtual_event(
                        "discovery_sweep",
                        received_at=received_at,
                        duration_s=time.monotonic() - sweep_start,
                        files=len(available),
                        active_streams=sum(f.running() for f in futures),
                        outstanding_files=len(futures),
                    )
                    for coord, size in available:
                        log_virtual_event(
                            "discovered",
                            file=coord.get_url(),
                            received_at=received_at,
                            bytes=size,
                        )
                        futures[pool.submit(self._ready_file_refs, coord, size)] = coord
                    next_sweep = sweep_start + self.tick_interval.total_seconds()
                    first_sweep = False
                cutoff = pd.Timestamp.now() >= self.poll_deadline
                if cutoff and futures:
                    wait(futures, return_when=FIRST_EXCEPTION)
                completed = [
                    (futures.pop(future), future.result())
                    for future in list(futures)
                    if future.done()
                ]
                batch = [(coord, refs) for coord, refs in completed if refs]
                removed = {
                    id(coord)
                    for coord, refs in completed
                    if refs or not self.retain_failed_files
                }
                pending = [coord for coord in pending if id(coord) not in removed]
                if batch:
                    yield batch
                if not pending or cutoff:
                    if pending:
                        log.info(
                            f"Poll deadline reached with {len(pending)} source files "
                            f"not yet ingested, leaving them to the next update "
                            f"(first: {pending[0].get_url()})"
                        )
                    return
                timeout = max(
                    0.0,
                    min(
                        next_sweep - time.monotonic(),
                        (self.poll_deadline - pd.Timestamp.now()).total_seconds(),
                    ),
                )
                if futures:
                    wait(futures, timeout=timeout, return_when=FIRST_COMPLETED)
                else:
                    time.sleep(timeout)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def _ready_file_refs(
        self, coord: EcmwfFastSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        refs = self._file_refs_or_skip(coord, file_size)
        log_virtual_event(
            "ready",
            file=coord.get_url(),
            ready_at=pd.Timestamp.now(tz="UTC").isoformat(),
            ref_count=len(refs),
        )
        return refs

    def file_refs(
        self, coord: EcmwfFastSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        listed = self._listed_etags.get(coord.key)
        assert listed is None or listed[0] == file_size, (
            "File size differs from its discovery observation"
        )
        return retry(
            partial(
                self._stream_file_refs, coord, file_size, listed[1] if listed else None
            ),
            max_attempts=2,
            retryable_exceptions=(InterruptedGribStreamError, GenericError),
        )

    def _stream_file_refs(
        self, coord: EcmwfFastSourceFileCoord, file_size: int, etag: str | None
    ) -> list[VirtualRef]:
        store = s3_store_from_secret(
            SOURCE_PREFIX,
            region="auto",
            k8s_secret_name=SOURCE_SECRET,
            endpoint=ENDPOINT,
        )
        with source_access():
            last_modified = self._last_modified.get(coord.key)
            age_at_start = (
                f"{(datetime.now(UTC) - last_modified).total_seconds():.1f}s"
                if last_modified is not None
                else "unknown"
            )
            request_started = monotonic()
            if self.processing_mode == "update":
                log_virtual_event(
                    "request",
                    file=coord.get_url(),
                    key=coord.key,
                    requested_at=datetime.now(UTC).isoformat(),
                    last_modified=last_modified.isoformat()
                    if last_modified is not None
                    else None,
                    bytes=file_size,
                )
            chunks = None
            response = None
            reader = None
            try:
                if self.processing_mode == "update":
                    chunks = _ranged_chunks(store, coord.key, file_size, etag)
                    reader = StreamReader(chunks)
                    request_seconds = 0.0
                else:
                    response = obstore.get(store, coord.key)
                    request_seconds = monotonic() - request_started
                    if response.meta["size"] != file_size:
                        raise SourceContractError("Source size differs from listing")
                    reader = StreamReader(
                        iter(response.stream(min_chunk_size=64 * 1024))
                    )
                scan_started = monotonic()
                messages = inventory(reader, file_size, coord)
                scan_seconds = monotonic() - scan_started
            finally:
                if reader is not None:
                    reader.buffer = memoryview(b"")
                    reader.chunks = iter(())
                if chunks is not None:
                    chunks.close()
                response = None
        variables = {
            var.internal_attrs.grib_index_param: var for var in coord.data_vars
        }
        members = set(self.template_ds.ensemble_member.values)
        path_suffix = " ranges" if self.processing_mode == "update" else ""
        log.info(
            f"Scanned {coord.key}: {len(messages)} fields, {reader.position} bytes"
            f" (age_at_start={age_at_start}, request={request_seconds:.3f}s,"
            f" stream_scan={scan_seconds:.3f}s,"
            f" stream_wait={reader.stream_wait_seconds:.3f}s){path_suffix}"
        )
        return [
            VirtualRef(
                data_var=variables[message.param],
                out_loc={
                    "init_time": coord.init_time,
                    "lead_time": coord.lead_time,
                    "ensemble_member": message.member,
                },
                location=coord.get_url(),
                offset=message.offset,
                length=message.length,
            )
            for message in messages
            if message.param in variables and message.member in members
        ]


def _ranged_chunks(
    store: ObjectStore, key: str, size: int, etag: str | None
) -> Generator[Buffer]:
    if not etag or etag.startswith("W/"):
        raise SourceContractError("Source listing lacks a strong ETag")
    count = min(10, (size + TARGET_RANGE_CHUNK - 1) // TARGET_RANGE_CHUNK)
    assert count > 0
    buffers: list[Buffer] = [b""] * count
    futures: list[Future[None]] = []
    errors: list[BaseException] = []

    def fetch(index: int) -> None:
        start, end = size * index // count, size * (index + 1) // count
        response = payload = None
        try:
            response = obstore.get(
                store, key, options={"range": (start, end), "if_match": etag}
            )
            if response.range != (start, end):
                raise SourceContractError("Source returned a different range")
            if response.meta["size"] != size:
                raise SourceContractError("Source size differs from listing")
            if response.meta["e_tag"] != etag:
                raise SourceContractError("Source ETag differs from pinned version")
            payload = response.bytes()
            if len(payload) < end - start:
                raise InterruptedGribStreamError("Source range body was truncated")
            if len(payload) > end - start:
                raise SourceContractError(
                    "Source range body exceeds requested interval"
                )
            buffers[index] = payload
        finally:
            response = payload = None

    try:
        with ThreadPoolExecutor(count) as pool:
            futures.extend(pool.submit(fetch, index) for index in range(count))
        for future in futures:
            error = future.exception()
            if error is not None:
                errors.append(error)
        for error in errors:
            if isinstance(
                error,
                (
                    AssertionError,
                    SourceAccessError,
                    PermissionDeniedError,
                    UnauthenticatedError,
                ),
            ) or not isinstance(error, Exception):
                raise error
        for error in errors:
            if not isinstance(error, (InterruptedGribStreamError, GenericError)):
                raise error
        if errors:
            raise errors[0]
        yield from buffers
    finally:
        buffers.clear()
        futures.clear()
        errors.clear()
