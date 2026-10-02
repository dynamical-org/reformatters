"""Transport for the ECMWF Data Store (ECDS) OGC-API Processes retrieval service.

ECDS has no addressable source files: a selection is submitted as a job, polled to
completion, and downloaded once from a short-lived signed URL. Signed URLs and
server-side results expire without a published SLA, so download immediately after
a job succeeds.
"""

import itertools
import json
import os
import re
import shutil
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, NoReturn
from urllib.parse import urljoin, urlsplit

import requests

from reformatters.common.logging import get_logger
from reformatters.common.retry import retry

from .grib_inventory import count_grib_messages

log = get_logger(__name__)

ECDS_API_URL: Final[str] = "https://ecds.ecmwf.int/api"
S2S_FORECASTS_PROCESS: Final[str] = "s2s-forecasts"
ECDS_FAILURE_STATUSES: Final[frozenset[str]] = frozenset(
    {"failed", "rejected", "dismissed", "cancelled"}
)
# Assigned by this client, not ECDS: the job or its result is gone from ECDS, or
# polling gave up on it. Either way the job is never resumed, only replaced.
EXPIRED_STATUS: Final[str] = "expired"
ABANDONED_STATUS: Final[str] = "abandoned"
TERMINAL_FAILURE_STATUSES: Final[frozenset[str]] = ECDS_FAILURE_STATUSES | {
    EXPIRED_STATUS,
    ABANDONED_STATUS,
}
REQUEST_TIMEOUT_SECONDS: Final[float] = 60
DOWNLOAD_TIMEOUT_SECONDS: Final[float] = 120
MAXIMUM_POLL_BACKOFF_EXPONENT: Final[int] = 6
RESUBMIT_WAIT_SECONDS: Final[float] = 60
RESUBMIT_BUDGET_SECONDS: Final[float] = 3600
PARALLEL_RANGE_DOWNLOADS: Final[int] = 4
MINIMUM_RANGE_BYTES: Final[int] = 8 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES: Final[int] = 1024 * 1024


class EcdsJobFailedError(Exception):
    """The job ended in a terminal status: ECDS failed it, or its result is gone."""


class EcdsRangeMismatchError(RuntimeError):
    """The result was not served as verifiable byte ranges of one result, so any parts were discarded."""


class _ResultGoneError(Exception):
    pass


@dataclass(frozen=True)
class _Part:
    start: int
    end: int
    path: Path

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class _Transfer:
    connections: int
    transferred_bytes: int
    assembly_seconds: float


def _https_origin(url: str) -> tuple[str, int]:
    try:
        parsed = urlsplit(url)
        port = parsed.port
        hostname = parsed.hostname
    except ValueError:
        raise ValueError("ECDS requires a valid HTTPS URL") from None
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or any(character.isspace() for character in url)
    ):
        raise ValueError("ECDS requires an HTTPS URL without user information")
    return hostname, port if port is not None else 443


def _api_url(url: str, base_url: str) -> str:
    resolved = urljoin(base_url, url)
    if _https_origin(resolved) != _https_origin(base_url):
        raise ValueError("ECDS API URL must stay within the configured HTTPS origin")
    return resolved


class _EcdsSession(requests.Session):
    def __init__(self, api_url: str | None = None) -> None:
        super().__init__()
        self.api_origin = _https_origin(api_url) if api_url is not None else None

    def send(
        self,
        request: requests.PreparedRequest,
        **kwargs: Any,  # noqa: ANN401
    ) -> requests.Response:
        assert request.url is not None
        origin = _https_origin(request.url)
        if self.api_origin is not None:
            if origin != self.api_origin:
                raise ValueError(
                    "ECDS API URL must stay within the configured HTTPS origin"
                )
        else:
            request.headers.pop("PRIVATE-TOKEN", None)
            request.headers.pop("Authorization", None)
        return super().send(request, **kwargs)


@dataclass
class RequestState:
    """Durable record of one submitted ECDS job, so a restart can resume it."""

    request_id: str
    payload: dict[str, Any]
    submitted_at: str
    status_url: str
    status: str = "submitted"
    result_url: str | None = None
    downloaded_bytes: int | None = None
    grib_messages: int | None = None
    poll_failures: int = 0
    errors: list[str] = field(default_factory=list)


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> RequestState:
        return RequestState(**json.loads(self.path.read_text()))

    def read_if_exists(self) -> RequestState | None:
        return self.read() if self.path.exists() else None

    def write(self, state: RequestState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True))
        temporary_path.replace(self.path)


def process_url(
    process: str = S2S_FORECASTS_PROCESS, api_url: str | None = None
) -> str:
    base = (
        api_url
        or os.environ.get("ECDS_API_ENDPOINT")
        or read_cdsapi_config().get("url")
        or ECDS_API_URL
    ).rstrip("/")
    return f"{base}/retrieve/v1/processes/{process}"


def _post_inputs(
    url: str,
    inputs: Mapping[str, Any],
    session: requests.Session | None,
) -> dict[str, Any]:
    """POST an `inputs` body, retrying the transient 5xx these endpoints intermittently return."""

    def post() -> dict[str, Any]:
        response = (session or requests).post(
            url, json={"inputs": dict(inputs)}, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return dict(response.json())

    return retry(
        post, max_attempts=4, retryable_exceptions=(requests.RequestException,)
    )


def constraints(
    inputs: Mapping[str, Any],
    session: requests.Session | None = None,
    api_url: str | None = None,
) -> dict[str, list[str]]:
    """The values still valid for each selection key, given the other selected keys.

    Unauthenticated, and precise to a single `year`/`month`/`day`: an initialization
    ECDS does not hold returns empty lists.
    """
    return dict(
        _post_inputs(f"{process_url(api_url=api_url)}/constraints", inputs, session)
    )


def costing(
    inputs: Mapping[str, Any],
    session: requests.Session | None = None,
    api_url: str | None = None,
) -> tuple[float, float]:
    """Return `(cost, limit)` for a selection. Unauthenticated."""
    body = _post_inputs(f"{process_url(api_url=api_url)}/costing", inputs, session)
    return float(body["cost"]), float(body["limit"])


class EcdsRequest:
    """One ECDS job, with its progress persisted to `state_store`."""

    def __init__(
        self,
        state_store: StateStore,
        api_url: str | None = None,
        session: requests.Session | None = None,
        download_session: requests.Session | None = None,
    ) -> None:
        self.state_store = state_store
        self.execution_url = f"{process_url(api_url=api_url)}/execution"
        _https_origin(self.execution_url)
        self.session = session or _EcdsSession(self.execution_url)
        self.download_session = download_session or _EcdsSession()
        assert self.download_session is not self.session, (
            "ECDS API and downloads require separate sessions"
        )
        api_key = os.environ.get("ECDS_API_KEY") or read_cdsapi_config().get("key")
        if api_key:
            self.session.headers["PRIVATE-TOKEN"] = api_key

    def retrieve(
        self,
        payload: Mapping[str, Any],
        target: Path,
        poll_seconds: float = 30,
        maximum_polls: int = 240,
        resubmit_wait_seconds: float = RESUBMIT_WAIT_SECONDS,
        resubmit_budget_seconds: float = RESUBMIT_BUDGET_SECONDS,
        parallel_range_downloads: int = PARALLEL_RANGE_DOWNLOADS,
    ) -> Path:
        """Submit `payload` if it is not already in flight, then download to `target`.

        A blob already downloaded for `payload` is kept rather than fetched again: an
        ECDS result expires without a published SLA, so a second download of the same
        result may be impossible.

        A job ECDS fails, or no longer holds, is submitted again after
        `resubmit_wait_seconds`, doubling each time. No job is submitted once
        `resubmit_budget_seconds` have passed since this call began; the failure is
        raised as `EcdsJobFailedError` instead. A job still incomplete after
        `maximum_polls` is abandoned and replaced once; a second such job raises
        `TimeoutError`.
        """
        state = self.state_store.read_if_exists()
        if state is None or state.payload != dict(payload):
            self._submit_replacing(payload, target)
        elif _downloaded_blob_is_intact(state, target):
            log.info("Reusing the %s already downloaded for this request", target)
            return target
        elif state.status in TERMINAL_FAILURE_STATUSES:
            self._submit_replacing(payload, target)
        deadline = time.monotonic() + resubmit_budget_seconds
        wait_seconds = resubmit_wait_seconds
        replaced_after_timeout = False
        while True:
            try:
                _, result_url = self.poll_until_complete(poll_seconds, maximum_polls)
                self.download(target, result_url, parallel_range_downloads)
                return target
            except EcdsJobFailedError as e:
                if time.monotonic() + wait_seconds >= deadline:
                    raise
                log.warning("%s; submitting it again in %.0f s", e, wait_seconds)
                time.sleep(wait_seconds)
                self._submit_replacing(payload, target)
                wait_seconds *= 2
            except TimeoutError as e:
                if replaced_after_timeout:
                    raise
                replaced_after_timeout = True
                log.warning("%s; submitting it again", e)
                self._submit_replacing(payload, target)

    def _submit_replacing(
        self, payload: Mapping[str, Any], target: Path
    ) -> RequestState:
        """Submit a job for `target`, discarding any partial download of an earlier job's result."""
        _partial_path(target).unlink(missing_ok=True)
        _delete_part_files(target)
        return self.submit(payload)

    def submit(self, payload: Mapping[str, Any]) -> RequestState:
        assert self.session.headers.get("PRIVATE-TOKEN"), (
            "Set ECDS_API_KEY or a `key:` line in ~/.cdsapirc"
        )
        response = self.session.post(
            self.execution_url,
            json={"inputs": dict(payload)},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        request_id = str(body.get("jobID") or body.get("id"))
        assert request_id != "None", body
        status_url = response.headers.get("Location") or str(
            body.get("location") or f"{self.execution_url}/{request_id}"
        )
        status_url = _api_url(status_url, response.url or self.execution_url)
        status_url = _api_url(status_url, self.execution_url)
        state = RequestState(
            request_id=request_id,
            payload=dict(payload),
            submitted_at=_utc_now(),
            status_url=status_url,
        )
        self.state_store.write(state)
        log.info("Submitted ECDS job %s", request_id)
        return state

    def poll_once(self) -> tuple[RequestState, str | None]:
        state = self.state_store.read()
        status_url = _api_url(state.status_url, self.execution_url)
        response = self.session.get(status_url, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code == requests.codes.not_found:
            return self._expire(state, f"job {state.status_url}"), None
        response.raise_for_status()
        body = response.json()
        status = body.get("status") or body.get("state")
        assert isinstance(status, str), (
            f"ECDS job status response has no status: {body}"
        )
        previous_status = state.status
        state.status = status.lower()
        if state.status != previous_status:
            log.info(
                "ECDS job %s is %s %.0f s after submission",
                state.request_id,
                state.status,
                (
                    datetime.now(UTC) - datetime.fromisoformat(state.submitted_at)
                ).total_seconds(),
            )
        state.poll_failures = 0
        result_url = _result_url(body)
        results_url = _related_url(body, "results")
        response_url = _api_url(response.url or status_url, self.execution_url)
        if result_url is not None:
            result_url = urljoin(response_url, result_url)
        if results_url is not None:
            results_url = _api_url(results_url, response_url)
        if state.status in TERMINAL_FAILURE_STATUSES:
            error_body = body
            if results_url is not None:
                error_body = self.session.get(
                    results_url, timeout=REQUEST_TIMEOUT_SECONDS
                ).json()
            state.errors.append(json.dumps(error_body, sort_keys=True))
        elif (
            state.status == "successful"
            and result_url is None
            and results_url is not None
        ):
            results_response = self.session.get(
                results_url, timeout=REQUEST_TIMEOUT_SECONDS
            )
            if results_response.status_code == requests.codes.not_found:
                return self._expire(state, f"results {results_url}"), None
            results_response.raise_for_status()
            result_url = _result_url(results_response.json())
            if result_url is not None:
                result_url = urljoin(results_response.url or results_url, result_url)
        if result_url is not None:
            state.result_url = result_url
        self.state_store.write(state)
        return state, result_url

    def _expire(self, state: RequestState, what: str) -> RequestState:
        """Record that ECDS no longer holds the job or its result."""
        state.status = EXPIRED_STATUS
        state.errors.append(f"404 Not Found: {what}")
        self.state_store.write(state)
        return state

    def poll_until_complete(
        self, poll_seconds: float, maximum_polls: int
    ) -> tuple[RequestState, str]:
        """Poll until the job succeeds, giving up after `maximum_polls` or the time they span.

        Transport errors back off by the number of consecutive failures, so a blip late
        in a long poll does not sleep for hours, and the whole call stays within
        `maximum_polls * poll_seconds`.
        """
        deadline = time.monotonic() + poll_seconds * maximum_polls
        for poll_index in range(maximum_polls):
            if poll_index > 0 and time.monotonic() >= deadline:
                break
            try:
                state, result_url = self.poll_once()
            except requests.RequestException as e:
                state = self.state_store.read()
                state.poll_failures += 1
                state.errors.append(str(e))
                self.state_store.write(state)
                backoff_seconds = poll_seconds * 2 ** min(
                    state.poll_failures - 1, MAXIMUM_POLL_BACKOFF_EXPONENT
                )
                _sleep_bounded(backoff_seconds, deadline)
                continue
            if state.status in TERMINAL_FAILURE_STATUSES:
                raise EcdsJobFailedError(
                    f"ECDS job {state.request_id} ended with status {state.status}: "
                    f"{state.errors[-1] if state.errors else ''}"
                )
            if state.status == "successful" and result_url is not None:
                return state, result_url
            _sleep_bounded(poll_seconds, deadline)
        state = self.state_store.read()
        state.status = ABANDONED_STATUS
        self.state_store.write(state)
        raise TimeoutError(
            f"ECDS job {state.request_id} did not complete within {maximum_polls} "
            f"polls of {poll_seconds}s"
        )

    def download(
        self,
        target: Path,
        result_url: str | None = None,
        parallel_range_downloads: int = PARALLEL_RANGE_DOWNLOADS,
    ) -> RequestState:
        """Download the result to `target` over up to `parallel_range_downloads` concurrent Range requests.

        Parts are never resumed across calls, because a fresh probe cannot vouch for
        bytes an earlier call left.
        """
        state = self.state_store.read()
        result_url = result_url or state.result_url
        assert result_url is not None, (
            "Poll the request to completion before downloading"
        )
        _https_origin(result_url)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial_path = _partial_path(target)
        started = time.monotonic()
        try:
            transfer = self._download_ranges(
                result_url, target, parallel_range_downloads
            )
        except _ResultGoneError:
            # The signed URL is left out of the record: it is a credential.
            self._expire(state, "result download")
            raise EcdsJobFailedError(
                f"ECDS job {state.request_id} ended with status {EXPIRED_STATUS}: "
                "its result is no longer downloadable"
            ) from None
        transfer_seconds = time.monotonic() - started - transfer.assembly_seconds
        scan_started = time.monotonic()
        state.grib_messages = count_grib_messages(partial_path)
        scan_seconds = time.monotonic() - scan_started
        partial_path.replace(target)
        state.downloaded_bytes = target.stat().st_size
        state.status = "downloaded"
        self.state_store.write(state)
        log.info(
            "Downloaded %s (%d bytes, %d transferred) over %d ranges: transfer %.1f s "
            "(%.1f MB/s), assembly %.1f s, counted %d GRIB messages in %.1f s",
            target,
            state.downloaded_bytes,
            transfer.transferred_bytes,
            transfer.connections,
            transfer_seconds,
            transfer.transferred_bytes / max(transfer_seconds, 1e-6) / 1e6,
            transfer.assembly_seconds,
            state.grib_messages,
            scan_seconds,
        )
        return state

    def _get_result(
        self, result_url: str, headers: Mapping[str, str]
    ) -> requests.Response:
        try:
            response = self.download_session.get(
                result_url,
                headers=dict(headers),
                stream=True,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            )
        except requests.RequestException as e:
            raise _without_url(e) from None
        if response.status_code == requests.codes.not_found:
            response.close()
            raise _ResultGoneError
        return response

    def _download_part(
        self,
        result_url: str,
        part: _Part,
        size: int,
        etag: str,
        active: _ActiveResponses,
    ) -> None:
        response = self._get_result(
            result_url,
            {"Range": f"bytes={part.start}-{part.end}", "If-Match": etag},
        )
        active.add(response)
        try:
            if response.status_code >= 400 and response.status_code not in {
                requests.codes.precondition_failed,
                requests.codes.requested_range_not_satisfiable,
            }:
                _raise_for_download_status(response)
            if (
                response.status_code != requests.codes.partial
                or _content_range_size(response, part.start, part.end) != size
                or response.headers.get("ETag") != etag
                or not _is_identity_encoded(response)
            ):
                raise EcdsRangeMismatchError(
                    f"ECDS result range {part.start}-{part.end} was answered with "
                    f"HTTP {response.status_code} for a different range or result"
                )
            written = 0
            with part.path.open("wb") as output:
                for chunk in _iter_body(response):
                    if active.aborted.is_set():
                        return
                    written += len(chunk)
                    if written > part.length:
                        break
                    output.write(chunk)
            if written != part.length and not active.aborted.is_set():
                raise EcdsRangeMismatchError(
                    f"ECDS result range {part.start}-{part.end} returned "
                    f"{written} bytes, not {part.length}"
                )
        finally:
            active.discard(response)
            response.close()

    def _download_ranges(
        self, result_url: str, target: Path, parallel_range_downloads: int
    ) -> _Transfer:
        """Download the result as parallel byte ranges into `target`'s partial file."""
        _delete_part_files(target)
        probe = self._get_result(result_url, {"Range": "bytes=0-0"})
        probe.close()
        if probe.status_code >= 400:
            _raise_for_download_status(probe)
        size = _content_range_size(probe, 0, 0)
        etag = probe.headers.get("ETag")
        if probe.status_code != requests.codes.partial:
            _raise_unverifiable_probe(probe, "status is not 206")
        if size is None:
            _raise_unverifiable_probe(probe, "Content-Range is not exactly bytes 0-0")
        if etag is None or not _is_strong_etag(etag):
            _raise_unverifiable_probe(probe, "ETag is weak or missing")
        if not _is_identity_encoded(probe):
            _raise_unverifiable_probe(probe, "Content-Encoding is not identity")
        count = max(1, min(parallel_range_downloads, size // MINIMUM_RANGE_BYTES))
        bounds = [size * index // count for index in range(count + 1)]
        parts = [
            _Part(start, end - 1, _part_path(target, index))
            for index, (start, end) in enumerate(itertools.pairwise(bounds))
        ]
        active = _ActiveResponses()
        with ThreadPoolExecutor(len(parts)) as pool:
            futures = [
                pool.submit(self._download_part, result_url, part, size, etag, active)
                for part in parts
            ]
            done, _ = wait(futures, return_when=FIRST_EXCEPTION)
            if any(future.exception() is not None for future in done):
                for future in futures:
                    future.cancel()
                active.abort()
        # Leaving the pool joined every worker, so no part file is written after this.
        # Failures that surface during the abort can outrank the one that began it.
        failures = [
            error
            for future in sorted(futures, key=lambda future: future not in done)
            if not future.cancelled() and (error := future.exception()) is not None
        ]
        if failures:
            _delete_part_files(target)
            raise min(failures, key=_failure_priority)
        assembly_seconds = _assemble(parts, _partial_path(target))
        _delete_part_files(target)
        return _Transfer(
            connections=count,
            transferred_bytes=size,
            assembly_seconds=assembly_seconds,
        )


class _ActiveResponses:
    """The open responses of a ranged download's parts, so one failure can stop them all."""

    def __init__(self) -> None:
        self.aborted = threading.Event()
        self._responses: set[requests.Response] = set()
        self._lock = threading.Lock()

    def add(self, response: requests.Response) -> None:
        with self._lock:
            self._responses.add(response)

    def discard(self, response: requests.Response) -> None:
        with self._lock:
            self._responses.discard(response)

    def abort(self) -> None:
        self.aborted.set()
        with self._lock:
            for response in self._responses:
                response.close()


def _assemble(parts: Sequence[_Part], partial_path: Path) -> float:
    """Concatenate `parts` into `partial_path` through a fresh file, returning the seconds taken."""
    started = time.monotonic()
    assembly_path = partial_path.with_suffix(f"{partial_path.suffix}.assembly")
    with assembly_path.open("wb") as output:
        for part in parts:
            with part.path.open("rb") as source:
                shutil.copyfileobj(source, output, DOWNLOAD_CHUNK_BYTES)
    assembly_path.replace(partial_path)
    return time.monotonic() - started


def read_cdsapi_config() -> dict[str, str]:
    config_path = Path(os.environ.get("CDSAPI_RC", Path.home() / ".cdsapirc"))
    if not config_path.exists():
        return {}
    config: dict[str, str] = {}
    for line in config_path.read_text().splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"url", "key"}:
            config[key.strip()] = value.strip()
    return config


def _downloaded_blob_is_intact(state: RequestState, target: Path) -> bool:
    """Whether `target` still holds the blob `state` recorded downloading."""
    if state.status != "downloaded" or not target.exists():
        return False
    found = (target.stat().st_size, count_grib_messages(target))
    recorded = (state.downloaded_bytes, state.grib_messages)
    if found != recorded:
        log.warning(
            "%s holds %s (bytes, messages), not the %s recorded at download; retrieving it again",
            target,
            found,
            recorded,
        )
        return False
    return True


def _partial_path(target: Path) -> Path:
    return target.with_suffix(f"{target.suffix}.partial")


def _part_path(target: Path, index: int) -> Path:
    return target.with_suffix(f"{target.suffix}.partial.{index}")


def _delete_part_files(target: Path) -> None:
    prefix = f"{_partial_path(target).name}."
    if target.parent.exists():
        for path in target.parent.iterdir():
            if path.name.startswith(prefix):
                path.unlink()


def _without_url(error: requests.RequestException) -> requests.RequestException:
    """The same kind of error without its message, which can quote the signed result URL."""
    return type(error)(f"ECDS result download failed: {type(error).__name__}")


def _iter_body(response: requests.Response) -> Iterator[bytes]:
    try:
        yield from response.iter_content(DOWNLOAD_CHUNK_BYTES)
    except requests.RequestException as e:
        raise _without_url(e) from None


def _raise_for_download_status(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError:
        response.close()
        raise requests.HTTPError(
            f"ECDS result download failed: HTTP {response.status_code}",
            response=response,
        ) from None


def _raise_unverifiable_probe(probe: requests.Response, reason: str) -> NoReturn:
    raise EcdsRangeMismatchError(
        f"ECDS result probe answered HTTP {probe.status_code}, which cannot vouch "
        f"for ranges: {reason}"
    )


def _content_range_size(
    response: requests.Response, start: int, end: int
) -> int | None:
    """The total size in `response`'s Content-Range, if it is exactly `start`-`end`."""
    match = re.fullmatch(
        rf"bytes {start}-{end}/(\d+)", response.headers.get("Content-Range", "")
    )
    return int(match.group(1)) if match else None


def _is_strong_etag(etag: str) -> bool:
    return re.fullmatch(r'"[^"]*"', etag) is not None


def _is_identity_encoded(response: requests.Response) -> bool:
    return response.headers.get("Content-Encoding", "identity").lower() == "identity"


def _failure_priority(error: BaseException) -> int:
    """Rank concurrent part failures so the one that decides what happens next is raised."""
    if isinstance(error, EcdsRangeMismatchError):
        return 0
    if isinstance(error, _ResultGoneError):
        return 1
    return 2


def _sleep_bounded(seconds: float, deadline: float) -> None:
    time.sleep(max(0.0, min(seconds, deadline - time.monotonic())))


def _result_url(body: Mapping[str, Any]) -> str | None:
    for key in ("href", "location", "result_url"):
        value = body.get(key)
        if isinstance(value, str):
            return value
    for key in ("result", "asset", "value"):
        nested = body.get(key)
        if isinstance(nested, Mapping):
            result_url = _result_url(nested)
            if result_url is not None:
                return result_url
    return None


def _related_url(body: Mapping[str, Any], relation: str) -> str | None:
    links = body.get("links")
    if not isinstance(links, Sequence):
        return None
    for link in links:
        if isinstance(link, Mapping) and link.get("rel") == relation:
            href = link.get("href")
            if isinstance(href, str):
                return href
    return None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
