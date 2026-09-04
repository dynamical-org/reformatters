"""Transport for the ECMWF Data Store (ECDS) OGC-API Processes retrieval service.

ECDS has no addressable source files: a selection is submitted as a job, polled to
completion, and downloaded once from a short-lived signed URL. Signed URLs and
server-side results expire without a published SLA, so download immediately after
a job succeeds.
"""

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import requests

from reformatters.common.logging import get_logger
from reformatters.common.retry import retry

from .grib_inventory import count_grib_messages

log = get_logger(__name__)

ECDS_API_URL: Final[str] = "https://ecds.ecmwf.int/api"
S2S_FORECASTS_PROCESS: Final[str] = "s2s-forecasts"
TERMINAL_FAILURE_STATUSES: Final[frozenset[str]] = frozenset(
    {"failed", "rejected", "dismissed", "cancelled"}
)
REQUEST_TIMEOUT_SECONDS: Final[float] = 60
DOWNLOAD_TIMEOUT_SECONDS: Final[float] = 120
MAXIMUM_POLL_BACKOFF_EXPONENT: Final[int] = 6
RESUBMIT_WAIT_SECONDS: Final[float] = 60
RESUBMIT_BUDGET_SECONDS: Final[float] = 3600


class EcdsJobFailedError(Exception):
    """ECDS ran the job and ended it in a terminal failure status."""


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
    ) -> None:
        self.state_store = state_store
        self.execution_url = f"{process_url(api_url=api_url)}/execution"
        self.session = session or requests.Session()
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
    ) -> Path:
        """Submit `payload` if it is not already in flight, then download to `target`.

        A blob already downloaded for `payload` is kept rather than fetched again: an
        ECDS result expires without a published SLA, so a second download of the same
        result may be impossible.

        A job ECDS fails is submitted again after `resubmit_wait_seconds`, doubling each
        time. No job is submitted once `resubmit_budget_seconds` have passed since this
        call began; the failure is raised as `EcdsJobFailedError` instead.
        """
        state = self.state_store.read_if_exists()
        if state is None or state.payload != dict(payload):
            self.submit(payload)
        elif _downloaded_blob_is_intact(state, target):
            log.info("Reusing the %s already downloaded for this request", target)
            return target
        elif state.status in TERMINAL_FAILURE_STATUSES:
            self.submit(payload)
        deadline = time.monotonic() + resubmit_budget_seconds
        wait_seconds = resubmit_wait_seconds
        while True:
            try:
                _, result_url = self.poll_until_complete(poll_seconds, maximum_polls)
                break
            except EcdsJobFailedError as e:
                if time.monotonic() + wait_seconds >= deadline:
                    raise
                log.warning("%s; submitting it again in %.0f s", e, wait_seconds)
                time.sleep(wait_seconds)
                self.submit(payload)
                wait_seconds *= 2
        self.download(target, result_url)
        return target

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
        response = self.session.get(state.status_url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        body = response.json()
        status = body.get("status") or body.get("state")
        assert isinstance(status, str), (
            f"ECDS job status response has no status: {body}"
        )
        state.status = status.lower()
        state.poll_failures = 0
        result_url = _result_url(body)
        results_url = _related_url(body, "results")
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
            results_response.raise_for_status()
            result_url = _result_url(results_response.json())
        if result_url is not None:
            state.result_url = result_url
        self.state_store.write(state)
        return state, result_url

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
        raise TimeoutError(
            f"ECDS job did not complete within {maximum_polls} polls of {poll_seconds}s"
        )

    def download(self, target: Path, result_url: str | None = None) -> RequestState:
        state = self.state_store.read()
        result_url = result_url or state.result_url
        assert result_url is not None, (
            "Poll the request to completion before downloading"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        partial_path = target.with_suffix(f"{target.suffix}.partial")
        existing_bytes = partial_path.stat().st_size if partial_path.exists() else 0
        headers = {"Range": f"bytes={existing_bytes}-"} if existing_bytes else {}
        started = time.monotonic()
        response = self.session.get(
            result_url, headers=headers, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS
        )
        if response.status_code == requests.codes.requested_range_not_satisfiable:
            # The partial file is longer than the result or otherwise unresumable,
            # so ask for the whole body and overwrite it.
            response.close()
            response = self.session.get(
                result_url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS
            )
        response.raise_for_status()
        # A server that ignores the Range header replies 200 with the whole body,
        # which must overwrite rather than extend the partial file.
        append = existing_bytes > 0 and response.status_code == requests.codes.partial
        with partial_path.open("ab" if append else "wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
        state.grib_messages = count_grib_messages(partial_path)
        partial_path.replace(target)
        state.downloaded_bytes = target.stat().st_size
        state.status = "downloaded"
        self.state_store.write(state)
        log.info(
            "Downloaded %s (%d bytes, %d GRIB messages) in %.1fs",
            target,
            state.downloaded_bytes,
            state.grib_messages,
            time.monotonic() - started,
        )
        return state


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
