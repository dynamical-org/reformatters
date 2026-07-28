import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import requests

from reformatters.common.logging import get_logger

log = get_logger(__name__)

DEFAULT_API_URL = (
    "https://ecds.ecmwf.int/api/retrieve/v1/processes/s2s-forecasts/execution"
)
TERMINAL_FAILURES = {"failed", "dismissed", "cancelled"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class RequestMeasurement:
    request_id: str
    payload: dict[str, Any]
    submitted_at: str
    status_url: str
    status: str = "submitted"
    completed_at: str | None = None
    result_url: str | None = None
    queue_seconds: float | None = None
    server_processing_seconds: float | None = None
    download_seconds: float | None = None
    downloaded_bytes: int | None = None
    grib_messages: int | None = None
    expected_members: list[int] = field(default_factory=list)
    received_members: list[int] = field(default_factory=list)
    expected_steps: list[int] = field(default_factory=list)
    received_steps: list[int] = field(default_factory=list)
    expected_variables: list[str] = field(default_factory=list)
    received_variables: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    retry_count: int = 0
    retry_idempotent: bool | None = None
    interrupted_download_resumable: bool | None = None
    completed_result_reusable: bool | None = None


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> RequestMeasurement:
        return RequestMeasurement(**json.loads(self.path.read_text()))

    def write(self, measurement: RequestMeasurement) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(asdict(measurement), indent=2, sort_keys=True)
        )
        temporary_path.replace(self.path)


class EcdsRequest:
    def __init__(
        self,
        state_store: StateStore,
        api_url: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.state_store = state_store
        self.api_url = _execution_url(
            api_url or os.environ.get("ECDS_API_ENDPOINT", DEFAULT_API_URL)
        )
        self.session = session or requests.Session()
        api_key = os.environ.get("ECDS_API_KEY")
        if api_key and ":" in api_key:
            self.session.auth = tuple(api_key.split(":", maxsplit=1))
        elif api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def submit(self, payload: dict[str, Any]) -> RequestMeasurement:
        assert os.environ.get("ECDS_API_KEY") or self.session.auth, (
            "Set ECDS_API_KEY before submitting a request"
        )
        response = self.session.post(self.api_url, json={"inputs": payload}, timeout=60)
        response.raise_for_status()
        response_body = response.json()
        request_id = str(response_body.get("jobID") or response_body.get("id"))
        assert request_id != "None", response_body
        status_url = response.headers.get("Location") or str(
            response_body.get("location") or f"{self.api_url}/{request_id}"
        )
        measurement = RequestMeasurement(
            request_id=request_id,
            payload=payload,
            submitted_at=utc_now(),
            status_url=status_url,
            expected_members=[int(value) for value in payload.get("number", [])],
            expected_steps=[int(value) for value in payload.get("step", [])],
            expected_variables=_as_strings(payload.get("param", [])),
        )
        self.state_store.write(measurement)
        return measurement

    def poll_once(self) -> tuple[RequestMeasurement, str | None]:
        measurement = self.state_store.read()
        response = self.session.get(measurement.status_url, timeout=60)
        response.raise_for_status()
        response_body = response.json()
        measurement.status = str(
            response_body.get("status") or response_body.get("state")
        ).lower()
        if measurement.status in TERMINAL_FAILURES:
            measurement.errors.append(json.dumps(response_body, sort_keys=True))
        result_url = _result_url(response_body)
        if result_url is not None:
            measurement.completed_at = utc_now()
            measurement.result_url = result_url
        self.state_store.write(measurement)
        return measurement, result_url

    def poll_until_complete(
        self,
        poll_seconds: float,
        maximum_polls: int,
    ) -> tuple[RequestMeasurement, str]:
        for poll_index in range(maximum_polls):
            try:
                measurement, result_url = self.poll_once()
            except requests.RequestException as error:
                measurement = self.state_store.read()
                measurement.retry_count += 1
                measurement.errors.append(str(error))
                self.state_store.write(measurement)
                time.sleep(poll_seconds * 2 ** min(poll_index, 6))
                continue
            if measurement.status in TERMINAL_FAILURES:
                raise RuntimeError(f"ECDS request ended with {measurement.status}")
            if result_url is not None:
                return measurement, result_url
            time.sleep(poll_seconds)
        raise TimeoutError(f"Request did not complete after {maximum_polls} polls")

    def download(
        self, target: Path, result_url: str | None = None
    ) -> RequestMeasurement:
        measurement = self.state_store.read()
        result_url = result_url or measurement.result_url
        assert result_url is not None, (
            "Poll the request to completion before downloading"
        )
        partial_path = target.with_suffix(f"{target.suffix}.partial")
        existing_bytes = partial_path.stat().st_size if partial_path.exists() else 0
        headers = {"Range": f"bytes={existing_bytes}-"} if existing_bytes else {}
        started = time.monotonic()
        response = self.session.get(
            result_url, headers=headers, stream=True, timeout=120
        )
        response.raise_for_status()
        append = existing_bytes > 0 and response.status_code == requests.codes.partial
        with partial_path.open("ab" if append else "wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
        _validate_grib_container(partial_path)
        partial_path.replace(target)
        target.with_suffix(f"{target.suffix}.complete").write_text(utc_now())
        measurement.download_seconds = time.monotonic() - started
        measurement.downloaded_bytes = target.stat().st_size
        measurement.grib_messages = _count_grib_messages(target)
        measurement.status = "downloaded"
        measurement.interrupted_download_resumable = append
        self.state_store.write(measurement)
        return measurement


def _as_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return []


def _execution_url(api_endpoint: str) -> str:
    if api_endpoint.rstrip("/").endswith("/execution"):
        return api_endpoint.rstrip("/")
    return f"{api_endpoint.rstrip('/')}/retrieve/v1/processes/s2s-forecasts/execution"


def _result_url(response_body: Mapping[str, Any]) -> str | None:
    for key in ("href", "location", "result_url"):
        value = response_body.get(key)
        if isinstance(value, str):
            return value
    result = response_body.get("result")
    if isinstance(result, Mapping):
        return _result_url(result)
    return None


def _count_grib_messages(path: Path) -> int:
    return path.read_bytes().count(b"GRIB")


def _validate_grib_container(path: Path) -> None:
    contents = path.read_bytes()
    assert contents.startswith(b"GRIB"), "Download does not start with a GRIB message"
    assert contents.endswith(b"7777"), "Download ends inside a GRIB message"
    assert _count_grib_messages(path) > 0, "Download contains no GRIB messages"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--payload", type=Path, required=True)
    poll = subparsers.add_parser("poll")
    poll.add_argument("--poll-seconds", type=float, default=60)
    poll.add_argument("--maximum-polls", type=int, default=1)
    download = subparsers.add_parser("download")
    download.add_argument("--result-url")
    download.add_argument("--target", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    request = EcdsRequest(StateStore(arguments.state), api_url=arguments.api_url)
    command: Literal["submit", "poll", "download"] = arguments.command
    match command:
        case "submit":
            request.submit(json.loads(arguments.payload.read_text()))
        case "poll":
            if arguments.maximum_polls == 1:
                request.poll_once()
            else:
                request.poll_until_complete(
                    arguments.poll_seconds, arguments.maximum_polls
                )
        case "download":
            request.download(arguments.target, arguments.result_url)


if __name__ == "__main__":
    main()
