import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
from requests.adapters import BaseAdapter

from reformatters.ecmwf.archive_gribs import ecds_client
from reformatters.ecmwf.archive_gribs.ecds_client import EcdsRequest, StateStore

from .ecds_client_test import (
    FakeClock,
    range_session,
    wire_response,
)
from .grib_inventory_test import grib_message

API = "https://api.test/api"
JOBS = f"{API}/retrieve/v1/jobs"
PAYLOAD = {"variable": ["total_precipitation"]}
START = datetime(2026, 1, 1, tzinfo=UTC)


class QueueApi(BaseAdapter):
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.sent: list[tuple[str, str]] = []
        self.complete_on_poll: set[str] = set()
        self.delete_status = 200
        self.list_status = 200
        self.confirm_delete = True
        self.next_page: str | None = None
        self.hold_submission = False
        self.submission_started = threading.Event()
        self.release_submission = threading.Event()

    def add(
        self,
        job_id: str,
        payload: dict[str, Any],
        *,
        process: str = "s2s-forecasts",
    ) -> None:
        self.jobs[job_id] = {
            "jobID": job_id,
            "processID": process,
            "status": "running",
            "created": START.isoformat(),
            "metadata": {"request": {"ids": payload}},
        }

    def send(
        self,
        request: requests.PreparedRequest,
        *_args: object,
        **_kwargs: object,
    ) -> requests.Response:
        assert request.url is not None
        assert request.method is not None
        self.sent.append((request.method, request.url))
        parsed = urlsplit(request.url)
        job_id = parsed.path.rsplit("/", 1)[-1]
        status = 200
        if request.method == "POST":
            assert len(self.jobs) < 10, "Retrieval keeps replacing expired requests"
            assert parsed.path.endswith("/execution")
            assert isinstance(request.body, (str, bytes))
            job_id = f"new-{len(self.jobs)}"
            self.add(job_id, json.loads(request.body)["inputs"])
            body = {"jobID": job_id}
            if self.hold_submission and not self.submission_started.is_set():
                self.submission_started.set()
                assert self.release_submission.wait(timeout=5)
        elif request.method == "DELETE":
            status = self.delete_status
            if status == 200 and self.confirm_delete:
                self.jobs[job_id]["status"] = "dismissed"
            body = self.jobs[job_id]
        elif job_id == "jobs":
            status = self.list_status
            jobs = [
                job
                for job in self.jobs.values()
                if job["status"] in {"accepted", "running"}
            ]
            body = {"jobs": jobs, "links": []}
            if self.next_page and "page" not in parse_qs(parsed.query):
                body = {
                    "jobs": [],
                    "links": [{"rel": "next", "href": self.next_page}],
                }
        else:
            body = self.jobs[job_id].copy()
            if job_id in self.complete_on_poll and "request" not in parse_qs(
                parsed.query
            ):
                body["status"] = "successful"
                self.jobs[job_id]["status"] = "successful"
                body["asset"] = {"value": {"href": "https://download.test/blob"}}
        result = wire_response(body, status_code=status)
        result.url = request.url
        result.request = request
        if request.method == "POST":
            result.headers["Location"] = f"{JOBS}/{job_id}"
        return result

    def close(self) -> None:
        pass


@pytest.fixture
def lifecycle_clock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeClock:
    monkeypatch.setenv("ECDS_API_KEY", "fake-key")
    monkeypatch.setenv("CDSAPI_RC", str(tmp_path / "absent"))
    clock = FakeClock()
    monkeypatch.setattr(ecds_client, "time", clock)

    def utc_now() -> str:
        return (START + timedelta(seconds=clock.now)).isoformat()

    monkeypatch.setattr(ecds_client, "_utc_now", utc_now)
    return clock


def client(path: Path, api: QueueApi) -> EcdsRequest:
    session = requests.Session()
    session.trust_env = False
    session.mount("https://", api)
    return EcdsRequest(
        StateStore(path),
        api_url=API,
        session=session,
        download_session=range_session(grib_message()),
    )


def test_restart_without_local_state_adopts_a_matching_job_on_a_later_page(
    tmp_path: Path, lifecycle_clock: FakeClock
) -> None:
    api = QueueApi()
    api.add("existing", PAYLOAD)
    api.complete_on_poll.add("existing")
    api.next_page = f"{JOBS}?page=2"
    target = tmp_path / "result.grib2"

    client(tmp_path / "state.json", api).retrieve(PAYLOAD, target, poll_seconds=0)

    assert target.read_bytes() == grib_message()
    assert not any(method in {"POST", "DELETE"} for method, _ in api.sent)


def test_adoption_ignores_value_order_and_scalar_or_list_form(
    tmp_path: Path, lifecycle_clock: FakeClock
) -> None:
    payload = {"variable": ["a", "b"], "data_format": "grib"}
    api = QueueApi()
    api.add("existing", {"variable": ["b", "a"], "data_format": ["grib"]})
    state_path = tmp_path / "state.json"

    with pytest.raises(TimeoutError):
        client(state_path, api).retrieve(
            payload, tmp_path / "result.grib2", poll_seconds=1, maximum_polls=2
        )
    api.complete_on_poll.add("existing")
    client(state_path, api).retrieve(payload, tmp_path / "result.grib2", poll_seconds=1)

    assert not any(method in {"POST", "DELETE"} for method, _ in api.sent)
    assert StateStore(state_path).read().payload == payload


def test_a_different_outstanding_selection_is_neither_adopted_nor_waited_on(
    tmp_path: Path, lifecycle_clock: FakeClock
) -> None:
    api = QueueApi()
    api.add("other", {"variable": ["temperature"]})

    state = client(tmp_path / "state.json", api).submit(PAYLOAD)

    assert state.request_id == "new-1"
    assert lifecycle_clock.now == 0
    assert api.jobs["other"]["status"] == "running"
    assert not any(method == "DELETE" for method, _ in api.sent)


def test_a_sibling_selections_in_flight_job_is_not_adopted(
    tmp_path: Path, lifecycle_clock: FakeClock
) -> None:
    api = QueueApi()
    api.hold_submission = True
    payloads = [{"variable": [f"variable-{index}"]} for index in range(2)]
    stores = [StateStore(tmp_path / f"state-{index}.json") for index in range(2)]

    def submit(index: int) -> None:
        client(stores[index].path, api).submit(payloads[index])

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(submit, 0)
        assert api.submission_started.wait(timeout=5)
        pool.submit(submit, 1).result(timeout=5)
        api.release_submission.set()
        first.result()

    request_ids = [store.read().request_id for store in stores]
    assert len(set(request_ids)) == 2
    for request_id, payload in zip(request_ids, payloads, strict=True):
        assert api.jobs[request_id]["metadata"]["request"]["ids"] == payload


@pytest.mark.parametrize("legacy_status", ["running", "abandoned"])
def test_timeout_and_restart_keep_polling_the_same_request(
    tmp_path: Path, lifecycle_clock: FakeClock, legacy_status: str
) -> None:
    api = QueueApi()
    request = client(tmp_path / "state.json", api)
    request.submit(PAYLOAD)
    state = request.state_store.read()
    state.status = legacy_status
    request.state_store.write(state)

    with pytest.raises(TimeoutError):
        request.retrieve(
            PAYLOAD, tmp_path / "result.grib2", poll_seconds=1, maximum_polls=2
        )
    api.complete_on_poll.add(state.request_id)
    client(tmp_path / "state.json", api).retrieve(
        PAYLOAD, tmp_path / "result.grib2", poll_seconds=1
    )

    assert (tmp_path / "result.grib2").read_bytes() == grib_message()
    assert len(api.jobs) == 1
    assert not any(method == "DELETE" for method, _ in api.sent)


def test_identical_job_is_adopted_until_overage_then_cancelled_and_replaced(
    tmp_path: Path, lifecycle_clock: FakeClock
) -> None:
    api = QueueApi()
    api.add("another-producer", PAYLOAD)
    target = tmp_path / "result.grib2"

    with pytest.raises(TimeoutError):
        client(tmp_path / "first-pod.json", api).retrieve(
            PAYLOAD, target, poll_seconds=1, maximum_polls=2
        )
    assert not any(method in {"POST", "DELETE"} for method, _ in api.sent)

    lifecycle_clock.now = ecds_client.MAXIMUM_REQUEST_AGE_SECONDS
    api.complete_on_poll.add("new-1")
    client(tmp_path / "next-pod.json", api).retrieve(
        PAYLOAD, target, poll_seconds=1, maximum_polls=2
    )

    assert api.jobs["another-producer"]["status"] == "dismissed"
    assert [method for method, _ in api.sent if method in {"POST", "DELETE"}] == [
        "DELETE",
        "POST",
    ]
    assert target.read_bytes() == grib_message()


@pytest.mark.parametrize(("delete_status", "confirmed"), [(503, True), (200, False)])
def test_failed_or_unconfirmed_cancellation_never_submits_a_replacement(
    tmp_path: Path, lifecycle_clock: FakeClock, delete_status: int, confirmed: bool
) -> None:
    api = QueueApi()
    api.add("stuck", PAYLOAD)
    lifecycle_clock.now = ecds_client.MAXIMUM_REQUEST_AGE_SECONDS
    api.delete_status = delete_status
    api.confirm_delete = confirmed

    with pytest.raises((requests.HTTPError, TimeoutError)):
        client(tmp_path / "state.json", api).submit(PAYLOAD)

    assert list(api.jobs) == ["stuck"]
    assert api.jobs["stuck"]["status"] == "running"


def test_foreign_pagination_link_is_rejected_before_sending_credentials(
    tmp_path: Path, lifecycle_clock: FakeClock
) -> None:
    api = QueueApi()
    api.next_page = "https://foreign.test/jobs"

    with pytest.raises(ValueError, match="configured HTTPS origin"):
        client(tmp_path / "state.json", api).submit(PAYLOAD)

    assert all(url.startswith(API) for _, url in api.sent)


def test_unreadable_account_queue_does_not_submit(
    tmp_path: Path, lifecycle_clock: FakeClock
) -> None:
    api = QueueApi()
    api.list_status = 503

    with pytest.raises(requests.HTTPError):
        client(tmp_path / "state.json", api).submit(PAYLOAD)

    assert not api.jobs


def test_non_s2s_job_is_neither_adopted_nor_cancelled(
    tmp_path: Path, lifecycle_clock: FakeClock
) -> None:
    api = QueueApi()
    api.add("different-process", PAYLOAD, process="another-process")

    state = client(tmp_path / "state.json", api).submit(PAYLOAD)

    assert state.request_id != "different-process"
    assert api.jobs["different-process"]["status"] == "running"
    assert not any(method == "DELETE" for method, _ in api.sent)


@pytest.mark.parametrize(
    "created",
    ["2026-01-01T00:00:00", "2026-01-01T00:00:00Z", "2025-12-31T19:00:00-05:00"],
)
@pytest.mark.parametrize("persisted", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_server_timestamp_forms_allow_timeout_and_completion_after_restart(
    tmp_path: Path,
    lifecycle_clock: FakeClock,
    created: str,
    persisted: bool,
    complete: bool,
) -> None:
    api = QueueApi()
    api.add("existing", PAYLOAD)
    api.jobs["existing"]["created"] = created
    state_path = tmp_path / "state.json"
    if persisted:
        state_path.write_text(
            json.dumps(
                {
                    "request_id": "existing",
                    "payload": PAYLOAD,
                    "submitted_at": created,
                    "status_url": f"{JOBS}/existing",
                    "status": "running",
                }
            )
        )
    if complete:
        api.complete_on_poll.add("existing")
    request = client(state_path, api)
    target = tmp_path / "result.grib2"

    if complete:
        request.retrieve(PAYLOAD, target, poll_seconds=1, maximum_polls=2)
        assert target.read_bytes() == grib_message()
    else:
        with pytest.raises(TimeoutError):
            request.retrieve(PAYLOAD, target, poll_seconds=1, maximum_polls=2)
    assert not any(method in {"POST", "DELETE"} for method, _ in api.sent)
