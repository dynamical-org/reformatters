import json
import threading
from collections.abc import Callable
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
        self.listed_during_submission = threading.Event()
        self.list_count = 0
        self.before_list: Callable[[], None] = lambda: None
        self.finish_on_second_list: set[str] = set()

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
            self.submission_started.set()
            if self.hold_submission:
                assert self.release_submission.wait(timeout=5)
            job_id = f"new-{len(self.jobs)}"
            self.add(job_id, json.loads(request.body)["inputs"])
            body = {"jobID": job_id}
        elif request.method == "DELETE":
            status = self.delete_status
            if status == 200 and self.confirm_delete:
                self.jobs[job_id]["status"] = "dismissed"
            body = self.jobs[job_id]
        elif job_id == "jobs":
            self.before_list()
            self.list_count += 1
            if self.list_count == 2:
                for finished in self.finish_on_second_list:
                    self.jobs[finished]["status"] = "successful"
            status = self.list_status
            if (
                self.submission_started.is_set()
                and not self.release_submission.is_set()
            ):
                self.listed_during_submission.set()
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


def test_waiting_for_a_younger_selection_times_out_without_cancelling_it(
    tmp_path: Path, lifecycle_clock: FakeClock
) -> None:
    api = QueueApi()
    api.add("other", {"variable": ["temperature"]})

    with pytest.raises(TimeoutError):
        client(tmp_path / "state.json", api).retrieve(
            PAYLOAD, tmp_path / "result.grib2", poll_seconds=1, maximum_polls=2
        )

    assert not any(method in {"POST", "DELETE"} for method, _ in api.sent)
    assert api.jobs["other"]["status"] == "running"
    assert lifecycle_clock.now == 2


def test_concurrent_clients_only_submit_one_outstanding_selection(
    tmp_path: Path, lifecycle_clock: FakeClock
) -> None:
    api = QueueApi()
    api.hold_submission = True

    def submit(index: int) -> str:
        try:
            client(tmp_path / f"state-{index}.json", api).retrieve(
                {"variable": [f"variable-{index}"]},
                tmp_path / f"result-{index}.grib2",
                poll_seconds=1,
                maximum_polls=2,
            )
        except TimeoutError:
            return "blocked"
        return "submitted"

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(submit, 0)
        assert api.submission_started.wait(timeout=5)
        second = pool.submit(submit, 1)
        api.listed_during_submission.wait(timeout=1)
        api.release_submission.set()
        results = [first.result(), second.result()]

    assert results == ["blocked", "blocked"]
    assert len(api.jobs) == 1


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


def test_overage_request_is_cancelled_before_its_replacement_is_submitted(
    tmp_path: Path, lifecycle_clock: FakeClock
) -> None:
    api = QueueApi()
    request = client(tmp_path / "state.json", api)
    request.submit(PAYLOAD)
    lifecycle_clock.now = 86_400
    api.complete_on_poll.add("new-1")
    target = tmp_path / "result.grib2"

    request.retrieve(
        PAYLOAD,
        target,
        poll_seconds=1,
        maximum_polls=2,
        maximum_request_age_seconds=86_400,
    )

    writes = [method for method, _ in api.sent if method in {"POST", "DELETE"}]
    assert writes == ["POST", "DELETE", "POST"]
    assert api.jobs["new-0"]["status"] == "dismissed"
    assert target.read_bytes() == grib_message()


@pytest.mark.parametrize(("delete_status", "confirmed"), [(503, True), (200, False)])
def test_failed_or_unconfirmed_cancellation_never_submits_a_replacement(
    tmp_path: Path, lifecycle_clock: FakeClock, delete_status: int, confirmed: bool
) -> None:
    api = QueueApi()
    request = client(tmp_path / "state.json", api)
    request.submit(PAYLOAD)
    lifecycle_clock.now = 86_400
    api.delete_status = delete_status
    api.confirm_delete = confirmed

    with pytest.raises((requests.HTTPError, TimeoutError)):
        request.retrieve(
            PAYLOAD,
            tmp_path / "result.grib2",
            poll_seconds=1,
            maximum_polls=2,
            maximum_request_age_seconds=86_400,
        )

    assert len(api.jobs) == 1
    assert api.jobs["new-0"]["status"] == "running"


def test_foreign_pagination_link_is_rejected_before_sending_credentials(
    tmp_path: Path, lifecycle_clock: FakeClock
) -> None:
    api = QueueApi()
    api.next_page = "https://foreign.test/jobs"

    with pytest.raises(ValueError, match="configured HTTPS origin"):
        client(tmp_path / "state.json", api).submit(PAYLOAD)

    assert all(url.startswith(API) for _, url in api.sent)


def test_overage_replacements_do_not_loop_forever(
    tmp_path: Path, lifecycle_clock: FakeClock
) -> None:
    api = QueueApi()
    request = client(tmp_path / "state.json", api)

    with pytest.raises(TimeoutError):
        request.retrieve(
            PAYLOAD,
            tmp_path / "result.grib2",
            poll_seconds=1,
            maximum_polls=2,
            maximum_request_age_seconds=2,
        )

    assert len(api.jobs) == 2
    assert all(job["status"] == "dismissed" for job in api.jobs.values())


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


def test_identical_request_from_another_producer_is_adopted_then_cancelled_only_when_overage(
    tmp_path: Path, lifecycle_clock: FakeClock
) -> None:
    api = QueueApi()
    api.add("another-producer", PAYLOAD)
    state_path = tmp_path / "state.json"
    target = tmp_path / "result.grib2"

    with pytest.raises(TimeoutError):
        client(state_path, api).retrieve(
            PAYLOAD, target, poll_seconds=1, maximum_polls=2
        )
    assert not any(method in {"POST", "DELETE"} for method, _ in api.sent)

    lifecycle_clock.now = 86_400
    api.complete_on_poll.add("new-1")
    client(state_path, api).retrieve(PAYLOAD, target, poll_seconds=1, maximum_polls=2)

    assert api.jobs["another-producer"]["status"] == "dismissed"
    assert [method for method, _ in api.sent if method in {"POST", "DELETE"}] == [
        "DELETE",
        "POST",
    ]
    assert target.read_bytes() == grib_message()


def test_other_selection_finishes_while_waiting_then_requested_selection_downloads(
    tmp_path: Path, lifecycle_clock: FakeClock
) -> None:
    api = QueueApi()
    api.add("other", {"variable": ["temperature"]})
    api.finish_on_second_list.add("other")
    api.complete_on_poll.add("new-1")
    target = tmp_path / "result.grib2"

    client(tmp_path / "state.json", api).retrieve(
        PAYLOAD, target, poll_seconds=1, maximum_polls=2
    )

    assert lifecycle_clock.sleeps == [1]
    assert not any(method == "DELETE" for method, _ in api.sent)
    assert target.read_bytes() == grib_message()


def test_overage_other_selection_is_cancelled_before_requested_selection_is_submitted(
    tmp_path: Path, lifecycle_clock: FakeClock
) -> None:
    api = QueueApi()
    api.add("other", {"variable": ["temperature"]})
    lifecycle_clock.now = 20
    api.complete_on_poll.add("new-1")
    target = tmp_path / "result.grib2"

    client(tmp_path / "state.json", api).retrieve(
        PAYLOAD, target, poll_seconds=1, maximum_polls=2, maximum_request_age_seconds=10
    )

    assert api.jobs["other"]["status"] == "dismissed"
    assert [method for method, _ in api.sent if method in {"POST", "DELETE"}] == [
        "DELETE",
        "POST",
    ]
    assert target.read_bytes() == grib_message()


@pytest.mark.parametrize("list_delay_seconds", [0, 2])
def test_failed_job_retry_never_submits_after_budget_expires_in_account_queue(
    tmp_path: Path, lifecycle_clock: FakeClock, list_delay_seconds: int
) -> None:
    api = QueueApi()
    api.add("failed", PAYLOAD)
    api.jobs["failed"]["status"] = "failed"
    api.before_list = lambda: lifecycle_clock.sleep(list_delay_seconds)
    if not list_delay_seconds:
        api.add("other", {"variable": ["temperature"]})
        api.finish_on_second_list.add("other")
    request = client(tmp_path / "state.json", api)
    request.state_store.write(
        ecds_client.RequestState(
            request_id="failed",
            payload=PAYLOAD,
            submitted_at=START.isoformat(),
            status_url=f"{JOBS}/failed",
            status="running",
        )
    )
    api.complete_on_poll.add(f"new-{len(api.jobs)}")

    with pytest.raises(ecds_client.EcdsJobFailedError):
        request.retrieve(
            PAYLOAD,
            tmp_path / "result.grib2",
            poll_seconds=2,
            maximum_polls=3,
            resubmit_wait_seconds=0,
            resubmit_budget_seconds=1,
        )

    assert not any(method in {"POST", "DELETE"} for method, _ in api.sent)
    assert lifecycle_clock.now == max(1, list_delay_seconds)


def test_initial_submission_waits_for_queue_without_using_failed_job_retry_budget(
    tmp_path: Path, lifecycle_clock: FakeClock
) -> None:
    api = QueueApi()
    api.add("other", {"variable": ["temperature"]})
    api.finish_on_second_list.add("other")
    api.complete_on_poll.add("new-1")
    target = tmp_path / "result.grib2"

    client(tmp_path / "state.json", api).retrieve(
        PAYLOAD,
        target,
        poll_seconds=2,
        maximum_polls=3,
        resubmit_wait_seconds=0,
        resubmit_budget_seconds=1,
    )

    assert lifecycle_clock.now == 2
    assert target.read_bytes() == grib_message()
