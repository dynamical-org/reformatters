from datetime import timedelta

import httpx
import pytest

from reformatters.common import betterstack
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob


def _cron(cls: type, name: str, schedule: str, deadline: timedelta) -> ReformatCronJob:
    return cls(
        name=name,
        schedule=schedule,
        pod_active_deadline=deadline,
        image="image:tag",
        dataset_id="example-dataset",
        cpu="1",
        memory="1G",
    )


def test_heartbeat_key_and_name() -> None:
    assert (
        betterstack.heartbeat_key("noaa-gfs-forecast", "update", "start")
        == "noaa-gfs-forecast_update_start"
    )
    assert (
        betterstack.heartbeat_name("noaa-gfs-forecast", "validate", "complete")
        == "reformatters noaa-gfs-forecast validate complete"
    )


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        ("0 0 * * *", timedelta(days=1)),
        ("0 */6 * * *", timedelta(hours=6)),
        ("38 5,11,17,23 * * *", timedelta(hours=6)),
    ],
)
def test_schedule_period(schedule: str, expected: timedelta) -> None:
    assert betterstack.schedule_period(schedule) == expected


def test_heartbeat_specs() -> None:
    cron = _cron(
        ReformatCronJob, "example-dataset-update", "0 */6 * * *", timedelta(hours=2)
    )
    start, complete = betterstack.heartbeat_specs(cron)

    assert start.key == "example-dataset_update_start"
    assert start.name == "reformatters example-dataset update start"
    assert start.period == timedelta(hours=6)
    assert start.grace == timedelta(minutes=10)

    assert complete.key == "example-dataset_update_complete"
    assert complete.grace == timedelta(minutes=10) + timedelta(hours=2)


def test_heartbeat_specs_staging_prefix() -> None:
    # Staging crons are renamed; their heartbeats stay isolated from production's.
    cron = _cron(
        ReformatCronJob,
        "stage-example-dataset-v2-update",
        "0 */6 * * *",
        timedelta(hours=1),
    )
    start, _complete = betterstack.heartbeat_specs(cron)
    assert start.key == "stage-example-dataset-v2_update_start"
    assert start.name == "reformatters stage-example-dataset-v2 update start"


def test_ping_fail_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        calls.append(url)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(betterstack.httpx, "post", fake_post)

    betterstack.ping("https://hb.example/abc")
    betterstack.ping("https://hb.example/abc", failed=True)
    assert calls == ["https://hb.example/abc", "https://hb.example/abc/fail"]


def test_reconcile_heartbeats_create_and_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BETTERSTACK_API_KEY_RW", "test-token")

    # One heartbeat already exists with stale grace (forces a PATCH); others are created.
    existing_name = "reformatters example-dataset update start"
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "1",
                            "attributes": {
                                "name": existing_name,
                                "url": "https://hb.example/existing",
                                "period": 21600,
                                "grace": 1,
                            },
                        }
                    ],
                    "pagination": {"next": None},
                },
            )
        if request.method == "PATCH":
            return httpx.Response(200, json={"data": {"id": "1", "attributes": {}}})
        # POST create
        return httpx.Response(
            200,
            json={
                "data": {
                    "id": "new",
                    "attributes": {"url": f"https://hb.example/{len(requests)}"},
                }
            },
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        betterstack,
        "_api_client",
        lambda token: httpx.Client(
            transport=transport, base_url="https://uptime.betterstack.com/api/v2"
        ),
    )

    cron_jobs = [
        _cron(
            ReformatCronJob, "example-dataset-update", "0 */6 * * *", timedelta(hours=1)
        ),
        _cron(
            ValidationCronJob,
            "example-dataset-validate",
            "0 */6 * * *",
            timedelta(hours=1),
        ),
    ]
    url_map = betterstack.reconcile_heartbeats(cron_jobs)

    assert set(url_map) == {
        "example-dataset_update_start",
        "example-dataset_update_complete",
        "example-dataset_validate_start",
        "example-dataset_validate_complete",
    }
    # The pre-existing start heartbeat was patched (stale grace), the rest created.
    assert ("PATCH", "/api/v2/heartbeats/1") in requests
    assert sum(1 for method, _ in requests if method == "POST") == 3


def test_reconcile_skips_non_update_validate_crons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BETTERSTACK_API_KEY_RW", "test-token")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [], "pagination": {"next": None}})
        return httpx.Response(
            200,
            json={"data": {"id": "new", "attributes": {"url": "https://hb.example/x"}}},
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        betterstack,
        "_api_client",
        lambda token: httpx.Client(
            transport=transport, base_url="https://uptime.betterstack.com/api/v2"
        ),
    )

    archive = CronJob(
        command=["archive-grib-files"],
        workers_total=1,
        parallelism=1,
        name="example-dataset-archive-grib-files",
        schedule="0 0 * * *",
        image="image:tag",
        dataset_id="example-dataset",
        cpu="1",
        memory="1G",
    )
    update = _cron(
        ReformatCronJob, "example-dataset-update", "0 0 * * *", timedelta(hours=1)
    )
    url_map = betterstack.reconcile_heartbeats([archive, update])

    assert set(url_map) == {
        "example-dataset_update_start",
        "example-dataset_update_complete",
    }
