import json
import logging
from datetime import timedelta
from typing import cast

import httpx
import pytest
from logtail import LogtailHandler
from sentry_sdk.types import Event, Hint

from reformatters.common import betterstack, staging
from reformatters.common.betterstack import (
    REFORMATTERS_LOGGER_NAME,
    _ContextFilter,
    attach_logtail,
)
from reformatters.common.kubernetes import CronJob, ReformatCronJob


def _record() -> logging.LogRecord:
    return logging.LogRecord(
        name="reformatters.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )


def test_context_filter_sets_fields_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRON_JOB_NAME", "gfs-update")
    monkeypatch.setenv("JOB_NAME", "gfs-backfill")
    monkeypatch.setenv("POD_NAME", "gfs-worker-0")
    monkeypatch.setenv("DYNAMICAL_ENV", "prod")

    record = _record()
    assert _ContextFilter().filter(record) is True
    assert record.__dict__["cron_job_name"] == "gfs-update"
    assert record.__dict__["job_name"] == "gfs-backfill"
    assert record.__dict__["pod_name"] == "gfs-worker-0"
    assert record.__dict__["env"] == "prod"


def test_context_filter_omits_unset_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in ("CRON_JOB_NAME", "JOB_NAME", "POD_NAME", "DYNAMICAL_ENV"):
        monkeypatch.delenv(env_var, raising=False)

    record = _record()
    _ContextFilter().filter(record)
    assert not hasattr(record, "cron_job_name")


def test_attach_logtail_noop_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BETTERSTACK_SOURCE_TOKEN", raising=False)
    monkeypatch.delenv("BETTERSTACK_INGESTING_HOST", raising=False)

    logger = logging.getLogger(REFORMATTERS_LOGGER_NAME)
    before = list(logger.handlers)
    attach_logtail()
    assert logger.handlers == before


def test_attach_logtail_adds_handler_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BETTERSTACK_SOURCE_TOKEN", "test-token")
    monkeypatch.setenv("BETTERSTACK_INGESTING_HOST", "s1.example.betterstackdata.com")

    logger = logging.getLogger(REFORMATTERS_LOGGER_NAME)
    added = [h for h in logger.handlers if isinstance(h, LogtailHandler)]
    for handler in added:
        logger.removeHandler(handler)

    try:
        attach_logtail()
        attach_logtail()
        logtail_handlers = [h for h in logger.handlers if isinstance(h, LogtailHandler)]
        assert len(logtail_handlers) == 1
        assert any(isinstance(f, _ContextFilter) for f in logtail_handlers[0].filters)
    finally:
        for handler in logger.handlers[:]:
            if isinstance(handler, LogtailHandler):
                logger.removeHandler(handler)


def _exception_event(
    exc_type: str, message: str, frames: list[dict[str, object]]
) -> Event:
    return cast(
        "Event",
        {
            "exception": {
                "values": [
                    {
                        "type": exc_type,
                        "value": message,
                        "stacktrace": {"frames": frames},
                    }
                ]
            }
        },
    )


_HINT = cast("Hint", {})


def test_group_error_fingerprint_collapses_varied_messages() -> None:
    frames: list[dict[str, object]] = [
        {"module": "obstore._store", "function": "get", "in_app": False},
        {
            "module": "reformatters.common.download",
            "function": "http_download_to_disk",
            "in_app": True,
        },
    ]
    first = betterstack.group_error_fingerprint(
        _exception_event(
            "FileNotFoundError", "Object at hrrr.20161124/...t16z", frames
        ),
        _HINT,
    )
    second = betterstack.group_error_fingerprint(
        _exception_event(
            "FileNotFoundError", "Object at hrrr.20190630/...t18z", frames
        ),
        _HINT,
    )
    # Same failure mode, different interpolated path -> identical fingerprint.
    assert first["fingerprint"] == second["fingerprint"]
    # Fingerprints on the innermost in-app frame, not the noisy library frame.
    assert first["fingerprint"] == [
        "FileNotFoundError",
        "reformatters.common.download",
        "http_download_to_disk",
    ]


def test_group_error_fingerprint_separates_distinct_types() -> None:
    frames: list[dict[str, object]] = [
        {"module": "reformatters.x", "function": "f", "in_app": True}
    ]
    not_found = betterstack.group_error_fingerprint(
        _exception_event("FileNotFoundError", "a", frames), _HINT
    )
    value_error = betterstack.group_error_fingerprint(
        _exception_event("ValueError", "a", frames), _HINT
    )
    assert not_found["fingerprint"] != value_error["fingerprint"]


def test_group_error_fingerprint_leaves_non_exception_events() -> None:
    event = cast("Event", {"message": "just a log line"})
    result = betterstack.group_error_fingerprint(event, _HINT)
    assert "fingerprint" not in result


def test_cron_name_prefix_strips_step() -> None:
    assert (
        betterstack.cron_name_prefix("noaa-gfs-forecast-update", "update")
        == "noaa-gfs-forecast"
    )
    assert (
        betterstack.cron_name_prefix("noaa-gfs-forecast-validate", "validate")
        == "noaa-gfs-forecast"
    )


def test_heartbeat_key_and_name() -> None:
    assert (
        betterstack.heartbeat_key("noaa-gfs-forecast", "update", "start")
        == "noaa-gfs-forecast_update_start"
    )
    assert (
        betterstack.heartbeat_name("noaa-gfs-forecast", "update", "start")
        == "reformatters noaa-gfs-forecast update start"
    )


def test_ping_posts_to_url_and_fail_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: list[str] = []

    def fake_post(url: str, timeout: float) -> httpx.Response:
        posted.append(url)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(betterstack.httpx, "post", fake_post)
    betterstack.ping("https://hb/x")
    betterstack.ping("https://hb/x", failed=True)
    assert posted == ["https://hb/x", "https://hb/x/fail"]


def test_load_heartbeat_urls_from_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(betterstack, "load_secret", lambda name: {"k": "https://u"})
    assert betterstack.load_heartbeat_urls() == {"k": "https://u"}


_URLS = {
    "noaa-gfs-forecast_update_start": "https://hb/start",
    "noaa-gfs-forecast_update_complete": "https://hb/complete",
}


def test_monitor_heartbeat_pings_start_then_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(betterstack, "load_heartbeat_urls", lambda: _URLS)
    monkeypatch.setattr(
        betterstack, "ping", lambda url, *, failed=False: calls.append((url, failed))
    )
    with betterstack.monitor_heartbeat("noaa-gfs-forecast-update", "update"):
        pass
    assert calls == [("https://hb/start", False), ("https://hb/complete", False)]


def test_monitor_heartbeat_pings_fail_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(betterstack, "load_heartbeat_urls", lambda: _URLS)
    monkeypatch.setattr(
        betterstack, "ping", lambda url, *, failed=False: calls.append((url, failed))
    )
    with (
        pytest.raises(ValueError, match="boom"),
        betterstack.monitor_heartbeat("noaa-gfs-forecast-update", "update"),
    ):
        raise ValueError("boom")
    assert calls == [("https://hb/start", False), ("https://hb/complete", True)]


def test_monitor_heartbeat_noop_without_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(betterstack, "load_heartbeat_urls", dict)

    def _fail(url: str, *, failed: bool = False) -> None:
        raise AssertionError("should not ping when no urls are configured")

    monkeypatch.setattr(betterstack, "ping", _fail)
    with betterstack.monitor_heartbeat("x-update", "update"):
        pass


def _update_cron() -> ReformatCronJob:
    return ReformatCronJob(
        name="example-update",
        schedule="0 0 * * *",
        image="img",
        dataset_id="example",
        cpu="1",
        memory="1G",
    )


def test_monitor_cron_run_pings_update_cron(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRON_JOB_NAME", raising=False)
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        betterstack,
        "load_heartbeat_urls",
        lambda: {
            "example_update_start": "https://hb/start",
            "example_update_complete": "https://hb/complete",
        },
    )
    monkeypatch.setattr(
        betterstack, "ping", lambda url, *, failed=False: calls.append((url, failed))
    )
    with betterstack.monitor_cron_run(_update_cron(), "job-name"):
        pass
    assert calls == [("https://hb/start", False), ("https://hb/complete", False)]


def test_monitor_cron_run_noop_for_non_step_cron(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(url: str, *, failed: bool = False) -> None:
        raise AssertionError("archive crons have no heartbeat step")

    monkeypatch.setattr(betterstack, "ping", _fail)
    archive = CronJob(
        name="example-archive",
        command=["archive-grib-files"],
        schedule="0 0 * * *",
        image="img",
        dataset_id="example",
        cpu="1",
        memory="1G",
        workers_total=1,
        parallelism=1,
    )
    with betterstack.monitor_cron_run(archive, "job-name"):
        pass


def _gfs_update_cron(schedule: str, deadline_min: int) -> ReformatCronJob:
    return ReformatCronJob(
        name="noaa-gfs-forecast-update",
        schedule=schedule,
        image="img",
        dataset_id="noaa-gfs-forecast",
        cpu="1",
        memory="1G",
        pod_active_deadline=timedelta(minutes=deadline_min),
    )


class TestHeartbeatProvisioning:
    @pytest.mark.parametrize(
        ("schedule", "seconds"),
        [
            ("0 * * * *", 3600),
            ("0 */6 * * *", 6 * 3600),
            ("30 0 * * *", 24 * 3600),
            # Irregular schedule (00:00 + 01:00): the period is the largest gap (23h),
            # not the 1h short gap, so the long quiet stretch never false-alarms.
            ("0 0,1 * * *", 23 * 3600),
        ],
    )
    def test_schedule_period(self, schedule: str, seconds: int) -> None:
        assert betterstack.schedule_period(schedule).total_seconds() == seconds

    def test_heartbeat_specs_mirror_cron_config(self) -> None:
        specs = betterstack.heartbeat_specs(_gfs_update_cron("0 */6 * * *", 30))
        assert [s.key for s in specs] == [
            "noaa-gfs-forecast_update_start",
            "noaa-gfs-forecast_update_complete",
        ]
        assert all(s.period == timedelta(hours=6) for s in specs)
        # start grace = the 10 min start margin; complete grace also covers the run.
        assert specs[0].grace == timedelta(minutes=10)
        assert specs[1].grace == timedelta(minutes=10) + timedelta(minutes=30)

    def test_reconcile_creates_start_and_complete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json={"data": [], "pagination": {}})
            body = json.loads(request.content)
            created.append(body)
            hid = str(len(created))
            return httpx.Response(
                201,
                json={
                    "data": {
                        "id": hid,
                        "attributes": {**body, "url": f"https://hb/{hid}"},
                    }
                },
            )

        monkeypatch.setattr(
            betterstack,
            "_api_client",
            lambda token: httpx.Client(
                base_url=betterstack._UPTIME_API,
                transport=httpx.MockTransport(handler),
            ),
        )
        monkeypatch.setenv("BETTERSTACK_API_KEY_RW", "tok")

        url_map = betterstack.reconcile_heartbeats([_gfs_update_cron("0 * * * *", 30)])
        assert url_map == {
            "noaa-gfs-forecast_update_start": "https://hb/1",
            "noaa-gfs-forecast_update_complete": "https://hb/2",
        }
        assert created[0]["period"] == 3600
        assert created[0]["grace"] == 600
        assert created[1]["grace"] == 600 + 1800

    def test_reconcile_updates_only_on_drift(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        start_name = "reformatters noaa-gfs-forecast update start"
        complete_name = "reformatters noaa-gfs-forecast update complete"
        patched: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": "1",
                                "attributes": {
                                    "name": start_name,
                                    "period": 3600,
                                    "grace": 600,
                                    "url": "https://hb/1",
                                },
                            },
                            {
                                "id": "2",
                                "attributes": {
                                    "name": complete_name,
                                    "period": 3600,
                                    "grace": 1,  # drifted -> should be patched
                                    "url": "https://hb/2",
                                },
                            },
                        ],
                        "pagination": {},
                    },
                )
            patched.append(request.url.path)
            return httpx.Response(200, json={"data": {"id": "2", "attributes": {}}})

        monkeypatch.setattr(
            betterstack,
            "_api_client",
            lambda token: httpx.Client(
                base_url=betterstack._UPTIME_API,
                transport=httpx.MockTransport(handler),
            ),
        )
        monkeypatch.setenv("BETTERSTACK_API_KEY_RW", "tok")

        url_map = betterstack.reconcile_heartbeats([_gfs_update_cron("0 * * * *", 30)])
        assert url_map["noaa-gfs-forecast_update_complete"] == "https://hb/2"
        # Only the drifted complete heartbeat is patched; the matching start is left alone.
        assert len(patched) == 1
        assert patched[0].endswith("/heartbeats/2")

    def test_write_heartbeat_secret_merges_existing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        applied: dict[str, str] = {}
        monkeypatch.setattr(
            betterstack,
            "_load_existing_heartbeat_secret",
            lambda: {"old": "https://old"},
        )
        monkeypatch.setattr(betterstack, "_apply_heartbeat_secret", applied.update)
        betterstack.write_heartbeat_secret({"new": "https://new"})
        assert applied == {"old": "https://old", "new": "https://new"}

    def test_provision_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BETTERSTACK_API_KEY_RW", raising=False)
        with pytest.raises(RuntimeError, match="BETTERSTACK_API_KEY_RW"):
            betterstack.provision_heartbeats([_gfs_update_cron("0 * * * *", 30)])


def test_delete_staging_heartbeats(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BETTERSTACK_API_KEY_RW", "tok")
    update_prefix = betterstack.cron_name_prefix(
        staging.staging_cronjob_name("gfs", "1.2.3", "update"), "update"
    )
    start_name = betterstack.heartbeat_name(update_prefix, "update", "start")
    start_key = betterstack.heartbeat_key(update_prefix, "update", "start")

    deleted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "9", "attributes": {"name": start_name, "url": "u"}}
                    ],
                    "pagination": {},
                },
            )
        deleted.append(request.url.path)
        return httpx.Response(204, request=request)

    monkeypatch.setattr(
        betterstack,
        "_api_client",
        lambda token: httpx.Client(
            base_url=betterstack._UPTIME_API, transport=httpx.MockTransport(handler)
        ),
    )
    applied: dict[str, str] = {}
    monkeypatch.setattr(
        betterstack,
        "_load_existing_heartbeat_secret",
        lambda: {start_key: "u", "other-dataset_update_start": "keep"},
    )
    monkeypatch.setattr(betterstack, "_apply_heartbeat_secret", applied.update)

    betterstack.delete_staging_heartbeats("gfs", "1.2.3")

    assert any(path.endswith("/heartbeats/9") for path in deleted)
    # The staging key is removed from the secret; unrelated keys are kept.
    assert start_key not in applied
    assert applied["other-dataset_update_start"] == "keep"
