import logging

import pytest
from logtail import LogtailHandler

from reformatters.common.betterstack import (
    REFORMATTERS_LOGGER_NAME,
    _ContextFilter,
    attach_logtail,
)


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
