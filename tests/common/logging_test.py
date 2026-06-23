import logging

import pytest

from reformatters.common.logging import _KubernetesContextFilter


def _record() -> logging.LogRecord:
    return logging.LogRecord("name", logging.INFO, "path", 1, "message", None, None)


def test_context_filter_enriches_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRON_JOB_NAME", "example-update")
    monkeypatch.setenv("JOB_NAME", "example-update-abc")
    monkeypatch.setenv("POD_NAME", "example-update-abc-0")

    record = _record()
    assert _KubernetesContextFilter().filter(record) is True
    assert getattr(record, "cron_job_name") == "example-update"  # noqa: B009
    assert getattr(record, "job_name") == "example-update-abc"  # noqa: B009
    assert getattr(record, "pod_name") == "example-update-abc-0"  # noqa: B009


def test_context_filter_omits_unset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("CRON_JOB_NAME", "JOB_NAME", "POD_NAME"):
        monkeypatch.delenv(key, raising=False)

    record = _record()
    assert _KubernetesContextFilter().filter(record) is True
    assert not hasattr(record, "cron_job_name")
    assert not hasattr(record, "job_name")
    assert not hasattr(record, "pod_name")
