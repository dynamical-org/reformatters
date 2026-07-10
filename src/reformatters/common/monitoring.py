import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import sentry_sdk
import sentry_sdk.crons

from reformatters.common.config import Config
from reformatters.common.iterating import digest
from reformatters.common.kubernetes import CronJob


@contextmanager
def monitor_cron(
    cron_job: CronJob,
    reformat_job_name: str,
    *,
    send_in_progress: bool = True,
    send_result: bool = True,
) -> Iterator[None]:
    """Send Sentry cron check-ins for `cron_job` around the wrapped block, so a
    missed or overrunning run alerts, not just a raised exception."""
    if not Config.is_sentry_enabled:
        yield
        return

    # Use the actual cronjob name from k8s env when available. This ensures
    # staging cronjobs report to their own Sentry monitor, not production's.
    monitor_slug = os.getenv("CRON_JOB_NAME") or cron_job.name

    def capture_checkin(status: Literal["ok", "in_progress", "error"]) -> None:
        sentry_sdk.crons.capture_checkin(
            monitor_slug=monitor_slug,
            check_in_id=digest([reformat_job_name], length=32),
            status=status,
            monitor_config={
                "schedule": {"type": "crontab", "value": cron_job.schedule},
                "timezone": "UTC",
                "checkin_margin": 10,
                "max_runtime": int(cron_job.pod_active_deadline.total_seconds() / 60),
                "failure_issue_threshold": 1,
                "recovery_threshold": 1,
            },
        )

    if send_in_progress:
        capture_checkin("in_progress")
    try:
        yield
    except Exception:
        if send_result:
            capture_checkin("error")
        raise
    else:
        if send_result:
            capture_checkin("ok")
