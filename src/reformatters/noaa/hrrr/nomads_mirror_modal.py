"""The NOMADS mirror's hourly fire as a Modal app, an alternative to the
`noaa-hrrr-nomads-mirror-gribs` CronJob. Only one of the two may write the mirror.

Deploy: `uv run modal deploy src/reformatters/noaa/hrrr/nomads_mirror_modal.py::app`.
`pilot_app` holds a bounded one-off copy into another bucket, with its own secret.
"""

import json
import os
from datetime import timedelta
from typing import Final

import modal
import pandas as pd
import sentry_sdk
import sentry_sdk.crons

from reformatters.common.monitoring import log_peak_memory
from reformatters.noaa.hrrr.nomads_mirror import (
    hourly_fire_window,
    mirror_fire,
    mirror_pilot,
    mirror_store,
)

# sfc f00 lands ~init+51m and f18 by ~init+87m.
FIRE_MINUTE: Final = 49
SCHEDULE: Final = f"{FIRE_MINUTE} * * * *"
RUN_TIMEOUT: Final = timedelta(minutes=58)
PILOT_TIMEOUT: Final = timedelta(minutes=50)
# NOMADS is in College Park, MD.
REGION: Final = "us-east"
STORAGE_OPTIONS_ENV: Final = "NOAA_HRRR_NOMADS_MIRROR_STORAGE_OPTIONS"

app = modal.App("noaa-hrrr-nomads-mirror")
pilot_app = modal.App("noaa-hrrr-nomads-mirror-pilot")
image = (
    modal.Image.debian_slim(python_version="3.14")
    .uv_sync(extra_options="--no-dev")
    .add_local_python_source("reformatters")
)


@app.function(
    image=image,
    schedule=modal.Cron(SCHEDULE),
    secrets=[
        modal.Secret.from_name("noaa-hrrr-nomads-mirror-storage-options"),
        modal.Secret.from_name("sentry-reformatters"),
    ],
    max_containers=1,
    timeout=int(RUN_TIMEOUT.total_seconds()),
    region=REGION,
    cpu=1.0,
    memory=2048,
    retries=0,
)
def mirror_gribs() -> None:
    sentry_sdk.init(dsn=os.environ["DYNAMICAL_SENTRY_DSN"], environment="prod")
    init_time, deadline = hourly_fire_window(
        pd.Timestamp.now("UTC"), FIRE_MINUTE, RUN_TIMEOUT
    )
    with sentry_sdk.crons.monitor(
        monitor_slug="noaa-hrrr-nomads-mirror-modal",
        monitor_config={
            "schedule": {"type": "crontab", "value": SCHEDULE},
            "timezone": "UTC",
            "checkin_margin": 10,
            "max_runtime": int(RUN_TIMEOUT.total_seconds() // 60),
        },
    ):
        mirror_fire(
            init_time,
            deadline,
            mirror_store(json.loads(os.environ[STORAGE_OPTIONS_ENV])),
        )
        log_peak_memory()


@pilot_app.function(
    image=image,
    secrets=[modal.Secret.from_name("noaa-hrrr-nomads-mirror-pilot-storage-options")],
    timeout=int(PILOT_TIMEOUT.total_seconds()),
    region=REGION,
    cpu=1.0,
    memory=2048,
    retries=0,
)
def pilot_copy(
    init_time: str,
    bucket: str,
    file_types: list[str],
    lead_hours: list[int],
    deadline: str,
) -> dict[str, list[str]]:
    result = mirror_pilot(
        pd.Timestamp(init_time),
        json.loads(os.environ[STORAGE_OPTIONS_ENV]),
        bucket,
        file_types,
        lead_hours,
        deadline=pd.Timestamp(deadline),
    )
    log_peak_memory()
    return {"copied": result.copied, "pending": result.pending}


@pilot_app.local_entrypoint()
def pilot(
    init_time: str,
    bucket: str,
    file_types: str = "sfc",
    leads: str = "0",
    minutes: int = 10,
) -> None:
    # Fixed here, so a run restarted after preemption keeps the original deadline.
    deadline = pd.Timestamp.now("UTC") + timedelta(minutes=minutes)
    assert timedelta(minutes=minutes) < PILOT_TIMEOUT - timedelta(minutes=5)
    result = pilot_copy.remote(
        init_time,
        bucket,
        file_types.split(","),
        [int(lead) for lead in leads.split(",")],
        deadline.isoformat(),
    )
    print(json.dumps(result, indent=2))  # noqa: T201
