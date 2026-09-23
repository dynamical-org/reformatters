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
    MIRRORED_LEAD_HOURS,
    mirror_fire,
    mirror_pilot,
    mirror_store,
)

# sfc f00 lands ~init+51m and f18 by ~init+87m.
FIRE_MINUTE: Final = 49
SCHEDULE: Final = f"{FIRE_MINUTE} * * * *"
RUN_TIMEOUT: Final = timedelta(minutes=58)
# Room between a run's deadline and its timeout for copies in flight to finish.
FINISH_IN_FLIGHT: Final = timedelta(minutes=4)
PILOT_TIMEOUT: Final = timedelta(minutes=50)
MAX_PILOT_MINUTES: Final = 40
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


def fire_window(now: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(init to mirror, deadline) of the latest scheduled fire at or before `now`.

    A run that starts late, including one restarted after preemption, works on the
    latest fire's init: runs are serial and copies skip what is already mirrored."""
    minute = pd.Timedelta(minutes=FIRE_MINUTE)
    fire = (now - minute).floor("1h") + minute
    return fire.floor("1h"), fire + RUN_TIMEOUT - FINISH_IN_FLIGHT


@app.function(
    image=image,
    schedule=modal.Cron(SCHEDULE),
    secrets=[
        modal.Secret.from_name("noaa-hrrr-nomads-mirror-storage-options"),
        modal.Secret.from_name("sentry-reformatters"),
    ],
    region=REGION,
    cpu=1.0,
    memory=2048,
    max_containers=1,
    retries=0,
    timeout=int(RUN_TIMEOUT.total_seconds()),
)
def mirror_gribs() -> None:
    sentry_sdk.init(dsn=os.environ["DYNAMICAL_SENTRY_DSN"], environment="prod")
    init_time, deadline = fire_window(pd.Timestamp.now("UTC"))
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


# Non-preemptible so a restart cannot repeat the pilot's downloads.
@pilot_app.function(
    image=image,
    secrets=[modal.Secret.from_name("noaa-hrrr-nomads-mirror-pilot-storage-options")],
    region=REGION,
    cpu=1.0,
    memory=2048,
    max_containers=1,
    retries=0,
    nonpreemptible=True,
    timeout=int(PILOT_TIMEOUT.total_seconds()),
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


def pilot_arguments(
    init_time: str,
    bucket: str,
    file_types: str,
    leads: str,
    minutes: int,
    now: pd.Timestamp,
) -> tuple[str, str, list[str], list[int], str]:
    """`pilot_copy`'s arguments, checked before anything runs remotely. The deadline
    is fixed here so a later start cannot extend it."""
    types = [t for t in file_types.split(",") if t]
    lead_hours = [int(lead) for lead in leads.split(",") if lead]
    assert types, "choose at least one file type"
    assert lead_hours, "choose at least one lead"
    assert set(lead_hours) <= set(MIRRORED_LEAD_HOURS), "leads must be 0-18"
    assert 0 < minutes <= MAX_PILOT_MINUTES, f"minutes must be 1-{MAX_PILOT_MINUTES}"
    assert pd.Timestamp(init_time) + pd.Timedelta(hours=2) < now.tz_localize(None), (
        "pilot an init NOMADS has finished publishing"
    )
    deadline = now + timedelta(minutes=minutes)
    return init_time, bucket, types, lead_hours, deadline.isoformat()


@pilot_app.local_entrypoint()
def pilot(
    init_time: str,
    bucket: str,
    file_types: str = "sfc",
    leads: str = "0",
    minutes: int = 10,
) -> None:
    arguments = pilot_arguments(
        init_time, bucket, file_types, leads, minutes, pd.Timestamp.now("UTC")
    )
    print(json.dumps(pilot_copy.remote(*arguments), indent=2))  # noqa: T201
