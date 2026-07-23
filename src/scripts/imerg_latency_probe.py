# ruff: noqa: T201
"""Measure IMERG granule availability latency on the PPS NRT server (jsimpson).

Single-purpose ops probe for setting the operational update CronJob schedules. For
each recent Early and Late granule it HEADs the jsimpson URL, reads Last-Modified,
and computes the latency between the granule's nominal end time and when it became
available. It prints p50/p95/p99 per run and a recommended schedule cadence of
~3 minutes after p95 (or p99).

Requires the `nasa-pps` cluster secret (interactive 1Password), so it can only be
run from a credentialed session:

    DYNAMICAL_ENV=prod uv run src/scripts/imerg_latency_probe.py --days 5
"""

import argparse
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from reformatters.common.config import Config, Env
from reformatters.nasa.imerg.imerg_config_models import ImergRun
from reformatters.nasa.imerg.region_job import NasaImergAnalysisSourceFileCoord
from reformatters.nasa.nasa_auth import get_pps_session


def _granule_latency_minutes(run: ImergRun, time: pd.Timestamp) -> float | None:
    coord = NasaImergAnalysisSourceFileCoord(run=run, time=time)
    url = coord.get_url("jsimpson")
    response = get_pps_session().head(url, timeout=30, allow_redirects=True)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    last_modified = response.headers.get("Last-Modified")
    if last_modified is None:
        return None
    available_at = pd.Timestamp(
        datetime.strptime(last_modified, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=UTC)
    ).tz_localize(None)
    # Latency measured from the granule's nominal end time (start + 30 min).
    nominal_end = time + pd.Timedelta("30min")
    return (available_at - nominal_end).total_seconds() / 60.0


def probe_run(run: ImergRun, days: int) -> None:
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    times = pd.date_range(
        (now - pd.Timedelta(days=days)).floor("30min"),
        now.floor("30min"),
        freq="30min",
    )
    latencies = [
        latency
        for time in times
        if (latency := _granule_latency_minutes(run, time)) is not None
    ]
    if not latencies:
        print(f"{run}: no granules found in the last {days} days")
        return

    arr = np.array(latencies)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    print(
        f"{run}: n={len(arr)} latency minutes "
        f"p50={p50:.0f} p95={p95:.0f} p99={p99:.0f} max={arr.max():.0f}"
    )
    print(
        f"  recommended update schedule: ~{p95 + 3:.0f} min (p95+3) "
        f"or ~{p99 + 3:.0f} min (p99+3) after each 30-minute granule's nominal time"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=5)
    args = parser.parse_args()

    assert Config.env == Env.prod, (
        "run with DYNAMICAL_ENV=prod to load the nasa-pps secret"
    )
    for run in ("early", "late"):
        probe_run(run, args.days)


if __name__ == "__main__":
    main()
