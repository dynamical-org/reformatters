"""Cross-invocation guard so a backup-cron `update` skips a source run that a
webhook-triggered `update` has already completed or is currently processing.

Both trigger paths run the same `update` command and derive the same data-driven
`run_key`, then coordinate through marker files in the object store (reusing the
StoreFactory `_internal/` coordination layer). See docs/webhooks.md.
"""

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pandas as pd
import xarray as xr

from reformatters.common.logging import get_logger
from reformatters.common.region_job import RegionJob
from reformatters.common.storage import StoreFactory
from reformatters.common.types import AppendDim

log = get_logger(__name__)

_NAMESPACE = "operational"
_LEASE_TIMEOUT = timedelta(minutes=30)


def run_key(
    all_jobs: Sequence[RegionJob],  # type: ignore[type-arg]
    template_ds: xr.Dataset,
    append_dim: AppendDim,
) -> str:
    """The newest append-dim value the jobs will write, identifying the source run.

    Webhook (~init+5h) and cron (~init+5h22m) read the same store and template, so
    they derive the same key for a given run.
    """
    # Region stops are shard-aligned and can extend past the data; clamp to the template.
    last_index = (
        min(max(job.region.stop for job in all_jobs), template_ds.sizes[append_dim]) - 1
    )
    value = template_ds[append_dim].values[last_index]
    return pd.Timestamp(value).isoformat()


def _safe(run_key_value: str) -> str:
    return run_key_value.replace(":", "").replace("+", "")


def _complete_prefix(run_key_value: str) -> str:
    return f"runs/{_safe(run_key_value)}/complete"


def _lease_key(run_key_value: str) -> str:
    return f"runs/{_safe(run_key_value)}/in_progress/marker.json"


def _lease_prefix(run_key_value: str) -> str:
    return f"runs/{_safe(run_key_value)}/in_progress"


def should_skip(
    store_factory: StoreFactory,
    run_key_value: str,
    reformat_job_name: str,
    lease_timeout: timedelta = _LEASE_TIMEOUT,
) -> bool:
    """True if `run_key_value` is already complete or in progress by another job."""
    if (
        store_factory.count_coordination_files(
            _NAMESPACE, _complete_prefix(run_key_value)
        )
        > 0
    ):
        log.info(f"Run {run_key_value} already complete; skipping.")
        return True

    leases = store_factory.read_all_coordination_files(
        _NAMESPACE, _lease_prefix(run_key_value)
    )
    if leases:
        lease = json.loads(leases[0])
        started_at = datetime.fromisoformat(lease["started_at"])
        is_other_job = lease["job_name"] != reformat_job_name
        is_fresh = datetime.now(UTC) - started_at < lease_timeout
        if is_other_job and is_fresh:
            log.info(
                f"Run {run_key_value} in progress by {lease['job_name']}; skipping."
            )
            return True
    return False


def claim(
    store_factory: StoreFactory, run_key_value: str, reformat_job_name: str
) -> None:
    store_factory.write_coordination_file(
        _NAMESPACE,
        _lease_key(run_key_value),
        json.dumps(
            {"job_name": reformat_job_name, "started_at": datetime.now(UTC).isoformat()}
        ).encode(),
    )


def release(store_factory: StoreFactory, run_key_value: str) -> None:
    store_factory.delete_coordination_file(_NAMESPACE, _lease_key(run_key_value))


def mark_complete(store_factory: StoreFactory, run_key_value: str) -> None:
    store_factory.write_coordination_file(
        _NAMESPACE,
        f"{_complete_prefix(run_key_value)}/marker.json",
        json.dumps({"finished_at": datetime.now(UTC).isoformat()}).encode(),
    )
