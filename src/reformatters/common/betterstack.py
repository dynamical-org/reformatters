from typing import Literal

import httpx

from reformatters.common.kubernetes import (
    BETTERSTACK_HEARTBEATS_SECRET_NAME,
    load_secret,
)

Step = Literal["update", "validate"]
Role = Literal["start", "complete"]

# Only update and validate crons get heartbeats; other crons (e.g. archive) are skipped.
HEARTBEAT_STEPS: tuple[Step, ...] = ("update", "validate")


def heartbeat_key(dataset_id: str, step: Step, role: Role) -> str:
    return f"{dataset_id}_{step}_{role}"


def heartbeat_name(dataset_id: str, step: Step, role: Role) -> str:
    return f"reformatters {dataset_id} {step} {role}"


def ping(url: str, *, failed: bool = False) -> None:
    response = httpx.post(f"{url}/fail" if failed else url, timeout=10)
    response.raise_for_status()


def load_heartbeat_urls() -> dict[str, str]:
    """{dataset_id}_{step}_{role} -> url. Empty outside prod."""
    return load_secret(BETTERSTACK_HEARTBEATS_SECRET_NAME)
