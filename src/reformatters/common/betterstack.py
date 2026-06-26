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
HEARTBEAT_ROLES: tuple[Role, ...] = ("start", "complete")


def cron_name_prefix(cron_name: str, step: Step) -> str:
    """The cron name without its `-{step}` suffix; equals dataset_id for prod crons
    and the staging-prefixed name for staging crons, so each gets its own heartbeats."""
    return cron_name.removesuffix(f"-{step}")


def heartbeat_key(name_prefix: str, step: Step, role: Role) -> str:
    return f"{name_prefix}_{step}_{role}"


def heartbeat_name(name_prefix: str, step: Step, role: Role) -> str:
    return f"reformatters {name_prefix} {step} {role}"


def ping(url: str, *, failed: bool = False) -> None:
    response = httpx.post(f"{url}/fail" if failed else url, timeout=10)
    response.raise_for_status()


def load_heartbeat_urls() -> dict[str, str]:
    """{dataset_id}_{step}_{role} -> url. Empty outside prod."""
    return load_secret(BETTERSTACK_HEARTBEATS_SECRET_NAME)
