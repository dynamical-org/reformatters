"""Register the reformatters webhook endpoint as a wxopticon subscription.

Builds the subscription targets from every dataset's `source_arrival_triggers()`
and calls the admin-gated wxopticon API. See docs/webhooks.md.
"""

import os
from collections.abc import Sequence
from typing import Any

import httpx

from reformatters.common.dynamical_dataset import DynamicalDataset

WXOPTICON_BASE_URL = "https://status.dynamical.org"
_ADMIN_TOKEN_ENV = "WXOPTICON_ADMIN_TOKEN"  # noqa: S105


def build_targets(
    datasets: Sequence[DynamicalDataset[Any, Any]],
) -> list[dict[str, Any]]:
    by_product: dict[str, set[str]] = {}
    for dataset in datasets:
        for trigger in dataset.source_arrival_triggers():
            by_product.setdefault(trigger.product_id, set()).add(trigger.trigger)
    return [
        {"id": product_id, "triggers": sorted(triggers)}
        for product_id, triggers in sorted(by_product.items())
    ]


def register_subscription(
    datasets: Sequence[DynamicalDataset[Any, Any]],
    webhook_url: str,
    *,
    subscription_id: str | None = None,
) -> dict[str, Any]:
    """Create (or PUT-update, if subscription_id given) the wxopticon subscription."""
    token = os.environ[_ADMIN_TOKEN_ENV]
    targets = build_targets(datasets)
    assert targets, "No source_arrival_triggers declared on any dataset"
    body = {
        "url": webhook_url,
        "targets": targets,
        "description": "reformatters operational updates",
    }
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(base_url=WXOPTICON_BASE_URL, timeout=30) as http:
        if subscription_id is not None:
            response = http.put(
                f"/v1/subscriptions/{subscription_id}", json=body, headers=headers
            )
        else:
            response = http.post("/v1/subscriptions", json=body, headers=headers)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result
