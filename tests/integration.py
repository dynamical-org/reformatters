"""Helpers for real integration tests that download from live data sources.

Tests that hit credentialed sources (e.g. NASA Earthdata) can only run where the
relevant kubernetes secret is reachable: a developer machine with a configured
kubectl context, or a CI job with the secret mounted at the expected path. Use
``require_secret`` so those tests run for real where credentials exist and skip
cleanly everywhere else (plain PR CI, contributor machines) instead of erroring.

Public sources that need no credentials (e.g. NOAA MRMS on S3) require no guard.
"""

from typing import Any

import pytest

from reformatters.common.config import Config, Env
from reformatters.common.kubernetes import load_secret


def require_secret(monkeypatch: pytest.MonkeyPatch, secret_name: str) -> dict[str, Any]:
    """Enable secret loading (prod env) and skip the test if the secret is unavailable."""
    monkeypatch.setattr(Config, "env", Env.prod)
    try:
        secret = load_secret(secret_name)
    except Exception as exc:  # noqa: BLE001 - any failure here means creds are unavailable
        pytest.skip(f"{secret_name} credentials unavailable: {exc}")
    if not secret:
        pytest.skip(f"{secret_name} credentials unavailable")
    return secret
