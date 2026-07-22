"""Real-credential integration tests for NASA Earthdata and PPS authentication.

Require `DYNAMICAL_ENV=prod` and the `nasa-earthdata` / `nasa-pps` cluster
secrets:

    DYNAMICAL_ENV=prod uv run pytest tests/nasa/nasa_auth_integration_test.py
"""

import os

import pytest

from reformatters.common.config import Config, Env
from reformatters.nasa.nasa_auth import get_earthdata_session, get_pps_session

# Needs cluster credentials (kubeconfig/op), which CI does not have — CI runs the full
# suite, so gate on an explicit opt-in rather than the `slow` marker alone.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("NASA_CREDENTIALED_TESTS") != "1",
        reason="requires cluster credentials; set NASA_CREDENTIALED_TESTS=1 to run",
    ),
]


@pytest.fixture(autouse=True)
def _prod_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # conftest forces DYNAMICAL_ENV=test; these tests need load_secret to hit the
    # cluster, so opt into prod at runtime.
    monkeypatch.setattr(Config, "env", Env.prod)


def test_earthdata_session_authorizes() -> None:
    session = get_earthdata_session()
    assert str(session.headers.get("Authorization", "")).startswith("Bearer ")
    # Cached per thread.
    assert get_earthdata_session() is session


def test_pps_session_has_basic_auth() -> None:
    session = get_pps_session()
    assert session.auth is not None
    assert get_pps_session() is session
