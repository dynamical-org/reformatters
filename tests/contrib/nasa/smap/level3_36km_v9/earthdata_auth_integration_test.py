import pytest

from reformatters.common.config import Config, Env
from reformatters.contrib.nasa.smap.level3_36km_v9.earthdata_auth import (
    get_authenticated_session,
)


@pytest.mark.skip(reason="Requires real NASA Earthdata credentials")
def test_earthdata_authentication_with_real_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Integration test for NASA Earthdata authentication.

    This test uses real credentials from the nasa-earthdata secret and makes
    actual network requests to verify authentication works end-to-end.

    To run this test:
    1. Ensure you have the nasa-earthdata secret configured in your cluster
    2. Remove the @pytest.mark.skip decorator
    3. Run: uv run pytest tests/contrib/nasa/smap/level3_36km_v9/earthdata_auth_integration_test.py
    """
    # Monkeypatch to prod environment so load_secret() fetches real credentials
    monkeypatch.setattr(Config, "env", Env.prod)

    # Get authenticated session
    session = get_authenticated_session()

    # Verify we got a session
    assert session is not None
    assert session.auth is not None

    # Test with a real SMAP file URL (using HEAD to avoid downloading)
    test_url = (
        "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected"
        "/SMAP/SPL3SMP/009/2015/04/SMAP_L3_SM_P_20150401_R19240_001.h5"
    )

    response = session.head(test_url, timeout=30, allow_redirects=True)

    # Should get 200 OK if authentication worked
    assert response.status_code == 200, (
        f"Authentication failed. Status: {response.status_code}, URL: {test_url}"
    )

    # Verify we can reuse the cached session
    session2 = get_authenticated_session()
    assert session2 is session, "Session should be cached and reused"
