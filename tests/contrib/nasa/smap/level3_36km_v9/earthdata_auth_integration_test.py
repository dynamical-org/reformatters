# import pytest

# from reformatters.common.config import Config, Env
# from reformatters.contrib.nasa.smap.level3_36km_v9.earthdata_auth import (
#     get_authenticated_session,
# )


# def test_earthdata_authentication_with_real_credentials(
#     monkeypatch: pytest.MonkeyPatch,
# ) -> None:
#     """
#     Integration test for NASA Earthdata authentication.

#     This test uses real credentials from the nasa-earthdata secret and makes
#     actual network requests to verify authentication works end-to-end.

#     To run this test ensure you have the nasa-earthdata secret configured in your cluster
#     and your local environment is authorized with a kubectl context.
#     """
#     # Monkeypatch to prod environment so load_secret() fetches real credentials
#     monkeypatch.setattr(Config, "env", Env.prod)

#     # Get authenticated session
#     session = get_authenticated_session()

#     # Verify we got a session
#     assert session is not None

#     # Test with a real SMAP file URL
#     test_url = (
#         "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected"
#         "/SMAP/SPL3SMP/009/2015/04/SMAP_L3_SM_P_20150401_R19240_001.h5"
#     )

#     response = session.get(test_url, allow_redirects=True, stream=True)

#     # Should get 200 OK if authentication worked
#     assert response.status_code == 200, (
#         f"Authentication failed. Status: {response.status_code}, URL: {test_url}"
#     )

#     # Verify we can reuse the cached session
#     session2 = get_authenticated_session()
#     assert session2 is session, "Session should be cached and reused"
