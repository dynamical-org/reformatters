import functools

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from reformatters.common.kubernetes import load_secret
from reformatters.common.logging import get_logger
from reformatters.common.retry import retry

log = get_logger(__name__)


@functools.cache
def get_authenticated_session() -> requests.Session:
    """
    Get an authenticated requests.Session for NASA Earthdata.

    Cached per thread/process. Credentials are loaded from the nasa-earthdata secret.
    """
    return retry(_create_authenticated_session)


def _create_authenticated_session() -> requests.Session:
    """Create and authenticate a requests session with NASA Earthdata."""
    credentials = load_secret("nasa-earthdata")
    username = credentials["username"]
    password = credentials["password"]

    session = requests.Session()
    session.auth = (username, password)

    # Configure retries for the session
    retry_strategy = Retry(
        total=5,
        backoff_factor=1.0,
        backoff_jitter=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Trigger authentication by making a request to URS
    # This will follow redirects and establish the authenticated session
    auth_url = "https://urs.earthdata.nasa.gov/oauth/authorize"
    response = session.get(
        auth_url,
        params={
            "client_id": "earthdata",
            "response_type": "code",
            "redirect_uri": "https://data.nsidc.earthdatacloud.nasa.gov/",
        },
        allow_redirects=True,
        timeout=30,
    )
    response.raise_for_status()

    log.info("Successfully authenticated with NASA Earthdata")
    return session
