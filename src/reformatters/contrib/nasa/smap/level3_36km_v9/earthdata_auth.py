import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from reformatters.common.kubernetes import load_secret
from reformatters.common.logging import get_logger
from reformatters.common.retry import retry

log = get_logger(__name__)


class _ThreadLocalStorage(threading.local):
    session: requests.Session


_thread_local = _ThreadLocalStorage()


def get_authenticated_session() -> requests.Session:
    """
    Get an authenticated requests.Session for NASA Earthdata.

    Cached per thread. Credentials are loaded from the nasa-earthdata secret.
    """
    if not hasattr(_thread_local, "session"):
        _thread_local.session = retry(_create_authenticated_session)
    return _thread_local.session


def _create_authenticated_session() -> requests.Session:
    """Create and authenticate a requests session with NASA Earthdata."""
    credentials = load_secret("nasa-earthdata")
    username = credentials["username"]
    password = credentials["password"]

    session = requests.Session()

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

    # Find or create token using NASA Earthdata API
    token_url = "https://urs.earthdata.nasa.gov/api/users/find_or_create_token"
    auth = (username, password)
    
    token_response = session.post(token_url, auth=auth, timeout=10)
    token_response.raise_for_status()
    
    token_data = token_response.json()
    token = token_data["access_token"]

    session.headers["Authorization"] = f"Bearer {token}"

    log.info("Created authenticated session for NASA Earthdata")
    return session
