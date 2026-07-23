import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from reformatters.common.kubernetes import load_secret
from reformatters.common.logging import get_logger
from reformatters.common.retry import retry

log = get_logger(__name__)

_TOKEN_URL = "https://urs.earthdata.nasa.gov/api/users/find_or_create_token"  # noqa: S105


class _ThreadLocalStorage(threading.local):
    earthdata_session: requests.Session
    pps_session: requests.Session


_thread_local = _ThreadLocalStorage()


def _session_with_retries() -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1.0,
        backoff_jitter=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_earthdata_session() -> requests.Session:
    """Authenticated requests.Session for NASA Earthdata (GES DISC, NSIDC).

    Cached per thread. Credentials load from the `nasa-earthdata` secret
    (`{"username", "password"}`).
    """
    if not hasattr(_thread_local, "earthdata_session"):
        _thread_local.earthdata_session = retry(_create_earthdata_session)
    return _thread_local.earthdata_session


def _create_earthdata_session() -> requests.Session:
    credentials = load_secret("nasa-earthdata")
    session = _session_with_retries()
    token_response = session.post(
        _TOKEN_URL,
        auth=(credentials["username"], credentials["password"]),
        timeout=10,
    )
    try:
        token_response.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Failed to get token from NASA Earthdata: {token_response.status_code}: {token_response.text}"
        ) from e
    session.headers["Authorization"] = f"Bearer {token_response.json()['access_token']}"
    return session


def get_pps_session() -> requests.Session:
    """Basic-auth requests.Session for the NASA PPS NRT server (jsimpson).

    Cached per thread. Credentials load from the `nasa-pps` secret
    (`{"username", "password"}`, both the lowercased PPS-registered email).
    """
    if not hasattr(_thread_local, "pps_session"):
        _thread_local.pps_session = retry(_create_pps_session)
    return _thread_local.pps_session


def _create_pps_session() -> requests.Session:
    credentials = load_secret("nasa-pps")
    session = _session_with_retries()
    session.auth = (credentials["username"], credentials["password"])
    return session
