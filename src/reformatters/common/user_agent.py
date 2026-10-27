from importlib.metadata import version

import httpx
import icechunk

CLIENT = f"dynamical-reformatters/{version('reformatters')}"
HTTPX_USER_AGENT = f"{CLIENT} python-httpx/{httpx.__version__}"


def icechunk_attribution(workload: str) -> icechunk.Attribution:
    """Labels the User-Agent of every request a repository makes, virtual chunk
    fetches included."""
    return icechunk.Attribution(client=CLIENT, workload=workload)
