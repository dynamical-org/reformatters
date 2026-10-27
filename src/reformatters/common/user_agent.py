from importlib.metadata import version

import httpx
import icechunk

CLIENT = f"dynamical-reformatters/{version('reformatters')}"
HTTPX_USER_AGENT = f"{CLIENT} python-httpx/{httpx.__version__}"


def icechunk_attribution(workload: str) -> icechunk.Attribution:
    """Labels the User-Agent of a repository's attributed requests, virtual chunk
    fetches included. icechunk leaves list and delete requests unlabelled."""
    return icechunk.Attribution(client=CLIENT, workload=workload)
