from base64 import b64decode
from typing import NamedTuple

import httpx
import icechunk
from kubernetes.client.exceptions import ApiException

from reformatters.common import kubernetes, monitoring
from reformatters.common.logging import get_logger
from reformatters.common.retry import retry

PROXY_LOCATION_PREFIX = "https://wn.dynamical.org/chunks/"
OBJECTS_LOCATION = "https://wn.dynamical.org/objects"
SERVICE_TOKEN_SECRET_NAME = "wn-service-token"  # noqa: S105

log = get_logger(__name__)


class NativeObjectMetadata(NamedTuple):
    size: int
    etag_checksum: str


def weathernext_virtual_chunk_containers() -> tuple[
    icechunk.VirtualChunkContainer, ...
]:
    return (
        icechunk.VirtualChunkContainer(PROXY_LOCATION_PREFIX, icechunk.http_store()),
    )


def service_token_headers() -> dict[str, str]:
    """The proxy's service-tier header, or none. The token only selects a rate-limit
    allowance, so a secret that is absent or not readable by this caller (404/403 from
    the Kubernetes API outside a job) falls back to the public tier. Anything else,
    such as a malformed secret, still fails."""
    try:
        secret = kubernetes.load_secret(SERVICE_TOKEN_SECRET_NAME)
    except ApiException as error:
        if error.status not in (403, 404):
            raise
        log.warning(
            f"{SERVICE_TOKEN_SECRET_NAME} not readable (HTTP {error.status}); "
            "reading wn.dynamical.org on the public tier"
        )
        return {}
    token = secret.get("WN_SERVICE_TOKEN")
    if not token:
        log.warning(
            f"{SERVICE_TOKEN_SECRET_NAME} has no WN_SERVICE_TOKEN; "
            "reading wn.dynamical.org on the public tier"
        )
        return {}
    monitoring.register_sensitive_value(token)
    return {"X-Dynamical-Service-Token": token}


def weathernext_read_chunk_containers() -> tuple[icechunk.VirtualChunkContainer, ...]:
    """Carries the service token, so must never reach a saved repository config."""
    return (
        icechunk.VirtualChunkContainer(
            PROXY_LOCATION_PREFIX,
            icechunk.http_store(headers=service_token_headers() or None),
        ),
    )


class ObjectListingQuery(NamedTuple):
    prefix: str
    match_glob: str | None = None
    delimiter: str | None = None


def list_objects(
    client: httpx.Client, query: ObjectListingQuery
) -> dict[str, NativeObjectMetadata] | None:
    objects: dict[str, NativeObjectMetadata] = {}
    page_token: str | None = None
    while True:
        params = {"prefix": query.prefix, "maxResults": "1000"}
        if query.match_glob is not None:
            params["matchGlob"] = query.match_glob
        if query.delimiter is not None:
            params["delimiter"] = query.delimiter
        if page_token is not None:
            params["pageToken"] = page_token

        def get_page(params: dict[str, str] = params) -> httpx.Response:
            response = client.get(OBJECTS_LOCATION, params=params)
            if response.status_code in {408, 429} or response.status_code >= 500:
                response.raise_for_status()
            return response

        response = retry(
            get_page,
            retryable_exceptions=(httpx.RequestError, httpx.HTTPStatusError),
        )
        if response.status_code in {403, 404}:
            return None
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("items", []):
            key = str(item["name"])
            assert key.startswith(query.prefix), (
                f"listed object escaped prefix {query.prefix}: {key}"
            )
            if key.endswith("/"):
                continue
            size = int(item["size"])
            assert size > 0, f"invalid object size for {key}: {size}"
            location = f"{PROXY_LOCATION_PREFIX}{key}"
            assert location not in objects, f"duplicate listed object: {key}"
            md5 = b64decode(str(item["md5Hash"]), validate=True)
            assert len(md5) == 16, f"invalid object MD5 for {key}"
            objects[location] = NativeObjectMetadata(
                size=size, etag_checksum=f'"{md5.hex()}"'
            )
        page_token = payload.get("nextPageToken")
        if page_token is None:
            return objects
