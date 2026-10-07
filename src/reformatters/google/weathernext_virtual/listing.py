from base64 import b64decode
from typing import NamedTuple

import httpx
import icechunk

from reformatters.common.retry import retry

PROXY_LOCATION_PREFIX = "https://wn.dynamical.org/chunks/"
OBJECTS_LOCATION = "https://wn.dynamical.org/objects"


class NativeObjectMetadata(NamedTuple):
    size: int
    etag_checksum: str


def weathernext_virtual_chunk_containers() -> tuple[
    icechunk.VirtualChunkContainer, ...
]:
    return (
        icechunk.VirtualChunkContainer(PROXY_LOCATION_PREFIX, icechunk.http_store()),
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
