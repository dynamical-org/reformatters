import gzip
import hashlib
import json
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import IO
from urllib.parse import unquote, urljoin, urlsplit

import httpx
import pandas as pd

from reformatters.common.download import RateLimiter
from reformatters.common.logging import get_logger
from reformatters.common.retry import retry
from reformatters.google.weathernext_virtual.listing import (
    OBJECTS_LOCATION,
    PROXY_LOCATION_PREFIX,
)

log = get_logger(__name__)


@dataclass(frozen=True)
class Listing:
    objects: dict[str, pd.Timestamp]
    note: str
    provenance: str = "current_object_write"


def canonical_url(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if parsed.scheme in {"s3", "gs"}:
        return url
    if host.endswith(".amazonaws.com"):
        bucket, s3_label, _ = host.partition(".s3")
        if s3_label:
            return f"s3://{bucket}{parsed.path}"
        if host.startswith(("s3.", "s3-")):
            return "s3://" + parsed.path.lstrip("/")
    if host == "storage.googleapis.com":
        return "gs://" + parsed.path.lstrip("/")
    return url


def listing_prefix(url: str) -> str:
    if url.startswith(PROXY_LOCATION_PREFIX + "weathernext_3_"):
        root, _, chunk = url.partition("/predictions.zarr/")
        variable = chunk.split("/", 1)[0]
        variable = variable.rsplit("_", 1)[0] + "_"
        return root + "/predictions.zarr/" + variable
    return canonical_url(url).rsplit("/", 1)[0] + "/"


@contextmanager
def atomic_file(path: Path) -> Iterator[IO[bytes]]:
    """Yield a private file that replaces path only once it is closed."""
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name, suffix=".tmp", delete=False
    ) as file:
        yield file
    Path(file.name).replace(path)


def write_atomic(path: Path, text: str) -> None:
    with atomic_file(path) as file:
        file.write(text.encode())


def parse_http_listing(url: str, body: str) -> dict[str, pd.Timestamp]:
    objects = {}
    pattern = r'<a\b[^>]*href="([^"]+)"[^>]*>[^<]*</a>([^\n]*)'
    date = r"\d{2}-[A-Za-z]{3}-\d{4}|\d{4}-\d{2}-\d{2}|\d{4}-[A-Za-z]{3}-\d{2}"
    for href, tail in re.findall(pattern, body):
        modified = re.search(
            rf"({date}) \d{{2}}:\d{{2}}(?::\d{{2}})?", re.sub(r"<[^>]+>", " ", tail)
        )
        if not href.endswith("/") and modified:
            objects[urljoin(url, unescape(href))] = pd.Timestamp(modified[0])
    return objects


class TimestampSource:
    def __init__(self, cache_dir: Path, workers: int) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.client = httpx.Client(timeout=60, follow_redirects=True)
        self.proxy_requests = 0
        self._proxy_rate_limiter = RateLimiter(max_per_minute=120)

    def close(self) -> None:
        self.client.close()

    def get(
        self,
        url: str,
        params: dict[str, str] | None = None,
        proxy_page: tuple[str, int] | None = None,
    ) -> httpx.Response:
        """GET, retrying throttling, server errors and dropped connections. A proxy
        page (prefix, page number) is paced and each attempt is logged."""

        def attempt() -> httpx.Response:
            response = (
                self.client.get(url, params=params)
                if proxy_page is None
                else self._proxy_get(url, params, *proxy_page)
            )
            if response.status_code in {408, 429} or response.status_code >= 500:
                response.raise_for_status()
            return response

        return retry(
            attempt,
            max_attempts=8,
            retryable_exceptions=(httpx.RequestError, httpx.HTTPStatusError),
        )

    def _proxy_get(
        self, url: str, params: dict[str, str] | None, prefix: str, page: int
    ) -> httpx.Response:
        waiting = time.monotonic()
        self._proxy_rate_limiter.wait()
        pacing_wait = time.monotonic() - waiting
        self.proxy_requests += 1
        started = time.monotonic()
        started_at = pd.Timestamp.now(tz="UTC").isoformat()
        status = None
        try:
            response = self.client.get(url, params=params)
            status = response.status_code
        finally:
            record = {
                "prefix": prefix,
                "page": page,
                "started_at": started_at,
                "ended_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "request_seconds": time.monotonic() - started,
                "pacing_wait_seconds": pacing_wait,
                "status": status,
            }
            with (self.cache_dir / "weathernext-requests.jsonl").open("a") as requests:
                requests.write(json.dumps(record) + "\n")
        return response

    def list_many(self, prefixes: set[str]) -> dict[str, Listing]:
        with ThreadPoolExecutor(self.workers) as pool:
            return dict(
                zip(
                    sorted(prefixes), pool.map(self.list, sorted(prefixes)), strict=True
                )
            )

    def list(self, prefix: str) -> Listing:
        path = self.cache_dir / (hashlib.sha256(prefix.encode()).hexdigest() + ".json")
        if path.exists():
            payload = json.loads(path.read_text())
            assert payload["prefix"] == prefix
            return Listing(
                {key: pd.Timestamp(value) for key, value in payload["objects"].items()},
                payload["note"],
                payload.get("provenance", "current_object_write"),
            )
        parsed = urlsplit(prefix)
        if prefix.startswith(PROXY_LOCATION_PREFIX):
            result = self._weathernext(prefix)
        elif parsed.scheme == "s3":
            result = self._s3(parsed.netloc, unquote(parsed.path.lstrip("/")))
        elif parsed.scheme == "gs":
            result = self._gcs(parsed.netloc, unquote(parsed.path.lstrip("/")))
        else:
            response = self.get(prefix)
            if response.status_code in {403, 404}:
                result = Listing(
                    {}, f"HTTP {response.status_code}; no anonymous listing"
                )
            else:
                response.raise_for_status()
                result = Listing(
                    parse_http_listing(prefix, response.text),
                    "HTTP directory Last modified (UTC; minute/second resolution)",
                )
        write_atomic(
            path,
            json.dumps(
                {
                    "prefix": prefix,
                    "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    "note": result.note,
                    "provenance": result.provenance,
                    "objects": {
                        key: value.isoformat() for key, value in result.objects.items()
                    },
                }
            ),
        )
        return result

    def _weathernext(self, prefix: str) -> Listing:
        key = prefix.removeprefix(PROXY_LOCATION_PREFIX)
        params = {"prefix": key, "maxResults": "1000"}
        if "weathernext_2_" in key:
            members = "{" + ",".join(map(str, range(64))) + "}"
            params.update(matchGlob=key + members + ".*", delimiter="/")
        objects = {}
        page = 0
        raw_path = self.cache_dir / (
            hashlib.sha256(prefix.encode()).hexdigest() + ".jsonl.gz"
        )
        with (
            atomic_file(raw_path) as file,
            gzip.open(file, "wt", compresslevel=1) as raw,
        ):
            while True:
                page += 1
                response = self.get(OBJECTS_LOCATION, params, (prefix, page))
                response.raise_for_status()
                payload = response.json()
                for entry in payload.get("items", []):
                    assert entry["name"].startswith(key)
                    raw.write(json.dumps(entry) + "\n")
                    if int(entry["size"]) > 0:
                        objects[PROXY_LOCATION_PREFIX + entry["name"]] = pd.Timestamp(
                            entry["timeFinalized"]
                        ).tz_convert(None)
                token = payload.get("nextPageToken")
                if token is None:
                    break
                params["pageToken"] = token
        return Listing(
            objects,
            "WeatherNext proxy timeFinalized; reconstructed current-generation source availability, not first observed proxy visibility; reuploads can overstate arrival",
            "reconstructed_source_finalization",
        )

    def _s3(self, bucket: str, prefix: str) -> Listing:
        objects = {}
        params = {"list-type": "2", "prefix": prefix}
        endpoint = (
            f"https://s3.us-west-2.amazonaws.com/{bucket}/"
            if bucket == "us-west-2.opendata.source.coop"
            else f"https://{bucket}.s3.amazonaws.com/"
        )
        while True:
            response = self.get(endpoint, params=params)
            if response.status_code in {403, 404}:
                return Listing(
                    {}, f"S3 HTTP {response.status_code}; no anonymous listing"
                )
            response.raise_for_status()
            root = ET.fromstring(response.content)  # noqa: S314
            ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
            for entry in root.findall("s3:Contents", ns):
                key = entry.findtext("s3:Key", namespaces=ns)
                modified = entry.findtext("s3:LastModified", namespaces=ns)
                assert key is not None
                assert modified is not None
                if int(entry.findtext("s3:Size", namespaces=ns) or 0) > 0:
                    objects[f"s3://{bucket}/{key}"] = pd.Timestamp(modified).tz_convert(
                        None
                    )
            token = root.findtext("s3:NextContinuationToken", namespaces=ns)
            if token is None:
                return Listing(
                    objects, "S3 LastModified; overwrites can overstate arrival"
                )
            params["continuation-token"] = token

    def _gcs(self, bucket: str, prefix: str) -> Listing:
        objects = {}
        params = {
            "prefix": prefix,
            "maxResults": "1000",
            "fields": "items(name,updated,size),nextPageToken",
        }
        while True:
            response = self.get(
                f"https://storage.googleapis.com/storage/v1/b/{bucket}/o", params=params
            )
            if response.status_code in {401, 403, 404}:
                return Listing(
                    {}, f"GCS HTTP {response.status_code}; no anonymous timestamps"
                )
            response.raise_for_status()
            payload = response.json()
            for entry in payload.get("items", []):
                if int(entry["size"]) > 0:
                    objects[f"gs://{bucket}/{entry['name']}"] = pd.Timestamp(
                        entry["updated"]
                    ).tz_convert(None)
            token = payload.get("nextPageToken")
            if token is None:
                return Listing(objects, "GCS updated; rewrites can overstate arrival")
            params["pageToken"] = token
