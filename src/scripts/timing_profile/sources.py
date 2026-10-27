import hashlib
import json
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import httpx
import pandas as pd

from reformatters.common.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Listing:
    objects: dict[str, pd.Timestamp]
    note: str


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
    return canonical_url(url).rsplit("/", 1)[0] + "/"


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

    def close(self) -> None:
        self.client.close()

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
            )
        parsed = urlsplit(prefix)
        if parsed.scheme == "s3":
            result = self._s3(parsed.netloc, unquote(parsed.path.lstrip("/")))
        elif parsed.scheme == "gs":
            result = self._gcs(parsed.netloc, unquote(parsed.path.lstrip("/")))
        else:
            response = self.client.get(prefix)
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
        path.write_text(
            json.dumps(
                {
                    "prefix": prefix,
                    "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    "note": result.note,
                    "objects": {
                        key: value.isoformat() for key, value in result.objects.items()
                    },
                }
            )
        )
        return result

    def _s3(self, bucket: str, prefix: str) -> Listing:
        objects = {}
        params = {"list-type": "2", "prefix": prefix}
        endpoint = (
            f"https://s3.us-west-2.amazonaws.com/{bucket}/"
            if bucket == "us-west-2.opendata.source.coop"
            else f"https://{bucket}.s3.amazonaws.com/"
        )
        while True:
            response = self.client.get(endpoint, params=params)
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

    def probe_gcs(self, bucket: str) -> str | None:
        response = self.client.get(
            f"https://storage.googleapis.com/storage/v1/b/{bucket}/o",
            params={"maxResults": "1", "fields": "items(name,updated)"},
        )
        if response.status_code in {401, 403, 404}:
            return f"GCS HTTP {response.status_code}; no anonymous source timestamps; proxy arrival timing unverified"
        response.raise_for_status()
        return None

    def _gcs(self, bucket: str, prefix: str) -> Listing:
        objects = {}
        params = {
            "prefix": prefix,
            "maxResults": "1000",
            "fields": "items(name,updated,size),nextPageToken",
        }
        while True:
            response = self.client.get(
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
