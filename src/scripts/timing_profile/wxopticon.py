import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx
import pandas as pd

from reformatters.common.logging import get_logger

from .datasets import Unit
from .sources import TimestampSource, canonical_url, write_atomic

BASE_URL = "https://assets.dynamical.org/wxopticon"
log = get_logger(__name__)

PRODUCTS = {
    "noaa-gfs": ("external-noaa-gfs-aws",),
    "noaa-hrrr-forecast-48-hour": ("external-noaa-hrrr-aws",),
    "noaa-hrrr": ("external-noaa-hrrr-18h-aws", "external-noaa-hrrr-aws"),
    "noaa-rrfs-forecast-84-hour": ("external-noaa-rrfs-84h-aws",),
    "noaa-rrfs-forecast-18-hour": ("external-noaa-rrfs-18h-aws",),
    "noaa-rrfs-forecast-sub-hourly": ("external-noaa-rrfs-subh-aws",),
    "noaa-rrfs-ens": ("external-noaa-rrfs-ens-aws",),
    "noaa-refs-forecast-hourly": ("external-noaa-refs-hourly-aws",),
    "noaa-refs-forecast-3-hourly": ("external-noaa-refs-3-hourly-aws",),
    "ecmwf-aifs-single": ("external-ecmwf-aifs-gcs",),
    "ecmwf-aifs-ens": ("external-ecmwf-aifs-ens-aws",),
    "dwd-icon-eu": ("external-dwd-icon-eu",),
}


class UnavailableError(ValueError):
    pass


@dataclass(frozen=True)
class Evidence:
    arrival: pd.Timestamp
    provenance: str


def products_for(dataset_id: str) -> tuple[str, ...]:
    return next(
        (
            products
            for prefix, products in PRODUCTS.items()
            if dataset_id.startswith(prefix)
        ),
        (),
    )


def parse_events(body: str, product: str) -> dict[str, Evidence]:
    objects = {}
    for line in body.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event["kind"] != "found":
            continue
        if event.get("schema_version") != 4 or event.get("product_id") != product:
            raise UnavailableError(
                "unsupported wxopticon event schema or product identity"
            )
        for artifact in event.get("artifact_bundle") or [event]:
            url = artifact.get("url")
            if url is None:
                continue
            provenance = artifact.get("provenance", event.get("provenance", "unknown"))
            if provenance not in {"directly_observed", "reconstructed_authoritative"}:
                continue
            evidence = Evidence(
                pd.Timestamp(artifact.get("found_at", event["ts"])).tz_convert(None),
                provenance,
            )
            key = canonical_url(url)
            if key not in objects or evidence.arrival < objects[key].arrival:
                objects[key] = evidence
    return objects


class WxopticonSource:
    def __init__(self, source: TimestampSource) -> None:
        self.source = source

    def day(self, product: str, day: pd.Timestamp) -> dict[str, Evidence]:
        return self.read(product, f"archive/{product}/{day:%Y-%m-%d}.jsonl")

    def read(self, product: str, key: str) -> dict[str, Evidence]:
        try:
            return self._read(product, key)
        except (httpx.HTTPError, UnavailableError, json.JSONDecodeError) as error:
            log.info("wxopticon shard unavailable: %s (%s)", key, error)
            return {}

    def _read(self, product: str, key: str) -> dict[str, Evidence]:
        url = f"{BASE_URL}/{key}"
        path = self.source.cache_dir / (
            hashlib.sha256(url.encode()).hexdigest() + ".wxopticon.json"
        )
        if path.exists():
            payload = json.loads(path.read_text())
            assert payload["url"] == url
        else:
            response = self.source.get(url)
            if response.status_code != 404:
                response.raise_for_status()
            payload = {
                "url": url,
                "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "status": response.status_code,
                "body": response.text if response.status_code == 200 else "",
            }
            write_atomic(path, json.dumps(payload))
        body = payload["body"]
        assert isinstance(body, str)
        return parse_events(body, product)

    def measure(
        self, dataset_id: str, units: Sequence[Unit]
    ) -> tuple[dict[str, Evidence], str]:
        products = products_for(dataset_id)
        if not products or not units:
            return {}, "listings fallback: no wxopticon per-file product mapping"
        required = {
            canonical_url(url) for unit in units for file in unit.files for url in file
        }
        days = pd.date_range(
            min(unit.anchor for unit in units).normalize() - pd.Timedelta(days=1),
            max(unit.anchor for unit in units).normalize(),
            freq="1D",
        )
        requests = [(product, day) for product in products for day in days]
        objects: dict[str, Evidence] = {}
        with ThreadPoolExecutor(self.source.workers) as pool:
            results = pool.map(
                lambda request: self.day(*request),
                requests,
                buffersize=self.source.workers,
            )
            for result in results:
                for url, evidence in result.items():
                    if url in required and (
                        url not in objects or evidence.arrival < objects[url].arrival
                    ):
                        objects[url] = evidence
        if required - objects.keys():
            for product in products:
                for url, evidence in self.read(
                    product, f"state/active/{product}.jsonl"
                ).items():
                    if url in required and (
                        url not in objects or evidence.arrival < objects[url].arrival
                    ):
                        objects[url] = evidence
        return objects, f"wxopticon per-file history: {', '.join(products)}"
