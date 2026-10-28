import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common.kubernetes import ReformatCronJob
from scripts.timing_profile.__main__ import measure_source
from scripts.timing_profile.datasets import Unit, expected_units
from scripts.timing_profile.report import aggregate_unit
from scripts.timing_profile.runtime import RuntimeStats
from scripts.timing_profile.sources import Listing, TimestampSource
from scripts.timing_profile.wxopticon import (
    Evidence,
    WxopticonSource,
    parse_events,
    products_for,
)

PRODUCT = "external-noaa-gfs-aws"
ANCHOR = pd.Timestamp("2026-10-20")
ROOT = "https://noaa-gfs-bdp-pds.s3.us-east-1.amazonaws.com/gfs.20261020/00/atmos/"


def event(
    name: str, seconds: int, *, provenance: str = "directly_observed"
) -> dict[str, Any]:
    url = ROOT + name
    return {
        "schema_version": 4,
        "product_id": PRODUCT,
        "init_time": ANCHOR.isoformat() + "Z",
        "kind": "found",
        "ts": (ANCHOR + pd.Timedelta(seconds=seconds + 10)).isoformat() + "Z",
        "url": url,
        "artifact_bundle": [
            {
                "url": url,
                "found_at": (ANCHOR + pd.Timedelta(seconds=seconds)).isoformat() + "Z",
                "provenance": provenance,
            },
            {
                "url": url + ".idx",
                "found_at": (ANCHOR + pd.Timedelta(seconds=seconds + 10)).isoformat()
                + "Z",
                "provenance": provenance,
            },
        ],
    }


def source_with_transport(
    tmp_path: Path, handler: Callable[[httpx.Request], httpx.Response]
) -> TimestampSource:
    source = TimestampSource(tmp_path, 1)
    source.close()
    source.client = httpx.Client(transport=httpx.MockTransport(handler))
    return source


def test_bundles_preserve_artifact_timestamps_provenance_and_cumulative_completion() -> (
    None
):
    rows = [
        event(
            "gfs.t00z.pgrb2.0p25.f000", 900, provenance="reconstructed_authoritative"
        ),
        event("gfs.t00z.pgrb2.0p25.f006", 600),
        event("gfs.t00z.pgrb2b.0p25.f006", 9999),
    ]
    objects = parse_events("\n".join(map(json.dumps, rows)), PRODUCT)
    files = tuple(
        frozenset(
            [
                "s3://noaa-gfs-bdp-pds/gfs.20261020/00/atmos/" + name + suffix
                for suffix in ("", ".idx")
            ]
        )
        for name in ("gfs.t00z.pgrb2.0p25.f000", "gfs.t00z.pgrb2.0p25.f006")
    )
    unit = Unit(ANCHOR, "position", files, 0)
    measured = aggregate_unit(unit, {}, evidence=objects)
    assert measured.first == 610
    assert measured.complete == 910
    assert measured.provenance == {
        "reconstructed_authoritative": 2,
        "directly_observed": 2,
    }
    assert measured.source == "wxopticon"
    assert objects[next(iter(files[0]))].arrival in {
        ANCHOR + pd.Timedelta(seconds=900),
        ANCHOR + pd.Timedelta(seconds=910),
    }
    assert (
        aggregate_unit(
            unit, {}, evidence=objects, as_of=ANCHOR + pd.Timedelta(seconds=905)
        ).complete
        is None
    )


@pytest.mark.parametrize("missing", [False, True])
def test_exact_expected_analysis_file_set_and_one_missing_artifact_falls_back(
    tmp_path: Path, missing: bool
) -> None:
    dataset = next(
        ds for ds in DYNAMICAL_DATASETS if ds.dataset_id == "noaa-gfs-analysis"
    )
    rows = [event(f"gfs.t00z.pgrb2.0p25.f{lead:03}", lead * 60) for lead in range(7)]
    if missing:
        rows[-1]["artifact_bundle"].pop()
    body = "\n".join(map(json.dumps, rows))
    source = source_with_transport(
        tmp_path,
        lambda request: (
            httpx.Response(404)
            if "/active/" in request.url.path
            else httpx.Response(200, text=body)
        ),
    )
    units = list(expected_units(dataset, ANCHOR, ANCHOR + pd.Timedelta("6h"), source))
    objects, note = WxopticonSource(source).measure(dataset.dataset_id, units)
    source.close()
    assert len(units) == 1
    if missing:
        assert objects is None
        assert "maps 13/14 required artifacts" in note
    else:
        assert objects is not None
        assert set(objects) == {
            url for unit in units for file in unit.files for url in file
        }
        assert aggregate_unit(units[0], {}, evidence=objects).complete == 370


@pytest.mark.parametrize("status", [200, 404])
def test_daily_capture_freezes_events_and_absent_shards(
    tmp_path: Path, status: int
) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(status, text=json.dumps(event("a", 60)))

    source = source_with_transport(tmp_path, handler)
    reader = WxopticonSource(source)
    first = reader.day(PRODUCT, ANCHOR)
    assert reader.day(PRODUCT, ANCHOR) == first
    source.close()
    assert len(requests) == 1
    assert bool(first) == (status == 200)
    payload = json.loads(next(tmp_path.glob("*.wxopticon.json")).read_text())
    assert payload["status"] == status
    assert payload["fetched_at"]


@pytest.mark.parametrize(
    "case", ["schema", "json", "http", "inferred", "batch", "mirror"]
)
def test_unusable_history_falls_back_without_inventing_file_evidence(
    tmp_path: Path, case: str
) -> None:
    row = event("a", 60)
    if case == "schema":
        row["schema_version"] = 5
    if case == "inferred":
        for artifact in row["artifact_bundle"]:
            artifact["provenance"] = "inferred"
    if case == "batch":
        row.pop("artifact_bundle")
        row["url"] += ".idx"
        row["member"] = "core-perturbed"
    if case == "mirror":
        for artifact in row["artifact_bundle"]:
            artifact["url"] = artifact["url"].replace(
                "noaa-gfs-bdp-pds", "another-mirror"
            )
    source = source_with_transport(
        tmp_path,
        lambda request: httpx.Response(
            503 if case == "http" else 200,
            text="invalid" if case == "json" else json.dumps(row),
        ),
    )
    unit = Unit(ANCHOR, "run", (frozenset([ROOT + "a", ROOT + "a.idx"]),), 0)
    objects, note = WxopticonSource(source).measure("noaa-gfs-forecast", [unit])
    source.close()
    assert objects is None
    assert "listings fallback" in note


def test_product_mapping_uses_primary_mirror_and_preserves_hourly_coverage() -> None:
    assert products_for("ecmwf-aifs-single-forecast") == ("external-ecmwf-aifs-gcs",)
    assert products_for("ecmwf-ifs-ens-forecast-15-day-0-25-degree") == ()
    assert products_for("noaa-hrrr-analysis")[0] == "external-noaa-hrrr-18h-aws"


def test_active_shard_completes_unarchived_unit(tmp_path: Path) -> None:
    source = source_with_transport(
        tmp_path,
        lambda request: (
            httpx.Response(200, text=json.dumps(event("a", 60)))
            if "/active/" in request.url.path
            else httpx.Response(404)
        ),
    )
    urls = frozenset([ROOT + "a", ROOT + "a.idx"])
    objects, _ = WxopticonSource(source).measure(
        "noaa-gfs-forecast", [Unit(ANCHOR, "run", (urls,), 0)]
    )
    source.close()
    assert objects is not None
    assert set(objects) == {
        "s3://noaa-gfs-bdp-pds/gfs.20261020/00/atmos/a",
        "s3://noaa-gfs-bdp-pds/gfs.20261020/00/atmos/a.idx",
    }


def test_midnight_hrrr_analysis_reads_previous_init_day(tmp_path: Path) -> None:
    dataset = next(
        ds for ds in DYNAMICAL_DATASETS if ds.dataset_id == "noaa-hrrr-analysis"
    )
    product = "external-noaa-hrrr-18h-aws"
    source = TimestampSource(tmp_path, 1)
    units = list(expected_units(dataset, ANCHOR, ANCHOR + pd.Timedelta("1h"), source))
    source.close()
    urls = {url for unit in units for file in unit.files for url in file}
    assert any("hrrr.20261019" in url for url in urls)

    def handler(request: httpx.Request) -> httpx.Response:
        if "external-noaa-hrrr-aws" in request.url.path:
            return httpx.Response(404)
        assert "/active/" not in request.url.path
        date = (
            "20261019" if request.url.path.endswith("2026-10-19.jsonl") else "20261020"
        )
        rows = [
            {
                "schema_version": 4,
                "product_id": product,
                "kind": "found",
                "url": url,
                "ts": "2026-10-20T01:00:00Z",
                "provenance": "directly_observed",
            }
            for url in urls
            if date in url
        ]
        return httpx.Response(200, text="\n".join(map(json.dumps, rows)))

    source = source_with_transport(tmp_path, handler)
    objects, _ = WxopticonSource(source).measure(dataset.dataset_id, units)
    source.close()
    assert objects is not None
    assert set(objects) == urls


@pytest.mark.parametrize("missing", [False, True])
def test_source_selection_applies_to_entire_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: bool
) -> None:
    dataset = next(
        ds for ds in DYNAMICAL_DATASETS if ds.dataset_id == "noaa-gfs-analysis"
    )
    cron = next(
        resource
        for resource in dataset.operational_kubernetes_resources("test")
        if isinstance(resource, ReformatCronJob)
    )
    source = TimestampSource(tmp_path, 1)
    units = list(expected_units(dataset, ANCHOR, ANCHOR + pd.Timedelta("6h"), source))
    urls = {url for unit in units for file in unit.files for url in file}
    objects = {
        url: Evidence(ANCHOR + pd.Timedelta(seconds=60), "reconstructed_authoritative")
        for url in urls
    }
    if missing:
        objects.pop(min(objects))
    monkeypatch.setattr(WxopticonSource, "read", lambda self, product, key: objects)
    called = []

    def list_many(prefixes: set[str]) -> dict[str, Listing]:
        called.append(prefixes)
        return {
            prefix: Listing(
                {url: ANCHOR + pd.Timedelta(seconds=120) for url in urls}, "S3"
            )
            for prefix in prefixes
        }

    monkeypatch.setattr(source, "list_many", list_many)
    rendered = measure_source(
        dataset,
        cron,
        RuntimeStats(None, None, None, None, 0, 0, 0, 0, None, None),
        source,
        ANCHOR,
        ANCHOR + pd.Timedelta("6h"),
        set(),
    )
    source.close()
    assert bool(called[0]) == missing
    assert ("source **listings**" if missing else "source **wxopticon**") in rendered
    if missing:
        assert "listings fallback" in rendered
        assert "reconstructed_authoritative" not in rendered
    else:
        assert "reconstructed_authoritative=14" in rendered
