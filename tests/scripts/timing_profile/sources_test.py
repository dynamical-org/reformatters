import gzip
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from scripts.timing_profile.sources import (
    TimestampSource,
    canonical_url,
    listing_prefix,
    parse_http_listing,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://noaa-gfs-bdp-pds.s3.us-east-1.amazonaws.com/a/b",
            "s3://noaa-gfs-bdp-pds/a/b",
        ),
        ("https://noaa-gefs-pds.s3.amazonaws.com/a/b", "s3://noaa-gefs-pds/a/b"),
        ("https://s3-bucket.s3.amazonaws.com/a/b", "s3://s3-bucket/a/b"),
        (
            "https://s3-us-west-2.amazonaws.com/us-west-2.opendata.source.coop/a/b",
            "s3://us-west-2.opendata.source.coop/a/b",
        ),
        (
            "https://storage.googleapis.com/ecmwf-open-data/a/b",
            "gs://ecmwf-open-data/a/b",
        ),
        ("https://example.org/a/b", "https://example.org/a/b"),
        ("https://data.s3.example.org/a/b", "https://data.s3.example.org/a/b"),
        ("https://s3.notamazonaws.com/a/b", "https://s3.notamazonaws.com/a/b"),
    ],
)
def test_anonymous_object_url_normalization(url: str, expected: str) -> None:
    assert canonical_url(url) == expected


def test_http_directory_timestamp_formats_preserve_seconds() -> None:
    objects = parse_http_listing(
        "https://example.org/data/",
        '<a href="a">a</a> 26-Oct-2026 02:45:26 123\n<a href="b">b</a> 2026-10-25 03:06 123\n<tr><td><a href="c" title="c">c</a></td><td>12 MiB</td><td>2026-Feb-19 23:46</td></tr>',
    )
    assert objects == {
        "https://example.org/data/a": pd.Timestamp("2026-10-26T02:45:26"),
        "https://example.org/data/b": pd.Timestamp("2026-10-25T03:06"),
        "https://example.org/data/c": pd.Timestamp("2026-02-19T23:46"),
    }


def test_s3_pagination_requires_nonempty_objects_and_caches_capture(
    tmp_path: Path,
) -> None:
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        continuation = request.url.params.get("continuation-token")
        body = '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        if not continuation:
            body += "<Contents><Key>a/empty</Key><Size>0</Size><LastModified>2026-10-26T00:00:00Z</LastModified></Contents><NextContinuationToken>next</NextContinuationToken>"
        else:
            assert continuation == "next"
            body += "<Contents><Key>a/file</Key><Size>10</Size><LastModified>2026-10-26T01:00:00Z</LastModified></Contents>"
        return httpx.Response(200, text=body + "</ListBucketResult>")

    source = TimestampSource(tmp_path, 1)
    source.client.close()
    source.client = httpx.Client(transport=httpx.MockTransport(respond))
    result = source.list("s3://bucket/a/")
    assert result.objects == {"s3://bucket/a/file": pd.Timestamp("2026-10-26T01")}
    assert source.list("s3://bucket/a/") == result
    assert len(requests) == 2
    source.close()


def test_gcs_uses_updated_and_follows_page_token(tmp_path: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        token = request.url.params.get("pageToken")
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "name": "a/one" if token is None else "a/two",
                        "size": "10",
                        "updated": "2026-10-26T01:00:00Z",
                    }
                ],
                **({"nextPageToken": "next"} if token is None else {}),
            },
        )

    source = TimestampSource(tmp_path, 1)
    source.client.close()
    source.client = httpx.Client(transport=httpx.MockTransport(respond))
    assert set(source.list("gs://bucket/a/").objects) == {
        "gs://bucket/a/one",
        "gs://bucket/a/two",
    }
    source.close()


def test_anonymous_access_gap(tmp_path: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    source = TimestampSource(tmp_path, 1)
    source.client.close()
    source.client = httpx.Client(transport=httpx.MockTransport(respond))
    assert source.list("gs://bucket/a/").objects == {}
    source.close()


def test_dotted_archive_bucket_uses_regional_path_endpoint(tmp_path: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "s3.us-west-2.amazonaws.com"
        assert request.url.path == "/us-west-2.opendata.source.coop/"
        return httpx.Response(
            200,
            text='<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"/>',
        )

    source = TimestampSource(tmp_path, 1)
    source.client.close()
    source.client = httpx.Client(transport=httpx.MockTransport(respond))
    assert source.list("s3://us-west-2.opendata.source.coop/dynamical/").objects == {}
    source.close()


@pytest.mark.parametrize("version", [2, 3])
def test_weathernext_proxy_finalization_pagination_and_raw_capture(
    version: int, tmp_path: Path
) -> None:
    root = (
        "weathernext_2_0_0/zarr/2025_to_present/20261029_06hr_01_preds/predictions.zarr/temperature/"
        if version == 2
        else "weathernext_3_0_0_statistics/zarr/2026_to_present/20261029_06hr_01_preds/predictions.zarr/temperature_2m_"
    )
    names = (
        [root + "0.0.0.0.0", root + "63.0.12.0.0"]
        if version == 2
        else [root + "mean/c/0/0/0", root + "p90/c/0/0/0"]
    )
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert str(request.url).startswith("https://wn.dynamical.org/objects?")
        assert request.url.params["prefix"] == root
        assert request.url.params["maxResults"] == "1000"
        if version == 2:
            assert (
                request.url.params["matchGlob"]
                == root + "{" + ",".join(map(str, range(64))) + "}.*"
            )
            assert request.url.params["delimiter"] == "/"
        else:
            assert "matchGlob" not in request.url.params
            assert "delimiter" not in request.url.params
        page = int(request.url.params.get("pageToken", "0"))
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "name": names[page],
                        "size": "10",
                        "generation": "original",
                        "timeCreated": "2026-10-29T12:00:00Z",
                        "timeFinalized": "2026-10-29T12:00:01Z",
                        "updated": "2026-10-29T15:00:00Z",
                    }
                ],
                **({"nextPageToken": "1"} if page == 0 else {}),
            },
        )

    source = TimestampSource(tmp_path, 1)
    source.client.close()
    source.client = httpx.Client(transport=httpx.MockTransport(respond))
    prefix = "https://wn.dynamical.org/chunks/" + root
    assert listing_prefix("https://wn.dynamical.org/chunks/" + names[0]) == prefix
    listing = source.list(prefix)
    assert listing.objects == {
        "https://wn.dynamical.org/chunks/" + name: pd.Timestamp("2026-10-29T12:00:01")
        for name in names
    }
    assert listing.provenance == "reconstructed_source_finalization"
    assert "not first observed proxy visibility" in listing.note
    assert source.proxy_requests == 2
    assert source.list(prefix) == listing
    assert len(requests) == 2
    records = [
        json.loads(line)
        for line in (tmp_path / "weathernext-requests.jsonl").read_text().splitlines()
    ]
    assert [record["page"] for record in records] == [1, 2]
    assert all(
        record["request_seconds"] >= 0 and record["pacing_wait_seconds"] >= 0
        for record in records
    )
    assert pd.Timestamp(records[1]["started_at"]) - pd.Timestamp(
        records[0]["started_at"]
    ) >= pd.Timedelta("0.49s")
    (raw,) = tmp_path.glob("*.jsonl.gz")
    with gzip.open(raw, "rt") as stream:
        objects = [json.loads(line) for line in stream]
    assert [item["name"] for item in objects] == names
    assert all(item["generation"] == "original" for item in objects)
    source.close()


def test_weathernext_retry_records_each_http_and_transport_attempt(
    tmp_path: Path,
) -> None:
    attempts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503)
        if attempts == 2:
            raise httpx.ReadTimeout("retryable", request=request)
        return httpx.Response(200, json={"items": []})

    source = TimestampSource(tmp_path, 1)
    source.client.close()
    source.client = httpx.Client(transport=httpx.MockTransport(respond))
    assert not source.list(
        "https://wn.dynamical.org/chunks/weathernext_3_0_0_statistics/zarr/2026_to_present/20261029_06hr_01_preds/predictions.zarr/temperature_2m_"
    ).objects
    records = [
        json.loads(line)
        for line in (tmp_path / "weathernext-requests.jsonl").read_text().splitlines()
    ]
    assert source.proxy_requests == 3
    assert [r["status"] for r in records] == [503, None, 200]
    assert [r["page"] for r in records] == [1, 1, 1]
    assert all(r["request_seconds"] >= 0 for r in records)
    source.close()
