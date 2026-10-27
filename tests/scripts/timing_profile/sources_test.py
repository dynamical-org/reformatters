from pathlib import Path

import httpx
import pandas as pd
import pytest

from scripts.timing_profile.sources import (
    TimestampSource,
    canonical_url,
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
        (
            "https://s3-us-west-2.amazonaws.com/us-west-2.opendata.source.coop/a/b",
            "s3://us-west-2.opendata.source.coop/a/b",
        ),
        (
            "https://storage.googleapis.com/ecmwf-open-data/a/b",
            "gs://ecmwf-open-data/a/b",
        ),
        ("https://example.org/a/b", "https://example.org/a/b"),
    ],
)
def test_anonymous_object_url_normalization(url: str, expected: str) -> None:
    assert canonical_url(url) == expected


def test_http_directory_timestamp_formats_preserve_seconds() -> None:
    objects = parse_http_listing(
        "https://example.org/data/",
        '<a href="a">a</a> 26-Oct-2026 02:45:26 123\n<a href="b">b</a> 2026-10-25 03:06 123',
    )
    assert objects == {
        "https://example.org/data/a": pd.Timestamp("2026-10-26T02:45:26"),
        "https://example.org/data/b": pd.Timestamp("2026-10-25T03:06"),
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


def test_http_head_and_anonymous_access_gap(tmp_path: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(
                200, headers={"Last-Modified": "Mon, 26 Oct 2026 01:02:03 GMT"}
            )
        return httpx.Response(403)

    source = TimestampSource(tmp_path, 1)
    source.client.close()
    source.client = httpx.Client(transport=httpx.MockTransport(respond))
    assert source.head("https://example.org/file") == pd.Timestamp(
        "2026-10-26T01:02:03"
    )
    assert source.probe_gcs("bucket") is not None
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
