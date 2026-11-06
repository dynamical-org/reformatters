from collections.abc import Iterator
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, PaginationError
from botocore.paginate import Paginator
from botocore.session import get_session

from reformatters.common.download import SourceAccessError
from reformatters.ecmwf.fast import source
from reformatters.ecmwf.fast.source import EcmwfFastSourceFileCoord, Family
from reformatters.ecmwf.fast.template_config import FastModel


@pytest.fixture(autouse=True)
def clear_client_cache() -> Iterator[None]:
    client_factory = source.listing_client
    client_factory.cache_clear()
    yield
    client_factory.cache_clear()


@pytest.fixture
def client() -> MagicMock:
    session = get_session()
    client = MagicMock()
    client.get_paginator.return_value = Paginator(
        client.list_objects_v2,
        session.get_paginator_model("s3").get_paginator("ListObjectsV2"),
        session.get_service_model("s3").operation_model("ListObjectsV2"),
    )
    return client


@pytest.mark.parametrize("model", ["ifs", "aifs"])
@pytest.mark.parametrize(("family", "member"), [("cf", 0), ("pf", 1)])
def test_filename_supplies_long_lead_and_family_member(
    model: FastModel, family: Family, member: int
) -> None:
    coord = EcmwfFastSourceFileCoord(
        model=model,
        family=family,
        init_time=pd.Timestamp("2026-11-03"),
        lead_time=pd.Timedelta("360h"),
        data_vars=[],
        source_members=(0,) if family == "cf" else tuple(range(1, 51)),
    )
    assert coord.key.endswith("20261103T000000Z_20261118T000000Z_360h")
    assert coord.get_url() == source.SOURCE_PREFIX + coord.key
    assert coord.out_loc()["ensemble_member"] == member
    assert coord.out_loc()["lead_time"] == pd.Timedelta("360h")
    assert list(coord.members) == ([0] if member == 0 else list(range(1, 51)))


def test_literal_prefix_pagination_revisits_earlier_keys(
    monkeypatch: pytest.MonkeyPatch,
    client: MagicMock,
) -> None:
    prefix = "dyn_a2_ifs-ens_od_enfo_pf_20261103T000000Z_"
    modified = datetime(2026, 11, 4, tzinfo=UTC)
    page = [
        {"Key": f"{prefix}{i}", "Size": 100, "LastModified": modified}
        for i in range(1000)
    ]
    client.list_objects_v2.side_effect = [
        {"Contents": page, "IsTruncated": True, "NextContinuationToken": "page-two"},
        {
            "Contents": [
                {"Key": prefix + "older-arrival", "Size": 42, "LastModified": modified}
            ],
            "IsTruncated": False,
        },
        {
            "Contents": [
                {
                    "Key": prefix + "earlier-than-previous",
                    "Size": 43,
                    "LastModified": modified,
                }
            ]
        },
    ]
    monkeypatch.setattr(source, "listing_client", lambda: client)
    assert len(list(source.list_prefix(prefix))) == 1001
    assert list(source.list_prefix(prefix)) == [
        (prefix + "earlier-than-previous", 43, modified, None)
    ]
    assert all(
        call.args == ("list_objects_v2",)
        for call in client.get_paginator.call_args_list
    )
    calls = client.list_objects_v2.call_args_list
    assert (
        calls[0].kwargs
        == calls[2].kwargs
        == {"Bucket": source.SOURCE_BUCKET, "Prefix": prefix, "MaxKeys": 1000}
    )
    assert calls[1].kwargs == {
        "Bucket": source.SOURCE_BUCKET,
        "Prefix": prefix,
        "ContinuationToken": "page-two",
        "MaxKeys": 1000,
    }


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("AccessDenied", 403),
        ("InvalidAccessKeyId", 400),
        ("ExpiredToken", 400),
        ("InvalidToken", 400),
        ("SignatureDoesNotMatch", 400),
        ("Unauthorized", 401),
    ],
)
@pytest.mark.parametrize("later_page", [False, True])
def test_authentication_failures_are_fatal(
    monkeypatch: pytest.MonkeyPatch,
    client: MagicMock,
    code: str,
    status: int,
    later_page: bool,
) -> None:
    failure = ClientError(
        {
            "Error": {"Code": code},
            "ResponseMetadata": {
                "HTTPStatusCode": status,
                "RequestId": "test",
                "HostId": "test",
                "HTTPHeaders": {},
                "RetryAttempts": 0,
            },
        },
        "ListObjectsV2",
    )
    client.list_objects_v2.side_effect = [
        *(
            [{"IsTruncated": True, "NextContinuationToken": "next"}]
            if later_page
            else []
        ),
        failure,
    ]
    monkeypatch.setattr(source, "listing_client", lambda: client)
    with pytest.raises(SourceAccessError):
        list(source.list_prefix("literal-prefix"))


@pytest.mark.parametrize(
    "failure",
    [
        ClientError({"Error": {"Code": "SlowDown"}}, "ListObjectsV2"),
        EndpointConnectionError(endpoint_url="https://synthetic.invalid"),
    ],
)
def test_pagination_failure_is_not_partial_success(
    client: MagicMock,
    failure: Exception,
) -> None:
    client.list_objects_v2.side_effect = [
        {
            "Contents": [
                {
                    "Key": "prefix-a",
                    "Size": 1,
                    "LastModified": datetime(2026, 11, 4, tzinfo=UTC),
                }
            ],
            "IsTruncated": True,
            "NextContinuationToken": "next",
        },
        failure,
    ]
    assert source.list_prefix_or_defer("prefix-", client) == []


def test_repeated_continuation_fails_closed(client: MagicMock) -> None:
    client.list_objects_v2.return_value = {
        "IsTruncated": True,
        "NextContinuationToken": "repeated",
    }
    with pytest.raises(PaginationError, match="same next token"):
        list(source.list_prefix("prefix", client))
    assert client.list_objects_v2.call_count == 2


@pytest.mark.parametrize("session_token", [None, "synthetic-session"])
def test_listing_uses_only_explicit_source_credentials(
    monkeypatch: pytest.MonkeyPatch, session_token: str | None
) -> None:
    credentials = {
        "access_key_id": "synthetic-source",
        "secret_access_key": "synthetic-source-secret",
    }
    if session_token is not None:
        credentials["session_token"] = session_token
    loader = MagicMock(return_value=credentials)
    factory = MagicMock()
    monkeypatch.setattr(source.download, "load_s3_credentials", loader)
    monkeypatch.setattr(source.boto3, "client", factory)
    source.listing_client()
    source.listing_client()
    loader.assert_called_once_with(source.SOURCE_SECRET)
    kwargs = factory.call_args.kwargs
    assert kwargs["aws_access_key_id"] == "synthetic-source"
    assert kwargs["aws_secret_access_key"] == credentials["secret_access_key"]
    assert kwargs["aws_session_token"] == session_token
    assert kwargs["endpoint_url"] == source.ENDPOINT
    assert kwargs["region_name"] == "auto"
    assert kwargs["config"].s3 == {"addressing_style": "path"}
    assert kwargs["config"].retries["total_max_attempts"] == 1
    assert kwargs["config"].max_pool_connections == 64
    assert kwargs["config"].connect_timeout == kwargs["config"].read_timeout == 2
    factory.assert_called_once()


@pytest.mark.parametrize("code", ["SlowDown", "InternalError", "RequestTimeout"])
def test_known_transient_listing_failure_defers_prefix(
    code: str, client: MagicMock
) -> None:
    client.list_objects_v2.side_effect = ClientError(
        {"Error": {"Code": code}}, "ListObjectsV2"
    )
    assert source.list_prefix_or_defer("prefix", client) == []


def test_transport_failure_defers_prefix_but_unexpected_errors_propagate(
    client: MagicMock,
) -> None:
    client.list_objects_v2.side_effect = EndpointConnectionError(
        endpoint_url="https://synthetic.invalid"
    )
    assert source.list_prefix_or_defer("prefix", client) == []
    client.list_objects_v2.side_effect = ClientError(
        {"Error": {"Code": "InvalidArgument"}}, "ListObjectsV2"
    )
    with pytest.raises(ClientError):
        source.list_prefix_or_defer("prefix", client)
    client.list_objects_v2.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied"}}, "ListObjectsV2"
    )
    with pytest.raises(SourceAccessError):
        source.list_prefix_or_defer("prefix", client)


@pytest.mark.parametrize("etag", [None, "", 'W/"weak"', '"opaque-multipart-17"'])
def test_listing_preserves_opaque_etag(client: MagicMock, etag: str | None) -> None:
    modified = datetime(2026, 11, 4, tzinfo=UTC)
    row = {"Key": "prefix-a", "Size": 42, "LastModified": modified}
    if etag is not None:
        row["ETag"] = etag
    client.list_objects_v2.return_value = {"Contents": [row]}
    assert list(source.list_prefix("prefix-", client)) == [
        ("prefix-a", 42, modified, etag)
    ]
