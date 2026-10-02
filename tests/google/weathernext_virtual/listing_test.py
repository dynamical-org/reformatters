from base64 import b64encode
from unittest.mock import Mock

import httpx
import pytest

from reformatters.google.weathernext_virtual.listing import (
    OBJECTS_LOCATION,
    PROXY_LOCATION_PREFIX,
    NativeObjectMetadata,
    ObjectListingQuery,
    list_objects,
)


@pytest.mark.parametrize("glob", [None, "store/temperature_*/c/0/0/0"])
@pytest.mark.parametrize("delimiter", [None, "/"])
def test_listing_delimiter_is_independent_and_placeholders_are_skipped(
    glob: str | None, delimiter: str | None
) -> None:
    prefix = "store/temperature_"
    key = prefix + "mean/c/0/0/0"
    md5 = bytes(range(16))
    client = Mock()
    request = httpx.Request("GET", OBJECTS_LOCATION)
    client.get.side_effect = [
        httpx.Response(
            200,
            request=request,
            json={
                "items": [{"name": prefix + "mean/", "size": "0"}],
                "nextPageToken": "second",
            },
        ),
        httpx.Response(
            200,
            request=request,
            json={
                "items": [
                    {"name": key, "size": "123", "md5Hash": b64encode(md5).decode()}
                ]
            },
        ),
    ]

    assert list_objects(client, ObjectListingQuery(prefix, glob, delimiter)) == {
        PROXY_LOCATION_PREFIX + key: NativeObjectMetadata(
            123, '"000102030405060708090a0b0c0d0e0f"'
        )
    }
    params = {"prefix": prefix, "maxResults": "1000"}
    if glob is not None:
        params["matchGlob"] = glob
    if delimiter is not None:
        params["delimiter"] = delimiter
    assert [(call.args, call.kwargs) for call in client.get.call_args_list] == [
        ((OBJECTS_LOCATION,), {"params": params}),
        ((OBJECTS_LOCATION,), {"params": params | {"pageToken": "second"}}),
    ]


def test_zero_byte_non_placeholder_is_rejected() -> None:
    client = Mock()
    client.get.return_value = httpx.Response(
        200,
        request=httpx.Request("GET", OBJECTS_LOCATION),
        json={"items": [{"name": "store/temperature/0.0", "size": "0"}]},
    )
    with pytest.raises(AssertionError, match="invalid object size"):
        list_objects(client, ObjectListingQuery("store/temperature/"))
