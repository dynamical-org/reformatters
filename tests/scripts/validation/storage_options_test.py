import json
from unittest.mock import Mock

import pytest

from scripts.validation.utils import _icechunk_storage


def test_private_storage_options_are_scoped_to_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = {
        "endpoint_url": "https://storage.example.invalid",
        "access_key_id": "example-access-key",
        "secret_access_key": "example-secret-key",
        "region": "auto",
        "force_path_style": True,
    }
    monkeypatch.setenv(
        "VALIDATION_S3_STORAGE_OPTIONS", json.dumps({"private-example": options})
    )
    storage = Mock()
    monkeypatch.setattr("scripts.validation.utils.icechunk.s3_storage", storage)

    _icechunk_storage("s3://private-example/forecast/v1.icechunk")
    storage.assert_called_once_with(
        bucket="private-example", prefix="forecast/v1.icechunk", **options
    )

    storage.reset_mock()
    _icechunk_storage("s3://public-reference/analysis/v1.icechunk")
    storage.assert_called_once_with(
        bucket="public-reference",
        prefix="analysis/v1.icechunk",
        anonymous=True,
        region="us-west-2",
    )


def test_private_storage_options_do_not_change_http_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VALIDATION_S3_STORAGE_OPTIONS", "invalid json")
    storage = Mock()
    monkeypatch.setattr("scripts.validation.utils.icechunk.http_storage", storage)
    _icechunk_storage("https://example.invalid/forecast/v1.icechunk")
    storage.assert_called_once_with("https://example.invalid/forecast/v1.icechunk")


def test_invalid_private_storage_options_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VALIDATION_S3_STORAGE_OPTIONS", "invalid json")
    with pytest.raises(json.JSONDecodeError):
        _icechunk_storage("s3://private-example/forecast/v1.icechunk")
