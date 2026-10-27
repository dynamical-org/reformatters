from unittest.mock import Mock

import pytest

from scripts.validation.utils import _icechunk_storage


def test_s3_storage_is_anonymous(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VALIDATION_S3_STORAGE_OPTIONS", "invalid json")
    storage = Mock()
    monkeypatch.setattr("scripts.validation.utils.icechunk.s3_storage", storage)
    _icechunk_storage("s3://public-example/forecast/v1.icechunk")
    storage.assert_called_once_with(
        bucket="public-example",
        prefix="forecast/v1.icechunk",
        anonymous=True,
        region="us-west-2",
    )


def test_https_storage_uses_public_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    storage = Mock()
    monkeypatch.setattr("scripts.validation.utils.icechunk.http_storage", storage)
    _icechunk_storage("https://example.invalid/forecast/v1.icechunk")
    storage.assert_called_once_with("https://example.invalid/forecast/v1.icechunk")
