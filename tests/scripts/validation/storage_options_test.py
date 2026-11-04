import json
from typing import Any
from unittest.mock import Mock, call

import icechunk
import pytest

from reformatters.common.download import SourceAccessError
from scripts.validation import utils
from scripts.validation.utils import (
    _icechunk_storage,
    is_virtual_store,
    open_icechunk_readonly,
)


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


@pytest.mark.parametrize("virtual", [False, True])
@pytest.mark.parametrize("private_output", [False, True])
def test_classification_open_and_reopen_apply_independent_bindings(
    monkeypatch: pytest.MonkeyPatch, virtual: bool, private_output: bool
) -> None:
    target = "s3://target/forecast/v1.icechunk"
    reference = "s3://reference/analysis/v1.icechunk"
    prefix = "s3://private-source/"
    binding: dict[str, object] = (
        {"output_secret_name": "writer"} if private_output else {}
    )
    if virtual:
        binding["source_secret_names"] = {prefix: "reader"}
    monkeypatch.setenv("REFORMATTERS_ICECHUNK_ACCESS", json.dumps({target: binding}))
    writer = {
        "access_key_id": "writer-id",
        "secret_access_key": "synthetic-writer-key",
        "endpoint_url": "https://output.invalid",
        "force_path_style": True,
    }
    reader = {"access_key_id": "reader-id", "secret_access_key": "synthetic-reader-key"}
    secrets = Mock(side_effect=lambda name: {"writer": writer, "reader": reader}[name])
    monkeypatch.setattr("reformatters.common.kubernetes.load_secret", secrets)
    config = icechunk.RepositoryConfig.default()
    if virtual:
        config.set_virtual_chunk_container(
            icechunk.VirtualChunkContainer(
                prefix, icechunk.s3_store(region="us-west-2")
            )
        )
    fetch = Mock(return_value=config)
    monkeypatch.setattr(utils.icechunk.Repository, "fetch_config", fetch)
    storage = Mock()
    monkeypatch.setattr(utils.icechunk, "s3_storage", storage)
    opened = Mock()
    monkeypatch.setattr(utils.icechunk.Repository, "open", opened)
    authorize = Mock(wraps=utils.virtual_chunk_credentials)
    monkeypatch.setattr(utils, "virtual_chunk_credentials", authorize)

    assert is_virtual_store(target) is virtual
    assert secrets.call_args_list == ([call("writer")] if private_output else [])
    for _ in range(2):
        assert (
            open_icechunk_readonly(target + "/")
            is opened.return_value.readonly_session.return_value.store
        )
    open_icechunk_readonly(reference)

    assert fetch.call_count == 4
    assert secrets.call_args_list.count(call("writer")) == (3 if private_output else 0)
    assert secrets.call_args_list.count(call("reader")) == (2 if virtual else 0)
    target_options = (
        writer if private_output else {"anonymous": True, "region": "us-west-2"}
    )
    assert storage.call_args_list == [
        call(bucket="target", prefix="forecast/v1.icechunk", **target_options)
    ] * 3 + [
        call(
            bucket="reference",
            prefix="analysis/v1.icechunk",
            anonymous=True,
            region="us-west-2",
        )
    ]
    if virtual:
        assert authorize.call_count == 3
        assert config.virtual_chunk_containers is not None
        assert authorize.call_args_list[0].args == (
            list(config.virtual_chunk_containers.values()),
            {prefix: "reader"},
        )
        assert authorize.call_args_list[-1].args[1] == {}
    else:
        authorize.assert_not_called()
        assert opened.call_args.kwargs["authorize_virtual_chunk_access"] is None


@pytest.mark.parametrize("role", ["output", "source"])
@pytest.mark.parametrize("credentials", [{}, {"access_key_id": "incomplete"}])
def test_invalid_bound_credentials_fail_before_repository_open(
    monkeypatch: pytest.MonkeyPatch, role: str, credentials: dict[str, str]
) -> None:
    url = "s3://private/forecast/v1.icechunk"
    prefix = "s3://source/"
    binding = (
        {"output_secret_name": "missing"}
        if role == "output"
        else {"source_secret_names": {prefix: "missing"}}
    )
    monkeypatch.setenv("REFORMATTERS_ICECHUNK_ACCESS", json.dumps({url: binding}))
    monkeypatch.setattr(
        "reformatters.common.kubernetes.load_secret", Mock(return_value=credentials)
    )
    monkeypatch.setattr(utils.icechunk, "s3_storage", Mock())
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(prefix, icechunk.s3_store(region="us-west-2"))
    )
    monkeypatch.setattr(
        utils.icechunk.Repository, "fetch_config", Mock(return_value=config)
    )
    opened = Mock()
    monkeypatch.setattr(utils.icechunk.Repository, "open", opened)
    with pytest.raises(ValueError if role == "output" else SourceAccessError):
        open_icechunk_readonly(url)
    opened.assert_not_called()


def test_unknown_source_binding_fails_against_materialized_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://public.invalid/v1.icechunk"
    monkeypatch.setenv(
        "REFORMATTERS_ICECHUNK_ACCESS",
        json.dumps({url: {"source_secret_names": {"s3://unknown/": "reader"}}}),
    )
    monkeypatch.setattr(utils.icechunk, "http_storage", Mock())
    monkeypatch.setattr(
        utils.icechunk.Repository, "fetch_config", Mock(return_value=None)
    )
    with pytest.raises(AssertionError, match="Unknown source container prefix"):
        open_icechunk_readonly(url)


def test_icechunk_access_exact_url_and_independent_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REFORMATTERS_ICECHUNK_ACCESS", raising=False)
    assert utils.icechunk_access_for_url("s3://output/repo") == {}
    monkeypatch.setenv(
        "REFORMATTERS_ICECHUNK_ACCESS",
        json.dumps(
            {
                "s3://output/repo/": {
                    "output_secret_name": "writer-ref",
                    "source_secret_names": {"s3://source/": "reader-ref"},
                },
                "https://public.example/repo": {
                    "source_secret_names": {"s3://source/": "other-ref"}
                },
            }
        ),
    )
    access = utils.icechunk_access_for_url("s3://output/repo///")
    assert access["output_secret_name"] == "writer-ref"  # noqa: S105
    assert access["source_secret_names"] == {"s3://source/": "reader-ref"}
    assert utils.icechunk_access_for_url("s3://output/repo/child") == {}
    public = utils.icechunk_access_for_url("https://public.example/repo")
    assert "output_secret_name" not in public
    assert public["source_secret_names"] == {"s3://source/": "other-ref"}


def test_private_storage_options_preserves_transport_loads_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = {
        "access_key_id": "writer-key",
        "secret_access_key": "writer-value",
        "session_token": "writer-token",
        "endpoint_url": "https://output.example",
        "region": "auto",
        "force_path_style": True,
    }
    loader = Mock(return_value=options)
    monkeypatch.setattr(utils.kubernetes, "load_secret", loader)
    assert utils.load_private_storage_options("writer-ref") == options
    loader.assert_called_once_with("writer-ref")


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"access_key_id": "writer-key"},
        {
            "access_key_id": "writer-key",
            "secret_access_key": "writer-value",
            "anonymous": True,
        },
    ],
)
def test_private_storage_options_guards_explicit_private_binding(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]
) -> None:
    loader = Mock(return_value=options)
    monkeypatch.setattr(utils.kubernetes, "load_secret", loader)
    with pytest.raises((ValueError, AssertionError)):
        utils.load_private_storage_options("writer-ref")
    loader.assert_called_once_with("writer-ref")
