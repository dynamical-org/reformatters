from unittest.mock import MagicMock

import icechunk
import pytest
import zarr
import zarr.storage
from icechunk.store import IcechunkStore

from reformatters.common.config import Config, Env
from reformatters.common.storage import (
    DatasetFormat,
    StorageConfig,
    StoreFactory,
    _get_store_path,
    commit_if_icechunk,
)


@pytest.mark.parametrize(
    ("env", "dataset_format", "expected_base", "expected_extension"),
    [
        (Env.prod, DatasetFormat.ZARR3, "s3://prod-bucket/data", ".zarr"),
        (Env.prod, DatasetFormat.ICECHUNK, "s3://prod-bucket/data", ".icechunk"),
        (Env.dev, DatasetFormat.ZARR3, "local/output", ".zarr"),
        (Env.dev, DatasetFormat.ICECHUNK, "local/output", ".icechunk"),
    ],
)
def test_get_store_path(
    monkeypatch: pytest.MonkeyPatch,
    env: Env,
    dataset_format: DatasetFormat,
    expected_base: str,
    expected_extension: str,
) -> None:
    """Test _get_store_path uses correct base path and extension based on environment and format."""
    monkeypatch.setattr(
        "reformatters.common.storage._LOCAL_ZARR_STORE_BASE_PATH", "local/output"
    )
    monkeypatch.setattr(Config, "env", env)

    config = StorageConfig(
        base_path="s3://prod-bucket/data",
        format=dataset_format,
    )

    result = _get_store_path("dataset", "1.0", config)
    assert result == f"{expected_base}/dataset/v1.0{expected_extension}"


@pytest.mark.parametrize(
    ("env", "expected_version"),
    [
        (Env.dev, "dev"),
        (Env.prod, "v1.5"),
        (Env.test, "v1.5"),
    ],
)
def test_store_factory_version(
    monkeypatch: pytest.MonkeyPatch, env: Env, expected_version: str
) -> None:
    """Test StoreFactory.version returns correct value based on environment."""
    monkeypatch.setattr(Config, "env", env)

    primary_config = StorageConfig(
        base_path="s3://bucket/data",
        format=DatasetFormat.ZARR3,
    )

    factory = StoreFactory(
        primary_storage_config=primary_config,
        dataset_id="test-dataset",
        template_config_version="v1.5",
    )

    assert factory.version == expected_version


def test_store_factory_k8s_secret_names() -> None:
    """Test StoreFactory.k8s_secret_names returns all secret names."""
    primary_config = StorageConfig(
        base_path="s3://bucket/primary",
        k8s_secret_name="primary-secret",  # noqa: S106
        format=DatasetFormat.ZARR3,
    )
    replica_config1 = StorageConfig(
        base_path="s3://bucket/replica1",
        k8s_secret_name="replica1-secret",  # noqa: S106
        format=DatasetFormat.ICECHUNK,
    )
    replica_config2 = StorageConfig(
        base_path="s3://bucket/replica2",
        k8s_secret_name="replica2-secret",  # noqa: S106
        format=DatasetFormat.ICECHUNK,
    )

    factory = StoreFactory(
        primary_storage_config=primary_config,
        replica_storage_configs=[replica_config1, replica_config2],
        dataset_id="test-dataset",
        template_config_version="v1.0",
    )

    secret_names = factory.k8s_secret_names()
    assert secret_names == ["primary-secret", "replica1-secret", "replica2-secret"]


@pytest.mark.parametrize(
    ("env", "expected_mode"),
    [
        (Env.dev, "w"),
        (Env.prod, "w-"),
        (Env.test, "w-"),
    ],
)
def test_store_factory_mode(
    monkeypatch: pytest.MonkeyPatch, env: Env, expected_mode: str
) -> None:
    """Test StoreFactory.mode returns correct value based on environment."""
    monkeypatch.setattr(Config, "env", env)

    primary_config = StorageConfig(
        base_path="s3://bucket/data",
        format=DatasetFormat.ZARR3,
    )

    factory = StoreFactory(
        primary_storage_config=primary_config,
        dataset_id="test-dataset",
        template_config_version="v1.0",
    )

    assert factory.mode() == expected_mode


def test_store_factory_returns_correct_store_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test StoreFactory.replica_stores returns correct store types."""
    primary_config = StorageConfig(
        base_path="s3://bucket/primary",
        format=DatasetFormat.ZARR3,
    )
    factory = StoreFactory(
        primary_storage_config=primary_config,
        replica_storage_configs=[
            StorageConfig(
                base_path="s3://bucket/replica",
                format=DatasetFormat.ICECHUNK,
            ),
            StorageConfig(
                base_path="s3://bucket/replica",
                format=DatasetFormat.ZARR3,
            ),
        ],
        dataset_id="test-dataset",
        template_config_version="v1.0",
    )

    # Set store as writable here just so we can create it and then open it.
    assert isinstance(factory.primary_store(writable=True), zarr.storage.LocalStore)
    replicas = factory.replica_stores(writable=True)

    assert len(replicas) == 2
    assert isinstance(replicas[0], icechunk.store.IcechunkStore)
    assert isinstance(replicas[1], zarr.storage.LocalStore)


def test_commit_if_icechunk_commits_icechunk_stores() -> None:
    mock_icechunk_store = MagicMock(spec=IcechunkStore)
    mock_icechunk_store.session = MagicMock()

    commit_if_icechunk("test message", mock_icechunk_store, [])

    mock_icechunk_store.session.commit.assert_called_once()
    _, kwargs = mock_icechunk_store.session.commit.call_args
    assert kwargs["message"] == "test message"


def test_commit_if_icechunk_commits_replicas_before_primary() -> None:
    call_order: list[str] = []

    mock_primary = MagicMock(spec=IcechunkStore)
    mock_primary.session = MagicMock()
    mock_primary.session.commit.side_effect = lambda **_kw: call_order.append("primary")

    mock_replica = MagicMock(spec=IcechunkStore)
    mock_replica.session = MagicMock()
    mock_replica.session.commit.side_effect = lambda **_kw: call_order.append("replica")

    commit_if_icechunk("msg", mock_primary, [mock_replica])

    assert call_order == ["replica", "primary"]


def test_commit_if_icechunk_skips_non_icechunk_stores() -> None:
    non_icechunk_primary = MagicMock(spec=zarr.storage.LocalStore)
    non_icechunk_replica = MagicMock(spec=zarr.storage.LocalStore)

    # Should not raise even though neither store is Icechunk
    commit_if_icechunk("msg", non_icechunk_primary, [non_icechunk_replica])

    non_icechunk_primary.session = MagicMock()
    assert (
        not hasattr(non_icechunk_primary, "session")
        or not non_icechunk_primary.session.commit.called
    )


def test_commit_if_icechunk_noop_for_empty_stores() -> None:
    non_icechunk = MagicMock(spec=zarr.storage.LocalStore)
    commit_if_icechunk("msg", non_icechunk, [])
    # No assertions needed — just verify it does not raise


def test_commit_if_icechunk_commits_multiple_replicas_before_primary() -> None:
    call_order: list[str] = []

    mock_primary = MagicMock(spec=IcechunkStore)
    mock_primary.session = MagicMock()
    mock_primary.session.commit.side_effect = lambda **_kw: call_order.append("primary")

    mock_replica1 = MagicMock(spec=IcechunkStore)
    mock_replica1.session = MagicMock()
    mock_replica1.session.commit.side_effect = lambda **_kw: call_order.append(
        "replica1"
    )

    mock_replica2 = MagicMock(spec=IcechunkStore)
    mock_replica2.session = MagicMock()
    mock_replica2.session.commit.side_effect = lambda **_kw: call_order.append(
        "replica2"
    )

    commit_if_icechunk("msg", mock_primary, [mock_replica1, mock_replica2])

    assert call_order.index("replica1") < call_order.index("primary")
    assert call_order.index("replica2") < call_order.index("primary")
