import pickle
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


def _local_factory(
    tmp_path: str, fmt: DatasetFormat = DatasetFormat.ZARR3
) -> StoreFactory:
    return StoreFactory(
        primary_storage_config=StorageConfig(base_path=str(tmp_path), format=fmt),
        dataset_id="test-dataset",
        template_config_version="v1.0",
    )


class TestCoordinationFiles:
    def test_write_and_read_round_trip(self, tmp_path: str) -> None:
        factory = _local_factory(tmp_path)
        factory.write_coordination_file(
            "test-job", "results/worker-0.pkl", pickle.dumps({"a": 1})
        )
        factory.write_coordination_file(
            "test-job", "results/worker-1.pkl", pickle.dumps({"b": 2})
        )

        files = factory.read_all_coordination_files("test-job", "results")
        assert len(files) == 2
        results = [pickle.loads(f) for f in files]  # noqa: S301
        assert {"a": 1} in results
        assert {"b": 2} in results

    def test_read_returns_empty_when_no_files(self, tmp_path: str) -> None:
        factory = _local_factory(tmp_path)
        assert factory.read_all_coordination_files("test-job", "results") == []

    def test_prefix_filtering(self, tmp_path: str) -> None:
        factory = _local_factory(tmp_path)
        factory.write_coordination_file("test-job", "setup/ready.pkl", b"ready")
        factory.write_coordination_file("test-job", "results/worker-0.pkl", b"data")

        assert len(factory.read_all_coordination_files("test-job", "setup")) == 1
        assert len(factory.read_all_coordination_files("test-job", "results")) == 1

    def test_clear_removes_all_files(self, tmp_path: str) -> None:
        factory = _local_factory(tmp_path)
        factory.write_coordination_file("test-job", "results/worker-0.pkl", b"data")
        factory.write_coordination_file("test-job", "setup/ready.pkl", b"ready")

        factory.clear_coordination_files("test-job")

        assert factory.read_all_coordination_files("test-job", "results") == []
        assert factory.read_all_coordination_files("test-job", "setup") == []

    def test_clear_noop_when_no_files(self, tmp_path: str) -> None:
        factory = _local_factory(tmp_path)
        factory.clear_coordination_files("test-job")  # should not raise


class TestIcechunkRepos:
    def test_returns_repos_for_icechunk_stores_only(self) -> None:
        factory = StoreFactory(
            primary_storage_config=StorageConfig(
                base_path="s3://bucket/primary", format=DatasetFormat.ZARR3
            ),
            replica_storage_configs=[
                StorageConfig(
                    base_path="s3://bucket/replica", format=DatasetFormat.ICECHUNK
                ),
            ],
            dataset_id="test-dataset",
            template_config_version="v1.0",
        )
        repos = factory.icechunk_repos()
        assert len(repos) == 1
        assert repos[0][0] == "replica-0"

    def test_returns_empty_for_zarr3_only(self) -> None:
        factory = StoreFactory(
            primary_storage_config=StorageConfig(
                base_path="s3://bucket/primary", format=DatasetFormat.ZARR3
            ),
            dataset_id="test-dataset",
            template_config_version="v1.0",
        )
        assert factory.icechunk_repos() == []

    def test_primary_comes_first(self) -> None:
        factory = StoreFactory(
            primary_storage_config=StorageConfig(
                base_path="s3://bucket/primary", format=DatasetFormat.ICECHUNK
            ),
            replica_storage_configs=[
                StorageConfig(
                    base_path="s3://bucket/replica", format=DatasetFormat.ICECHUNK
                ),
            ],
            dataset_id="test-dataset",
            template_config_version="v1.0",
        )
        repos = factory.icechunk_repos()
        assert len(repos) == 2
        assert repos[0][0] == "primary"
        assert repos[1][0] == "replica-0"


class TestBranchSupport:
    def test_icechunk_store_opens_on_specified_branch(self) -> None:
        factory = StoreFactory(
            primary_storage_config=StorageConfig(
                base_path="s3://bucket/data", format=DatasetFormat.ICECHUNK
            ),
            dataset_id="test-dataset",
            template_config_version="v1.0",
        )
        # Create the store with some data so we can commit
        store = factory.primary_store(writable=True)
        assert isinstance(store, IcechunkStore)
        zarr.open_group(store, mode="w", attributes={"init": True})
        snapshot = store.session.commit(message="init")

        # Create a branch at the current snapshot
        repo = factory.icechunk_repos()[0][1]
        repo.create_branch("test-branch", snapshot)

        # Open on the new branch
        branch_store = factory.primary_store(writable=True, branch="test-branch")
        assert isinstance(branch_store, IcechunkStore)
        assert branch_store.session.branch == "test-branch"

    def test_zarr3_store_ignores_branch_parameter(self) -> None:
        factory = StoreFactory(
            primary_storage_config=StorageConfig(
                base_path="s3://bucket/data", format=DatasetFormat.ZARR3
            ),
            dataset_id="test-dataset",
            template_config_version="v1.0",
        )
        # Should not raise even with a non-main branch
        store = factory.primary_store(writable=True, branch="some-branch")
        assert isinstance(store, zarr.storage.LocalStore)
