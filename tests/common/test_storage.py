import json
from typing import Any
from unittest.mock import MagicMock

import icechunk
import pytest
import zarr
import zarr.storage
from icechunk.store import IcechunkStore

from reformatters.common import retry as retry_module
from reformatters.common.config import Config, Env
from reformatters.common.storage import (
    DatasetFormat,
    StorageConfig,
    StoreFactory,
    _get_store_path,
    _icechunk_to_s3fs_storage_options,
    amend_if_icechunk,
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
            "test-job", "results/worker-0.json", json.dumps({"a": 1}).encode()
        )
        factory.write_coordination_file(
            "test-job", "results/worker-1.json", json.dumps({"b": 2}).encode()
        )

        files = factory.read_all_coordination_files("test-job", "results")
        assert len(files) == 2
        results = [json.loads(f) for f in files]
        assert {"a": 1} in results
        assert {"b": 2} in results

    def test_read_returns_empty_when_no_files(self, tmp_path: str) -> None:
        factory = _local_factory(tmp_path)
        assert factory.read_all_coordination_files("test-job", "results") == []

    def test_prefix_filtering(self, tmp_path: str) -> None:
        factory = _local_factory(tmp_path)
        factory.write_coordination_file("test-job", "setup/ready.json", b"ready")
        factory.write_coordination_file("test-job", "results/worker-0.json", b"data")

        assert len(factory.read_all_coordination_files("test-job", "setup")) == 1
        assert len(factory.read_all_coordination_files("test-job", "results")) == 1

    def test_clear_removes_all_files(self, tmp_path: str) -> None:
        factory = _local_factory(tmp_path)
        factory.write_coordination_file("test-job", "results/worker-0.json", b"data")
        factory.write_coordination_file("test-job", "setup/ready.json", b"ready")

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
        repos = factory.icechunk_repos(sort="primary-first")
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
        assert factory.icechunk_repos(sort="primary-first") == []

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
        repos = factory.icechunk_repos(sort="primary-first")
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
        repo = factory.icechunk_repos(sort="primary-first")[0][1]
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


@pytest.mark.parametrize(
    "env",
    [Env.prod, Env.dev, Env.test],
)
@pytest.mark.parametrize(
    "dataset_format",
    [DatasetFormat.ZARR3, DatasetFormat.ICECHUNK],
)
def test_coordination_base_path_ends_in_internal(
    monkeypatch: pytest.MonkeyPatch,
    env: Env,
    dataset_format: DatasetFormat,
) -> None:
    """_coordination_base_path must end with `/_internal` in every environment
    and format — downstream coordination key paths assume it."""
    monkeypatch.setattr(Config, "env", env)
    factory = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path="s3://bucket/prefix", format=dataset_format
        ),
        dataset_id="test-dataset",
        template_config_version="v1.0",
    )
    assert factory._coordination_base_path().endswith("/_internal")


def test_clear_coordination_files_rm_path_rooted_in_internal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """clear_coordination_files must only rm paths inside `/_internal/` — a
    safety boundary that prevents accidentally deleting dataset zarr data."""
    factory = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path="s3://bucket/prefix", format=DatasetFormat.ICECHUNK
        ),
        dataset_id="test-dataset",
        template_config_version="v1.0",
    )
    mock_fs = MagicMock()
    monkeypatch.setattr(factory, "_coordination_fs", lambda: mock_fs)

    factory.clear_coordination_files("my-job")

    mock_fs.rm.assert_called_once()
    path = mock_fs.rm.call_args.args[0]
    assert path.endswith("/_internal/my-job"), path


class TestIcechunkPrimaryWithZarr3Replica:
    """Primary=icechunk + replica=zarr3 is a supported staging configuration.
    `icechunk_repos` must return only the primary icechunk repo so
    `commit_if_icechunk` in finalize touches the primary but not the zarr3 replica."""

    def test_only_primary_returned_when_replica_is_zarr3(self) -> None:
        factory = StoreFactory(
            primary_storage_config=StorageConfig(
                base_path="s3://bucket/primary", format=DatasetFormat.ICECHUNK
            ),
            replica_storage_configs=[
                StorageConfig(
                    base_path="s3://bucket/replica", format=DatasetFormat.ZARR3
                ),
            ],
            dataset_id="test-dataset",
            template_config_version="v1.0",
        )
        repos = factory.icechunk_repos(sort="primary-first")
        assert len(repos) == 1
        assert repos[0][0] == "primary"

    def test_primary_last_sort_with_single_primary(self) -> None:
        """Single-primary icechunk config: `primary-last` still returns the
        primary (nothing in front of it). Guards the sort key logic at the
        boundary case of one element."""
        factory = StoreFactory(
            primary_storage_config=StorageConfig(
                base_path="s3://bucket/primary", format=DatasetFormat.ICECHUNK
            ),
            dataset_id="test-dataset",
            template_config_version="v1.0",
        )
        repos = factory.icechunk_repos(sort="primary-last")
        assert [role for role, _repo in repos] == ["primary"]


class TestCommitIfIcechunkFailureIsolation:
    """When a replica commit fails permanently, the primary MUST NOT commit.
    Otherwise we'd have primary ahead of replica — readers of the replica
    would miss data that primary's future work assumes is already present."""

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # retry() sleeps with exponential backoff between attempts — skip the
        # waits so 10-attempt retry tests stay fast.
        monkeypatch.setattr(retry_module.time, "sleep", lambda *_a, **_kw: None)

    def test_primary_not_committed_when_replica_commit_fails(self) -> None:
        primary = MagicMock(spec=IcechunkStore)
        primary.session = MagicMock()

        replica = MagicMock(spec=IcechunkStore)
        replica.session = MagicMock()
        replica.session.commit.side_effect = RuntimeError("replica unreachable")

        with pytest.raises(RuntimeError, match="replica unreachable"):
            commit_if_icechunk("msg", primary, [replica])

        # retry() calls the function max_attempts times before re-raising.
        assert replica.session.commit.call_count == 10
        primary.session.commit.assert_not_called()

    def test_transient_replica_failure_retried_then_primary_commits(self) -> None:
        primary = MagicMock(spec=IcechunkStore)
        primary.session = MagicMock()

        attempts = {"n": 0}

        def flaky_commit(**_kw: object) -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("transient")
            return "snap-ok"

        replica = MagicMock(spec=IcechunkStore)
        replica.session = MagicMock()
        replica.session.commit.side_effect = flaky_commit

        commit_if_icechunk("msg", primary, [replica])

        assert replica.session.commit.call_count == 3
        primary.session.commit.assert_called_once()


def test_amend_if_icechunk_amends_icechunk_stores() -> None:
    mock_icechunk_store = MagicMock(spec=IcechunkStore)
    mock_icechunk_store.session = MagicMock()

    amend_if_icechunk("test message", mock_icechunk_store, [])

    mock_icechunk_store.session.amend.assert_called_once()
    _, kwargs = mock_icechunk_store.session.amend.call_args
    assert kwargs["message"] == "test message"


def test_amend_if_icechunk_amends_replicas_before_primary() -> None:
    call_order: list[str] = []

    mock_primary = MagicMock(spec=IcechunkStore)
    mock_primary.session = MagicMock()
    mock_primary.session.amend.side_effect = lambda **_kw: call_order.append("primary")

    mock_replica = MagicMock(spec=IcechunkStore)
    mock_replica.session = MagicMock()
    mock_replica.session.amend.side_effect = lambda **_kw: call_order.append("replica")

    amend_if_icechunk("msg", mock_primary, [mock_replica])

    assert call_order == ["replica", "primary"]


def test_amend_if_icechunk_skips_non_icechunk_stores() -> None:
    non_icechunk_primary = MagicMock(spec=zarr.storage.LocalStore)
    non_icechunk_replica = MagicMock(spec=zarr.storage.LocalStore)

    # Should not raise even though neither store is Icechunk
    amend_if_icechunk("msg", non_icechunk_primary, [non_icechunk_replica])

    non_icechunk_primary.session = MagicMock()
    assert (
        not hasattr(non_icechunk_primary, "session")
        or not non_icechunk_primary.session.amend.called
    )


def test_amend_if_icechunk_noop_for_empty_stores() -> None:
    non_icechunk = MagicMock(spec=zarr.storage.LocalStore)
    amend_if_icechunk("msg", non_icechunk, [])
    # No assertions needed — just verify it does not raise


def test_amend_if_icechunk_amends_multiple_replicas_before_primary() -> None:
    call_order: list[str] = []

    mock_primary = MagicMock(spec=IcechunkStore)
    mock_primary.session = MagicMock()
    mock_primary.session.amend.side_effect = lambda **_kw: call_order.append("primary")

    mock_replica1 = MagicMock(spec=IcechunkStore)
    mock_replica1.session = MagicMock()
    mock_replica1.session.amend.side_effect = lambda **_kw: call_order.append(
        "replica1"
    )

    mock_replica2 = MagicMock(spec=IcechunkStore)
    mock_replica2.session = MagicMock()
    mock_replica2.session.amend.side_effect = lambda **_kw: call_order.append(
        "replica2"
    )

    amend_if_icechunk("msg", mock_primary, [mock_replica1, mock_replica2])

    assert call_order.index("replica1") < call_order.index("primary")
    assert call_order.index("replica2") < call_order.index("primary")


class TestAmendIfIcechunkConflictHandling:
    """ConflictError is handled by rebasing once with ConflictDetector and
    retrying the amend, all inside a single retry attempt. Non-conflict
    failures use the outer retry(max_attempts=10) just like commit_if_icechunk."""

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(retry_module.time, "sleep", lambda *_a, **_kw: None)

    def test_rebases_and_retries_amend_on_conflict_error(self) -> None:
        store = MagicMock(spec=IcechunkStore)
        store.session = MagicMock()
        conflict = icechunk.ConflictError("snap-expected", "snap-actual")
        store.session.amend.side_effect = [conflict, "snap-ok"]

        amend_if_icechunk("msg", store, [])

        assert store.session.amend.call_count == 2
        store.session.rebase.assert_called_once()
        # ConflictDetector is the configured solver — workers write disjoint
        # chunks so the solver must reject genuine concurrent edits.
        (solver,) = store.session.rebase.call_args.args
        assert isinstance(solver, icechunk.ConflictDetector)

    def test_does_not_rebase_when_first_amend_succeeds(self) -> None:
        store = MagicMock(spec=IcechunkStore)
        store.session = MagicMock()
        store.session.amend.return_value = "snap-ok"

        amend_if_icechunk("msg", store, [])

        store.session.amend.assert_called_once()
        store.session.rebase.assert_not_called()

    def test_outer_retry_runs_up_to_10_times_on_non_conflict_failure(self) -> None:
        store = MagicMock(spec=IcechunkStore)
        store.session = MagicMock()
        store.session.amend.side_effect = RuntimeError("transient")

        with pytest.raises(RuntimeError, match="transient"):
            amend_if_icechunk("msg", store, [])

        # retry() runs the inner function 10 times; rebase is never reached
        # because RuntimeError is not a ConflictError.
        assert store.session.amend.call_count == 10
        store.session.rebase.assert_not_called()

    def test_transient_failure_retried_then_succeeds(self) -> None:
        store = MagicMock(spec=IcechunkStore)
        store.session = MagicMock()
        attempts = {"n": 0}

        def flaky_amend(**_kw: object) -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("transient")
            return "snap-ok"

        store.session.amend.side_effect = flaky_amend
        amend_if_icechunk("msg", store, [])
        assert store.session.amend.call_count == 3


class TestAmendIfIcechunkFailureIsolation:
    """Same invariant as commit_if_icechunk: a permanently-failing replica must
    not leave the primary ahead, since primary drives future work."""

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(retry_module.time, "sleep", lambda *_a, **_kw: None)

    def test_primary_not_amended_when_replica_amend_fails(self) -> None:
        primary = MagicMock(spec=IcechunkStore)
        primary.session = MagicMock()

        replica = MagicMock(spec=IcechunkStore)
        replica.session = MagicMock()
        replica.session.amend.side_effect = RuntimeError("replica unreachable")

        with pytest.raises(RuntimeError, match="replica unreachable"):
            amend_if_icechunk("msg", primary, [replica])

        assert replica.session.amend.call_count == 10
        primary.session.amend.assert_not_called()


class TestPrimaryStoreReadonly:
    """`primary_store(writable=False)` opens a readonly icechunk session.
    Tested end-to-end for writable=True already; readonly is exercised by
    `all_stores_exist` and validators and was previously uncovered."""

    def test_readonly_session_opens_on_specified_branch(self) -> None:
        factory = StoreFactory(
            primary_storage_config=StorageConfig(
                base_path="s3://bucket/data", format=DatasetFormat.ICECHUNK
            ),
            dataset_id="test-dataset",
            template_config_version="v1.0",
        )
        # Initialize the repo by opening writable first.
        writable = factory.primary_store(writable=True)
        assert isinstance(writable, IcechunkStore)
        zarr.open_group(writable, mode="w", attributes={"v": 1})
        branch_snapshot = writable.session.commit(message="init")

        # Create a branch so we can open readonly on something other than main.
        # Point it at the initial snapshot so we can distinguish it from main
        # after a subsequent commit on main.
        repo = factory.icechunk_repos(sort="primary-first")[0][1]
        repo.create_branch("ro-branch", branch_snapshot)
        # Advance main past the branch point.
        writable_again = factory.primary_store(writable=True)
        assert isinstance(writable_again, IcechunkStore)
        zarr.open_group(writable_again, mode="w", attributes={"v": 2})
        main_snapshot = writable_again.session.commit(message="advance main")
        assert main_snapshot != branch_snapshot

        readonly = factory.primary_store(writable=False, branch="ro-branch")
        assert isinstance(readonly, IcechunkStore)
        assert readonly.read_only
        # Readonly session resolves the branch to its snapshot id at open time.
        assert readonly.session.snapshot_id == branch_snapshot


class TestIcechunkToS3fsStorageOptions:
    """Icechunk secrets are keyed for `icechunk.s3_storage(**options)` but
    coordination files on an icechunk primary go through fsspec/s3fs, which
    uses different option names. The translation keeps those two consumers
    of the same secret in sync."""

    def test_translates_credential_keys(self) -> None:
        assert _icechunk_to_s3fs_storage_options(
            {
                "access_key_id": "AKIA",
                "secret_access_key": "shh",
                "session_token": "tok",
            }
        ) == {"key": "AKIA", "secret": "shh", "token": "tok"}

    def test_region_moves_into_client_kwargs(self) -> None:
        assert _icechunk_to_s3fs_storage_options({"region": "us-east-1"}) == {
            "client_kwargs": {"region_name": "us-east-1"}
        }

    def test_region_merges_with_existing_client_kwargs(self) -> None:
        assert _icechunk_to_s3fs_storage_options(
            {"region": "us-east-1", "client_kwargs": {"verify": False}}
        ) == {"client_kwargs": {"region_name": "us-east-1", "verify": False}}

    def test_unknown_keys_pass_through(self) -> None:
        assert _icechunk_to_s3fs_storage_options({"endpoint_url": "https://x"}) == {
            "endpoint_url": "https://x"
        }

    def test_empty_options(self) -> None:
        assert _icechunk_to_s3fs_storage_options({}) == {}


class TestCoordinationFsStorageOptions:
    """`_coordination_fs` must pass s3fs-compatible kwargs to `fsspec.filesystem`
    regardless of which dialect the primary storage config's secret uses.
    Without translation, an icechunk-style secret (access_key_id, ...) causes
    s3fs/aiobotocore to raise `TypeError: AioSession.__init__() got an
    unexpected keyword argument 'access_key_id'`."""

    def _capture_filesystem_kwargs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        primary_format: DatasetFormat,
        storage_options: dict[str, Any],
    ) -> dict[str, Any]:
        monkeypatch.setattr(Config, "env", Env.prod)
        factory = StoreFactory(
            primary_storage_config=StorageConfig(
                base_path="s3://bucket/data", format=primary_format
            ),
            dataset_id="test-dataset",
            template_config_version="v1.0",
        )
        monkeypatch.setattr(
            StorageConfig,
            "load_storage_options",
            lambda self: storage_options,
        )
        captured: dict[str, Any] = {}

        def fake_filesystem(protocol: str, **kwargs: Any) -> MagicMock:  # noqa: ANN401
            captured["protocol"] = protocol
            captured["kwargs"] = kwargs
            return MagicMock()

        monkeypatch.setattr(
            "reformatters.common.storage.fsspec.filesystem", fake_filesystem
        )
        factory._coordination_fs()
        return captured

    def test_icechunk_primary_translates_to_s3fs_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._capture_filesystem_kwargs(
            monkeypatch,
            primary_format=DatasetFormat.ICECHUNK,
            storage_options={
                "access_key_id": "AKIA",
                "secret_access_key": "shh",
                "region": "us-east-1",
            },
        )
        assert captured["protocol"] == "s3"
        kwargs = captured["kwargs"]
        # icechunk-style keys must not leak through
        assert "access_key_id" not in kwargs
        assert "secret_access_key" not in kwargs
        assert "region" not in kwargs
        assert kwargs["key"] == "AKIA"
        assert kwargs["secret"] == "shh"  # noqa: S105
        assert kwargs["client_kwargs"] == {"region_name": "us-east-1"}

    def test_zarr3_primary_passes_options_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Zarr3 secrets are already s3fs-shaped — don't rewrite them.
        options = {"key": "AKIA", "secret": "shh", "endpoint_url": "https://s"}
        captured = self._capture_filesystem_kwargs(
            monkeypatch,
            primary_format=DatasetFormat.ZARR3,
            storage_options=options,
        )
        assert captured["protocol"] == "s3"
        assert captured["kwargs"] == options


class TestAllStoresExistWithIcechunk:
    """`all_stores_exist` is used as a precondition for backfills with
    `overwrite_existing=True`. It should return True for a populated icechunk
    primary and False for one that doesn't yet exist on disk."""

    def test_returns_true_for_initialized_icechunk_primary(self) -> None:
        factory = StoreFactory(
            primary_storage_config=StorageConfig(
                base_path="s3://bucket/data", format=DatasetFormat.ICECHUNK
            ),
            dataset_id="test-dataset",
            template_config_version="v1.0",
        )
        # Initialize the store so xr.open_zarr can read it.
        store = factory.primary_store(writable=True)
        assert isinstance(store, IcechunkStore)
        zarr.open_group(store, mode="w", attributes={"init": True})
        store.session.commit(message="init")

        assert factory.all_stores_exist() is True
