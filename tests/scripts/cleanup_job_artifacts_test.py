import hashlib
from pathlib import Path

import icechunk
import pytest
from kubernetes.client.exceptions import ApiException

from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from scripts import cleanup_job_artifacts


@pytest.fixture
def store_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StoreFactory:
    monkeypatch.chdir(tmp_path)
    factory = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
        dataset_id="test-dataset",
        template_config_version="1",
    )
    factory.primary_store(writable=True)
    return factory


def _repo(store_factory: StoreFactory) -> icechunk.Repository:
    return store_factory.icechunk_primary_repo()


def _create_artifacts(store_factory: StoreFactory, job_name: str) -> None:
    repo = _repo(store_factory)
    repo.create_branch(f"_job_{job_name}", repo.lookup_branch("main"))
    store_factory.write_coordination_file(job_name, "setup/ready.json", b"{}")


def test_apply_deletes_stale_branch_and_coordination_prefix(
    store_factory: StoreFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _create_artifacts(store_factory, "stale-job")
    monkeypatch.setattr(
        cleanup_job_artifacts.kubernetes, "job_is_active", lambda _: False
    )

    cleanup_job_artifacts.cleanup_job_artifacts(store_factory, apply=True)

    assert set(_repo(store_factory).list_branches()) == {"main"}
    assert store_factory.coordination_file_counts() == {}


def test_apply_refuses_active_job_without_deleting_anything(
    store_factory: StoreFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _create_artifacts(store_factory, "active-job")
    monkeypatch.setattr(
        cleanup_job_artifacts.kubernetes, "job_is_active", lambda _: True
    )

    with pytest.raises(SystemExit, match="still active"):
        cleanup_job_artifacts.cleanup_job_artifacts(store_factory, apply=True)

    assert set(_repo(store_factory).list_branches()) == {"main", "_job_active-job"}
    assert store_factory.coordination_file_counts() == {"active-job": 1}


def test_apply_never_touches_main(
    store_factory: StoreFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _create_artifacts(store_factory, "stale-job")
    main_snapshot = _repo(store_factory).lookup_branch("main")
    monkeypatch.setattr(
        cleanup_job_artifacts.kubernetes, "job_is_active", lambda _: False
    )

    cleanup_job_artifacts.cleanup_job_artifacts(store_factory, apply=True)

    repo = _repo(store_factory)
    assert set(repo.list_branches()) == {"main"}
    assert repo.lookup_branch("main") == main_snapshot


def test_dry_run_deletes_nothing(
    store_factory: StoreFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _create_artifacts(store_factory, "stale-job")
    monkeypatch.setattr(
        cleanup_job_artifacts.kubernetes, "job_is_active", lambda _: False
    )

    cleanup_job_artifacts.cleanup_job_artifacts(store_factory)

    assert set(_repo(store_factory).list_branches()) == {"main", "_job_stale-job"}
    assert store_factory.coordination_file_counts() == {"stale-job": 1}


def test_apply_leaves_non_job_branches_alone(
    store_factory: StoreFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _create_artifacts(store_factory, "stale-job")
    repo = _repo(store_factory)
    repo.create_branch("staging", repo.lookup_branch("main"))
    monkeypatch.setattr(
        cleanup_job_artifacts.kubernetes, "job_is_active", lambda _: False
    )

    cleanup_job_artifacts.cleanup_job_artifacts(store_factory, apply=True)

    assert set(_repo(store_factory).list_branches()) == {"main", "staging"}


@pytest.mark.parametrize("branch", ["main", "staging"])
def test_delete_job_branch_refuses_branches_without_the_job_prefix(
    store_factory: StoreFactory, branch: str
) -> None:
    repo = _repo(store_factory)
    repo.create_branch("staging", repo.lookup_branch("main"))

    with pytest.raises(SystemExit, match="not a disposable job branch"):
        cleanup_job_artifacts._delete_job_branch(repo, branch)

    assert set(_repo(store_factory).list_branches()) == {"main", "staging"}


def test_job_filter_matches_the_full_job_name_not_a_prefix(
    store_factory: StoreFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Job names differ only in a trailing random suffix, so a prefix must not match."""
    _create_artifacts(store_factory, "backfill-2026-09-06t05-00-0-muv9")
    monkeypatch.setattr(
        cleanup_job_artifacts.kubernetes, "job_is_active", lambda _: False
    )

    cleanup_job_artifacts.cleanup_job_artifacts(
        store_factory, job_names=["backfill-2026-09-06t05-00-0"], apply=True
    )

    assert set(_repo(store_factory).list_branches()) == {
        "main",
        "_job_backfill-2026-09-06t05-00-0-muv9",
    }
    assert store_factory.coordination_file_counts() == {
        "backfill-2026-09-06t05-00-0-muv9": 1
    }


def test_apply_stops_when_the_kubernetes_lookup_fails(
    store_factory: StoreFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreachable Kubernetes API must not read as an absent, and so stale, job."""

    def raise_api_error(job_name: str) -> bool:
        raise ApiException(status=500)

    _create_artifacts(store_factory, "stale-job")
    monkeypatch.setattr(
        cleanup_job_artifacts.kubernetes, "job_is_active", raise_api_error
    )

    with pytest.raises(ApiException):
        cleanup_job_artifacts.cleanup_job_artifacts(store_factory, apply=True)

    assert set(_repo(store_factory).list_branches()) == {"main", "_job_stale-job"}
    assert store_factory.coordination_file_counts() == {"stale-job": 1}


def test_dry_run_writes_nothing_to_the_store(
    store_factory: StoreFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _create_artifacts(store_factory, "stale-job")
    monkeypatch.setattr(
        cleanup_job_artifacts.kubernetes, "job_is_active", lambda _: False
    )
    before = _tree_digest(tmp_path)

    cleanup_job_artifacts.cleanup_job_artifacts(store_factory)

    assert _tree_digest(tmp_path) == before


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
