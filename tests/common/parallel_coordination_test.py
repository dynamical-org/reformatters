"""Fast unit tests for parallel_coordination.

Exercises the coordination logic in isolation by stubbing out zarr/icechunk
I/O with fakes and monkeypatching. End-to-end coverage — including real
zarr writes, real icechunk sessions, and worker restart semantics —
lives in tests/common/parallel_writes_test.py.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import parallel_coordination as pc
from reformatters.common.region_job import SourceFileResult, SourceFileStatus


class FakeStoreFactory:
    """In-memory stand-in for StoreFactory's coordination-file + store APIs."""

    def __init__(self) -> None:
        self.files: dict[str, dict[str, bytes]] = {}
        self._icechunk_repos_by_sort: dict[str, list[tuple[str, FakeRepo]]] = {
            "primary-first": [],
            "primary-last": [],
        }
        self._primary_store: object = MagicMock(name="primary_zarr3_store")
        self._replica_stores: list[object] = []
        self.published_tree: xr.DataTree = xr.DataTree.from_dict({"/": xr.Dataset()})
        self.persist_virtual_config_calls = 0

    def persist_virtual_config(self) -> None:
        self.persist_virtual_config_calls += 1

    def set_icechunk_repos(self, ordered: list[tuple[str, FakeRepo]]) -> None:
        self._icechunk_repos_by_sort["primary-first"] = list(ordered)
        self._icechunk_repos_by_sort["primary-last"] = list(reversed(ordered))

    def write_coordination_file(self, job_name: str, key: str, data: bytes) -> None:
        self.files.setdefault(job_name, {})[key] = data

    def read_all_coordination_files(self, job_name: str, prefix: str) -> list[bytes]:
        files = self.files.get(job_name, {})
        matching = {k: v for k, v in files.items() if k.startswith(f"{prefix}/")}
        return [matching[k] for k in sorted(matching)]

    def list_coordination_files(self, job_name: str, prefix: str) -> list[str]:
        files = self.files.get(job_name, {})
        return sorted(
            key.rsplit("/", 1)[-1] for key in files if key.startswith(f"{prefix}/")
        )

    def clear_coordination_files(self, job_name: str) -> None:
        self.files.pop(job_name, None)

    def icechunk_repos(self, *, sort: str) -> list[tuple[str, FakeRepo]]:
        return list(self._icechunk_repos_by_sort[sort])

    def primary_store(self, writable: bool = False) -> object:  # noqa: ARG002
        return self._primary_store

    def open_primary_datatree(self) -> xr.DataTree:
        return self.published_tree

    def replica_stores(self, writable: bool = False) -> list[object]:  # noqa: ARG002
        return list(self._replica_stores)


class FakeSession:
    _counter = 0

    def __init__(self, branch: str, repo: FakeRepo) -> None:
        FakeSession._counter += 1
        self.id = FakeSession._counter
        self.branch = branch
        self.repo = repo
        self.store: object = MagicMock(name=f"ic_store-{self.id}")
        self.commit_calls: list[tuple[str, object]] = []

    def commit(self, message: str, rebase_with: object = None) -> str:
        self.commit_calls.append((message, rebase_with))
        new_snapshot = f"snap-{self.id}-{len(self.commit_calls)}"
        self.repo._branches[self.branch] = new_snapshot
        return new_snapshot


class FakeRepo:
    def __init__(self, initial_main: str = "snap-initial") -> None:
        self._branches: dict[str, str] = {"main": initial_main}
        self.sessions: list[FakeSession] = []
        self.reset_calls: list[tuple[str, str, str]] = []
        self.create_branch_calls: list[tuple[str, str]] = []
        self.delete_branch_calls: list[str] = []
        # Newest-first commit history per branch, for finalize's retry detection.
        self.ancestry_by_branch: dict[str, list[str]] = {}

    def lookup_branch(self, name: str) -> str:
        return self._branches[name]

    def ancestry(self, *, branch: str) -> list[SimpleNamespace]:
        snapshots = self.ancestry_by_branch.get(branch, [self._branches[branch]])
        return [SimpleNamespace(id=snapshot) for snapshot in snapshots]

    def list_branches(self) -> list[str]:
        return list(self._branches)

    def create_branch(self, name: str, snapshot: str) -> None:
        self.create_branch_calls.append((name, snapshot))
        self._branches[name] = snapshot

    def delete_branch(self, name: str) -> None:
        self.delete_branch_calls.append(name)
        del self._branches[name]

    def writable_session(self, branch: str) -> FakeSession:
        session = FakeSession(branch, self)
        self.sessions.append(session)
        return session

    def reset_branch(self, name: str, snapshot: str, from_snapshot_id: str) -> None:
        self.reset_calls.append((name, snapshot, from_snapshot_id))
        self._branches[name] = snapshot


@pytest.fixture(autouse=True)
def stub_io(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Replace all real I/O called from parallel_coordination with mocks.

    Autouse so real write_metadata / copy_zarr_metadata / commit_if_icechunk
    never run during unit tests. Tests that want to inspect call history
    request this fixture by name.
    """
    write_metadata = MagicMock(name="write_metadata")
    commit_if_icechunk = MagicMock(name="commit_if_icechunk")
    copy_zarr_metadata = MagicMock(name="copy_zarr_metadata")
    monkeypatch.setattr(pc.template_utils, "write_metadata", write_metadata)
    monkeypatch.setattr(pc.storage, "commit_if_icechunk", commit_if_icechunk)
    monkeypatch.setattr(pc, "copy_zarr_metadata", copy_zarr_metadata)
    return {
        "write_metadata": write_metadata,
        "commit_if_icechunk": commit_if_icechunk,
        "copy_zarr_metadata": copy_zarr_metadata,
    }


def _template() -> xr.DataTree:
    # An empty dataset is enough — every real use of template_ds is stubbed.
    return xr.DataTree.from_dict({"/": xr.Dataset()})


def _time_tree(n_time: int) -> xr.DataTree:
    times = pd.date_range("2026-04-01", periods=n_time, freq="D")
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(
                {"v": xr.DataArray(np.zeros(n_time), dims=("time",))},
                coords={"time": times},
            )
        }
    )


class TestParallelSetupFirstWorker:
    def test_single_worker_zarr3_writes_metadata_locally_and_skips_ready_json(
        self, tmp_path: Path, stub_io: dict[str, MagicMock]
    ) -> None:
        factory = FakeStoreFactory()
        ds = _template()

        result = pc.parallel_setup(
            factory,  # ty: ignore[invalid-argument-type]
            is_first=True,
            workers_total=1,
            reformat_job_name="job",
            branch_name="temp",
            template_ds=ds,
            tmp_store=tmp_path,
            icechunk_repos=[],
            consolidated=True,
        )

        stub_io["write_metadata"].assert_called_once_with(
            ds, tmp_path, consolidated=True
        )
        # Single-worker: no ready.json is written.
        assert factory.files == {}
        assert result == {}

    def test_multi_worker_zarr3_writes_empty_ready_json(
        self, tmp_path: Path, stub_io: dict[str, MagicMock]
    ) -> None:
        factory = FakeStoreFactory()

        result = pc.parallel_setup(
            factory,  # ty: ignore[invalid-argument-type]
            is_first=True,
            workers_total=3,
            reformat_job_name="job",
            branch_name="temp",
            template_ds=_template(),
            tmp_store=tmp_path,
            icechunk_repos=[],
            consolidated=True,
        )

        assert result == {}
        ready = factory.files["job"]["setup/ready.json"]
        assert json.loads(ready) == {}
        stub_io["commit_if_icechunk"].assert_not_called()
        stub_io["copy_zarr_metadata"].assert_not_called()

    def test_icechunk_creates_branch_copies_metadata_and_commits(
        self, tmp_path: Path, stub_io: dict[str, MagicMock]
    ) -> None:
        factory = FakeStoreFactory()
        primary_repo = FakeRepo(initial_main="snap-primary-init")
        replica_repo = FakeRepo(initial_main="snap-replica-init")
        repos = [("primary", primary_repo), ("replica-0", replica_repo)]
        ds = _template()

        result = pc.parallel_setup(
            factory,  # ty: ignore[invalid-argument-type]
            is_first=True,
            workers_total=2,
            reformat_job_name="job",
            branch_name="temp-branch",
            template_ds=ds,
            tmp_store=tmp_path,
            icechunk_repos=repos,  # ty: ignore[invalid-argument-type]
            consolidated=True,
        )

        # Each repo had create_branch called with the main snapshot captured.
        assert primary_repo.create_branch_calls == [
            ("temp-branch", "snap-primary-init")
        ]
        assert replica_repo.create_branch_calls == [
            ("temp-branch", "snap-replica-init")
        ]

        # copy_zarr_metadata called once per ic_store (primary + replica).
        assert stub_io["copy_zarr_metadata"].call_count == 2
        primary_session = primary_repo.sessions[0]
        replica_session = replica_repo.sessions[0]
        assert stub_io["copy_zarr_metadata"].call_args_list[0].args == (
            ds,
            tmp_path,
            primary_session.store,
        )

        # commit_if_icechunk called once with primary as the primary_store and
        # replica as the lone entry in replicas.
        stub_io["commit_if_icechunk"].assert_called_once_with(
            "Expand dataset",
            primary_session.store,
            [replica_session.store],
        )

        # Worker 0 persists the virtual container config once after expanding.
        assert factory.persist_virtual_config_calls == 1

        # ready.json records the snapshot per role.
        ready = json.loads(factory.files["job"]["setup/ready.json"])
        assert ready == {
            "repo_snapshots": {
                "primary": "snap-primary-init",
                "replica-0": "snap-replica-init",
            }
        }
        # Returned SetupInfo reflects what was written.
        assert result == ready

    def test_icechunk_retry_preserves_snapshot_from_ready_json(
        self, tmp_path: Path
    ) -> None:
        factory = FakeStoreFactory()
        primary_repo = FakeRepo(initial_main="snap-primary-different-on-retry")
        repos = [("primary", primary_repo)]
        sentinel = "SENTINEL_FROM_ATTEMPT_1"
        factory.write_coordination_file(
            "job",
            "setup/ready.json",
            json.dumps({"repo_snapshots": {"primary": sentinel}}).encode(),
        )

        result = pc.parallel_setup(
            factory,  # ty: ignore[invalid-argument-type]
            is_first=True,
            workers_total=2,
            reformat_job_name="job",
            branch_name="temp-branch",
            template_ds=_template(),
            tmp_store=tmp_path,
            icechunk_repos=repos,  # ty: ignore[invalid-argument-type]
            consolidated=True,
        )

        # setdefault must preserve the prior-attempt snapshot, not refresh from main.
        assert result["repo_snapshots"]["primary"] == sentinel
        ready = json.loads(factory.files["job"]["setup/ready.json"])
        assert ready["repo_snapshots"]["primary"] == sentinel
        # create_branch still called, with the preserved sentinel as snapshot.
        assert primary_repo.create_branch_calls == [("temp-branch", sentinel)]

    def test_icechunk_primary_only_passes_empty_replicas(
        self, tmp_path: Path, stub_io: dict[str, MagicMock]
    ) -> None:
        """When the StoreFactory has only an icechunk primary (no icechunk
        replicas), parallel_setup must still create the branch on primary and
        call commit_if_icechunk with an empty replicas list."""
        factory = FakeStoreFactory()
        primary_repo = FakeRepo(initial_main="snap-primary-init")

        result = pc.parallel_setup(
            factory,  # ty: ignore[invalid-argument-type]
            is_first=True,
            workers_total=2,
            reformat_job_name="job",
            branch_name="temp-branch",
            template_ds=_template(),
            tmp_store=tmp_path,
            icechunk_repos=[("primary", primary_repo)],  # ty: ignore[invalid-argument-type]
            consolidated=True,
        )

        assert primary_repo.create_branch_calls == [
            ("temp-branch", "snap-primary-init")
        ]
        primary_session = primary_repo.sessions[0]
        stub_io["commit_if_icechunk"].assert_called_once_with(
            "Expand dataset",
            primary_session.store,
            [],
        )
        assert result["repo_snapshots"] == {"primary": "snap-primary-init"}

    def test_icechunk_retry_reuses_existing_branch(self, tmp_path: Path) -> None:
        factory = FakeStoreFactory()
        primary_repo = FakeRepo(initial_main="snap-primary-init")
        # Pretend an earlier worker 0 attempt already created the temp branch.
        primary_repo._branches["temp-branch"] = "snap-from-earlier-attempt"
        repos = [("primary", primary_repo)]

        pc.parallel_setup(
            factory,  # ty: ignore[invalid-argument-type]
            is_first=True,
            workers_total=2,
            reformat_job_name="job",
            branch_name="temp-branch",
            template_ds=_template(),
            tmp_store=tmp_path,
            icechunk_repos=repos,  # ty: ignore[invalid-argument-type]
            consolidated=True,
        )

        assert primary_repo.create_branch_calls == []


class TestParallelSetupLaterWorker:
    def test_single_worker_path_returns_empty_without_polling(
        self, tmp_path: Path, stub_io: dict[str, MagicMock]
    ) -> None:
        factory = FakeStoreFactory()
        # If the code polled, it would read from an empty store_factory and
        # spin forever — so reaching an empty return proves it did not poll.
        result = pc.parallel_setup(
            factory,  # ty: ignore[invalid-argument-type]
            is_first=False,
            workers_total=1,
            reformat_job_name="job",
            branch_name="temp",
            template_ds=_template(),
            tmp_store=tmp_path,
            icechunk_repos=[],
            consolidated=True,
        )
        assert result == {}
        stub_io["write_metadata"].assert_not_called()

    def test_multi_worker_polls_until_ready_json_appears(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        factory = FakeStoreFactory()
        payload = {"repo_snapshots": {"primary": "snap-A"}}
        sleep_calls: list[float] = []

        def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            # Worker 0 finishes after the first poll.
            if len(sleep_calls) == 1:
                factory.write_coordination_file(
                    "job", "setup/ready.json", json.dumps(payload).encode()
                )

        monkeypatch.setattr(pc.time, "sleep", fake_sleep)

        result = pc.parallel_setup(
            factory,  # ty: ignore[invalid-argument-type]
            is_first=False,
            workers_total=3,
            reformat_job_name="job",
            branch_name="temp",
            template_ds=_template(),
            tmp_store=tmp_path,
            icechunk_repos=[],
            consolidated=True,
        )

        assert result == payload
        assert sleep_calls == [5]


class TestWaitForWorkers:
    def test_single_worker_returns_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory = FakeStoreFactory()
        # If time.sleep were called we'd notice — raise to fail loudly.
        monkeypatch.setattr(pc.time, "sleep", lambda *_: pytest.fail("should not poll"))
        pc.wait_for_workers(factory, "job", workers_total=1)  # ty: ignore[invalid-argument-type]

    def test_polls_until_all_results_present(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        factory = FakeStoreFactory()
        factory.write_coordination_file("job", "results/worker-0.json", b"x")
        sleep_calls: list[float] = []

        def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            factory.write_coordination_file(
                "job", f"results/worker-{len(sleep_calls)}.json", b"x"
            )

        monkeypatch.setattr(pc.time, "sleep", fake_sleep)

        with caplog.at_level(logging.INFO):
            pc.wait_for_workers(factory, "job", workers_total=3)  # ty: ignore[invalid-argument-type]

        # Started with 1 file, needs 3 → 2 polls.
        assert sleep_calls == [10, 10]
        assert "Waiting for 2 of 3 workers" in caplog.text
        assert "missing worker indexes: [1, 2]" in caplog.text

    def test_missing_worker_log_caps_indexes(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # More workers than the 50-index cap, so the logged list is really cut.
        workers_total = 60
        factory = FakeStoreFactory()

        def fake_sleep(_: float) -> None:
            for worker_index in range(workers_total):
                factory.write_coordination_file(
                    "job", f"results/worker-{worker_index}.json", b"x"
                )

        monkeypatch.setattr(pc.time, "sleep", fake_sleep)

        with caplog.at_level(logging.INFO):
            pc.wait_for_workers(factory, "job", workers_total=workers_total)  # ty: ignore[invalid-argument-type]

        # Exactly one poll logs, since fake_sleep completes every worker.
        (message,) = [m for m in caplog.messages if "missing worker indexes" in m]
        # The count reports all 60 missing while the list stops at 50 — asserting
        # both is what pins the cap. Match on the message, not caplog.text, whose
        # "file.py:lineno" prefix can itself contain the digits under test.
        assert f"Waiting for {workers_total} of {workers_total} workers" in message
        assert f"missing worker indexes: {list(range(50))}" in message
        # Index 50 is the first past the cap and must not be logged.
        assert "50" not in message


class TestCollectResults:
    def test_merges_json_results_across_workers(self) -> None:
        factory = FakeStoreFactory()

        def r(ts: str, td: str, url: str) -> SourceFileResult:
            return SourceFileResult(
                status=SourceFileStatus.Succeeded,
                out_loc={"init_time": pd.Timestamp(ts), "lead_time": pd.Timedelta(td)},
                url=url,
            )

        worker_0 = {
            "var_a": [r("2026-04-15", "0h", "u0"), r("2026-04-15", "6h", "u1")],
            "var_b": [r("2026-04-15", "0h", "u2")],
        }
        worker_1 = {
            "var_a": [r("2026-04-15", "12h", "u3")],
            "var_c": [r("2026-04-15", "0h", "u4")],
        }
        factory.write_coordination_file(
            "job", "results/worker-0.json", pc.dump_worker_results_json(worker_0)
        )
        factory.write_coordination_file(
            "job", "results/worker-1.json", pc.dump_worker_results_json(worker_1)
        )

        merged = pc.collect_results(factory, "job", workers_total=2)  # ty: ignore[invalid-argument-type]

        # URLs identify each result unambiguously; check the merged set per var.
        merged_urls = {v: [r.url for r in rs] for v, rs in merged.items()}
        assert merged_urls == {
            "var_a": ["u0", "u1", "u3"],
            "var_b": ["u2"],
            "var_c": ["u4"],
        }
        # Spot-check that pandas types round-tripped.
        assert merged["var_a"][2].out_loc["lead_time"] == pd.Timedelta("12h")
        assert merged["var_a"][2].out_loc["init_time"] == pd.Timestamp("2026-04-15")


class TestFinalize:
    def test_branch_main_skips_icechunk_finalize(
        self, tmp_path: Path, stub_io: dict[str, MagicMock]
    ) -> None:
        factory = FakeStoreFactory()
        # Populate fake repos to prove they are never consulted when branch=="main".
        factory.set_icechunk_repos([("primary", FakeRepo()), ("replica-0", FakeRepo())])

        pc.finalize(
            factory,  # ty: ignore[invalid-argument-type]
            all_jobs=[],
            merged_results={},
            reformat_job_name="job",
            branch_name="main",
            template_ds=_template(),
            tmp_store=tmp_path,
            setup_info={},
            workers_total=1,
            update_template_with_results=False,
            consolidated=True,
        )

        # No metadata copies, no commits, and no replicas/primary stores queried.
        stub_io["copy_zarr_metadata"].assert_not_called()
        # One primary + one replica repo configured; none should have been touched.
        for _role, repo in factory.icechunk_repos(sort="primary-first"):
            assert repo.sessions == []
            assert repo.reset_calls == []
            assert repo.delete_branch_calls == []

    def test_icechunk_commits_metadata_resets_main_and_cleans_up(
        self, tmp_path: Path, stub_io: dict[str, MagicMock]
    ) -> None:
        factory = FakeStoreFactory()
        primary_repo = FakeRepo(initial_main="snap-primary-init")
        replica_repo = FakeRepo(initial_main="snap-replica-init")
        # Temp branches already exist (set up by parallel_setup in production).
        primary_repo._branches["temp-branch"] = "snap-primary-init"
        replica_repo._branches["temp-branch"] = "snap-replica-init"
        factory.set_icechunk_repos(
            [("primary", primary_repo), ("replica-0", replica_repo)]
        )
        setup_info: pc.SetupInfo = {
            "repo_snapshots": {
                "primary": "snap-primary-init",
                "replica-0": "snap-replica-init",
            }
        }

        pc.finalize(
            factory,  # ty: ignore[invalid-argument-type]
            all_jobs=[],
            merged_results={},
            reformat_job_name="job",
            branch_name="temp-branch",
            template_ds=_template(),
            tmp_store=tmp_path,
            setup_info=setup_info,
            workers_total=1,
            update_template_with_results=False,
            consolidated=True,
        )

        # Replica processed before primary per primary-last order.
        roles_reset = [call[0] for call in primary_repo.reset_calls]
        assert roles_reset == ["main"]
        assert primary_repo.reset_calls[0][2] == "snap-primary-init"
        assert replica_repo.reset_calls[0][2] == "snap-replica-init"

        # Commit happened per repo before reset_branch.
        assert len(primary_repo.sessions[0].commit_calls) == 1
        commit_msg = primary_repo.sessions[0].commit_calls[0][0]
        assert commit_msg.startswith("Update at ")
        assert commit_msg.endswith("Z")

        # Second pass deletes each repo's temp branch.
        assert primary_repo.delete_branch_calls == ["temp-branch"]
        assert replica_repo.delete_branch_calls == ["temp-branch"]

        # copy_zarr_metadata called once per repo with icechunk_only=True.
        assert stub_io["copy_zarr_metadata"].call_count == 2
        for call in stub_io["copy_zarr_metadata"].call_args_list:
            assert call.kwargs == {
                "icechunk_only": True,
                "exclude_coord_value_chunks": (),
            }

    def test_raises_when_main_diverged(
        self, tmp_path: Path, stub_io: dict[str, MagicMock]
    ) -> None:
        factory = FakeStoreFactory()
        repo = FakeRepo(initial_main="snap-moved-externally")
        repo._branches["temp-branch"] = "snap-temp"
        repo.ancestry_by_branch["temp-branch"] = ["snap-temp", "snap-original"]
        factory.set_icechunk_repos([("primary", repo)])
        setup_info: pc.SetupInfo = {"repo_snapshots": {"primary": "snap-original"}}

        with pytest.raises(RuntimeError, match="main moved during this job"):
            pc.finalize(
                factory,  # ty: ignore[invalid-argument-type]
                all_jobs=[],
                merged_results={},
                reformat_job_name="job",
                branch_name="temp-branch",
                template_ds=_template(),
                tmp_store=tmp_path,
                setup_info=setup_info,
                workers_total=1,
                update_template_with_results=False,
                consolidated=True,
            )

        # No session, no reset, no icechunk_only copy, and the temp branch is left
        # in place for inspection.
        assert repo.sessions == []
        assert repo.reset_calls == []
        stub_io["copy_zarr_metadata"].assert_not_called()
        assert repo.delete_branch_calls == []

    def test_skips_repo_already_reset_by_previous_attempt(self, tmp_path: Path) -> None:
        factory = FakeStoreFactory()
        # A previous finalize attempt committed on the branch and reset main to it,
        # then died before branch cleanup.
        repo = FakeRepo(initial_main="snap-finalized")
        repo._branches["temp-branch"] = "snap-finalized"
        repo.ancestry_by_branch["temp-branch"] = [
            "snap-finalized",
            "snap-worker-commit",
            "snap-original",
        ]
        factory.set_icechunk_repos([("primary", repo)])
        setup_info: pc.SetupInfo = {"repo_snapshots": {"primary": "snap-original"}}

        pc.finalize(
            factory,  # ty: ignore[invalid-argument-type]
            all_jobs=[],
            merged_results={},
            reformat_job_name="job",
            branch_name="temp-branch",
            template_ds=_template(),
            tmp_store=tmp_path,
            setup_info=setup_info,
            workers_total=1,
            update_template_with_results=False,
            consolidated=True,
        )

        # Publication already happened; this attempt only cleans up.
        assert repo.sessions == []
        assert repo.reset_calls == []
        assert repo.delete_branch_calls == ["temp-branch"]

    def test_concurrent_movement_on_replica_still_publishes_primary(
        self,
        tmp_path: Path,
        stub_io: dict[str, MagicMock],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Only the replica's main moved to a foreign snapshot (a concurrent,
        # uncoordinated write); the primary must still be published before
        # finalize raises for the diverged replica.
        factory = FakeStoreFactory()
        primary_repo = FakeRepo(initial_main="snap-primary-init")
        primary_repo._branches["temp-branch"] = "snap-primary-init"
        replica_repo = FakeRepo(initial_main="snap-replica-foreign")
        replica_repo._branches["temp-branch"] = "snap-replica-branch-tip"
        factory.set_icechunk_repos(
            [("primary", primary_repo), ("replica-0", replica_repo)]
        )
        setup_info: pc.SetupInfo = {
            "repo_snapshots": {
                "primary": "snap-primary-init",
                "replica-0": "snap-replica-init",
            }
        }

        with (
            caplog.at_level(logging.INFO),
            pytest.raises(
                RuntimeError, match=r"main moved during this job on \['replica-0'\]"
            ),
        ):
            pc.finalize(
                factory,  # ty: ignore[invalid-argument-type]
                all_jobs=[],
                merged_results={},
                reformat_job_name="job",
                branch_name="temp-branch",
                template_ds=_template(),
                tmp_store=tmp_path,
                setup_info=setup_info,
                workers_total=1,
                update_template_with_results=False,
                consolidated=True,
                job_uid="uid-job",
            )

        assert "state.json" not in factory.files.get("job", {})

        # Replica took the error path: no commit, no reset.
        assert replica_repo.sessions == []
        assert replica_repo.reset_calls == []
        error_messages = [
            r.getMessage() for r in caplog.records if r.levelno == logging.ERROR
        ]
        assert error_messages == [
            (
                "replica-0: main moved past this job's starting snapshot; "
                "branch temp-branch will not be published"
            )
        ]

        # The continue did not abort the loop: primary still committed and reset main.
        assert len(primary_repo.sessions) == 1
        assert len(primary_repo.sessions[0].commit_calls) == 1
        assert [(c[0], c[2]) for c in primary_repo.reset_calls] == [
            ("main", "snap-primary-init")
        ]
        assert stub_io["copy_zarr_metadata"].call_count == 1

        # The raise leaves both temp branches in place for inspection.
        assert primary_repo.delete_branch_calls == []
        assert replica_repo.delete_branch_calls == []

    def test_retry_after_partial_publish_is_benign_and_finishes_remaining(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # A previous attempt reset the replica's main to this job's branch tip,
        # then died before the primary. The retry must treat the replica as
        # already published (info, not error) and still publish the primary.
        factory = FakeStoreFactory()
        primary_repo = FakeRepo(initial_main="snap-primary-init")
        primary_repo._branches["temp-branch"] = "snap-primary-init"
        replica_repo = FakeRepo(initial_main="snap-replica-published")
        replica_repo._branches["temp-branch"] = "snap-replica-published"
        factory.set_icechunk_repos(
            [("primary", primary_repo), ("replica-0", replica_repo)]
        )
        setup_info: pc.SetupInfo = {
            "repo_snapshots": {
                "primary": "snap-primary-init",
                "replica-0": "snap-replica-init",
            }
        }

        with caplog.at_level(logging.INFO):
            pc.finalize(
                factory,  # ty: ignore[invalid-argument-type]
                all_jobs=[],
                merged_results={},
                reformat_job_name="job",
                branch_name="temp-branch",
                template_ds=_template(),
                tmp_store=tmp_path,
                setup_info=setup_info,
                workers_total=1,
                update_template_with_results=False,
                consolidated=True,
            )

        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
        assert any(
            "main already reset by a previous attempt" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.INFO
        )
        # Already-published replica is left as-is.
        assert replica_repo.sessions == []
        assert replica_repo.reset_calls == []
        assert replica_repo.lookup_branch("main") == "snap-replica-published"
        # The not-yet-reset primary still publishes.
        assert len(primary_repo.sessions[0].commit_calls) == 1
        assert [(c[0], c[2]) for c in primary_repo.reset_calls] == [
            ("main", "snap-primary-init")
        ]
        # Both temp branches cleaned up.
        assert primary_repo.delete_branch_calls == ["temp-branch"]
        assert replica_repo.delete_branch_calls == ["temp-branch"]

    def test_branch_already_deleted_is_benign(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # A previous attempt fully finished for this repo: main reset to the
        # branch tip and the temp branch deleted. FakeRepo.ancestry raises
        # KeyError on a missing branch, so this also pins that finalize never
        # walks the deleted branch's history.
        factory = FakeStoreFactory()
        repo = FakeRepo(initial_main="snap-published")
        factory.set_icechunk_repos([("primary", repo)])
        setup_info: pc.SetupInfo = {"repo_snapshots": {"primary": "snap-original"}}

        with caplog.at_level(logging.INFO):
            pc.finalize(
                factory,  # ty: ignore[invalid-argument-type]
                all_jobs=[],
                merged_results={},
                reformat_job_name="job",
                branch_name="temp-branch",
                template_ds=_template(),
                tmp_store=tmp_path,
                setup_info=setup_info,
                workers_total=1,
                update_template_with_results=False,
                consolidated=True,
            )

        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
        assert any(
            "main already reset by a previous attempt" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.INFO
        )
        assert repo.sessions == []
        assert repo.reset_calls == []
        assert repo.delete_branch_calls == []

    def test_cleanup_skips_repo_whose_branch_is_already_deleted(
        self, tmp_path: Path
    ) -> None:
        # Replica finished fully on a previous attempt (branch already deleted);
        # FakeRepo.delete_branch raises KeyError on a missing branch, so pass 2
        # must guard on branch existence per repo.
        factory = FakeStoreFactory()
        primary_repo = FakeRepo(initial_main="snap-primary-init")
        primary_repo._branches["temp-branch"] = "snap-primary-init"
        replica_repo = FakeRepo(initial_main="snap-replica-published")
        factory.set_icechunk_repos(
            [("primary", primary_repo), ("replica-0", replica_repo)]
        )
        setup_info: pc.SetupInfo = {
            "repo_snapshots": {
                "primary": "snap-primary-init",
                "replica-0": "snap-replica-init",
            }
        }

        pc.finalize(
            factory,  # ty: ignore[invalid-argument-type]
            all_jobs=[],
            merged_results={},
            reformat_job_name="job",
            branch_name="temp-branch",
            template_ds=_template(),
            tmp_store=tmp_path,
            setup_info=setup_info,
            workers_total=1,
            update_template_with_results=False,
            consolidated=True,
        )

        assert replica_repo.delete_branch_calls == []
        assert primary_repo.delete_branch_calls == ["temp-branch"]
        assert [(c[0], c[2]) for c in primary_repo.reset_calls] == [
            ("main", "snap-primary-init")
        ]

    def test_overwrite_backfill_publishes_zarr3_metadata(
        self, tmp_path: Path, stub_io: dict[str, MagicMock]
    ) -> None:
        factory = FakeStoreFactory()

        pc.finalize(
            factory,  # ty: ignore[invalid-argument-type]
            all_jobs=[],
            merged_results={},
            reformat_job_name="job",
            branch_name="main",
            template_ds=_template(),
            tmp_store=tmp_path,
            setup_info={},
            workers_total=1,
            update_template_with_results=False,
            consolidated=True,
            publish_zarr3_metadata=True,
            exclude_coord_value_chunks={"ingested_forecast_length"},
        )

        assert stub_io["copy_zarr_metadata"].call_count == 1
        copy_kwargs = stub_io["copy_zarr_metadata"].call_args.kwargs
        assert copy_kwargs["zarr3_only"] is True
        assert copy_kwargs["skip_unchanged"] is True
        assert copy_kwargs["exclude_coord_value_chunks"] == {"ingested_forecast_length"}

    def test_zarr3_metadata_copy_only_when_update_template_with_results(
        self, tmp_path: Path, stub_io: dict[str, MagicMock]
    ) -> None:
        factory = FakeStoreFactory()

        # Backfill path: update_template_with_results=False → no zarr3 copy.
        pc.finalize(
            factory,  # ty: ignore[invalid-argument-type]
            all_jobs=[],
            merged_results={},
            reformat_job_name="job",
            branch_name="main",
            template_ds=_template(),
            tmp_store=tmp_path,
            setup_info={},
            workers_total=1,
            update_template_with_results=False,
            consolidated=True,
        )
        stub_io["copy_zarr_metadata"].assert_not_called()
        # Even without update_template_with_results, finalize writes template
        # metadata to tmp_store so the copy below it never reads an empty dir.
        stub_io["write_metadata"].assert_called_once()
        stub_io["write_metadata"].reset_mock()

        # Update path: flips to True → zarr3 copy called once with zarr3_only=True.
        job = MagicMock()
        job.update_template_with_results.return_value = _template()
        merged = {
            "v": [
                SourceFileResult(
                    status=SourceFileStatus.Succeeded,
                    out_loc={"time": pd.Timestamp("2026-04-15")},
                    url="u",
                )
            ]
        }
        pc.finalize(
            factory,  # ty: ignore[invalid-argument-type]
            all_jobs=[job],
            merged_results=merged,
            reformat_job_name="job",
            branch_name="main",
            template_ds=_template(),
            tmp_store=tmp_path,
            setup_info={},
            workers_total=1,
            update_template_with_results=True,
            consolidated=True,
        )
        job.update_template_with_results.assert_called_once_with(merged)
        assert stub_io["copy_zarr_metadata"].call_count == 1
        copy_kwargs = stub_io["copy_zarr_metadata"].call_args.kwargs
        assert copy_kwargs["zarr3_only"] is True
        assert copy_kwargs["replica_stores"] == []
        stub_io["write_metadata"].assert_called_once()

    def test_update_refuses_to_publish_template_shorter_than_store(
        self, tmp_path: Path, stub_io: dict[str, MagicMock]
    ) -> None:
        """A source outage trims the update template back into published data; finalize
        must raise before any metadata write rather than un-publish those steps."""
        factory = FakeStoreFactory()
        factory.published_tree = _time_tree(5)
        job = MagicMock()
        job.append_dim = "time"
        job.update_template_with_results.return_value = _time_tree(3)

        with pytest.raises(ValueError, match="would retract already-published data"):
            pc.finalize(
                factory,  # ty: ignore[invalid-argument-type]
                all_jobs=[job],
                merged_results={},
                reformat_job_name="job",
                branch_name="main",
                template_ds=_time_tree(5),
                tmp_store=tmp_path,
                setup_info={},
                workers_total=1,
                update_template_with_results=True,
                consolidated=True,
            )

        stub_io["write_metadata"].assert_not_called()
        stub_io["copy_zarr_metadata"].assert_not_called()

    def test_clear_coordination_files_only_when_multi_worker(
        self, tmp_path: Path
    ) -> None:
        factory = FakeStoreFactory()
        factory.write_coordination_file("job", "results/worker-0.json", b"x")

        pc.finalize(
            factory,  # ty: ignore[invalid-argument-type]
            all_jobs=[],
            merged_results={},
            reformat_job_name="job",
            branch_name="main",
            template_ds=_template(),
            tmp_store=tmp_path,
            setup_info={},
            workers_total=1,
            update_template_with_results=False,
            consolidated=True,
        )
        assert "job" in factory.files  # not cleared

        pc.finalize(
            factory,  # ty: ignore[invalid-argument-type]
            all_jobs=[],
            merged_results={},
            reformat_job_name="job",
            branch_name="main",
            template_ds=_template(),
            tmp_store=tmp_path,
            setup_info={},
            workers_total=2,
            update_template_with_results=False,
            consolidated=True,
        )
        assert "job" not in factory.files  # cleared


class _ReadonlySession(SimpleNamespace):
    pass


class BranchFakeRepo(FakeRepo):
    """FakeRepo with readonly sessions, recorded by the branch or snapshot they pin."""

    def __init__(self, initial_main: str = "snap-main") -> None:
        super().__init__(initial_main)
        self.readonly_calls: list[dict[str, str | None]] = []

    def readonly_session(
        self, branch: str | None = None, *, snapshot_id: str | None = None
    ) -> _ReadonlySession:
        self.readonly_calls.append({"branch": branch, "snapshot_id": snapshot_id})
        return _ReadonlySession(branch=branch, snapshot_id=snapshot_id)


@pytest.fixture
def guarded_reads(monkeypatch: pytest.MonkeyPatch) -> list[_ReadonlySession]:
    """Make _open_session return the session itself so guards see what was pinned."""
    opened: list[_ReadonlySession] = []

    def open_session(session: _ReadonlySession) -> _ReadonlySession:
        opened.append(session)
        return session

    monkeypatch.setattr(pc, "_open_session", open_session)
    return opened


def _jobs(*regions: tuple[int, int], paths: tuple[str, ...] = ("v",)) -> list[object]:
    return [
        SimpleNamespace(
            append_dim="time",
            region=slice(start, stop),
            data_vars=[SimpleNamespace(path=path) for path in paths],
        )
        for start, stop in regions
    ]


def _fingerprint(jobs: list[object], template: xr.DataTree) -> str:
    return pc.work_fingerprint(
        jobs,  # ty: ignore[invalid-argument-type]
        template,
        append_dim="time",
        workers_total=2,
        worker_assignment="spread",
    )


class TestWorkFingerprint:
    def test_equivalent_independently_built_work_matches(self) -> None:
        assert _fingerprint(_jobs((0, 2), (2, 4)), _time_tree(4)) == _fingerprint(
            _jobs((0, 2), (2, 4)), _time_tree(4)
        )
        assert len(_fingerprint(_jobs((0, 2)), _time_tree(4))) == 64

    def test_equivalent_region_representations_match(self) -> None:
        explicit = _jobs((0, 2))
        explicit[0].region = slice(0, 2, 1)  # ty: ignore[unresolved-attribute]
        implicit = _jobs((0, 2))
        implicit[0].region = slice(None, 2)  # ty: ignore[unresolved-attribute]
        assert _fingerprint(explicit, _time_tree(4)) == _fingerprint(
            implicit, _time_tree(4)
        )

    def test_rejects_jobs_for_another_append_dim(self) -> None:
        jobs = _jobs((0, 2))
        jobs[0].append_dim = "init_time"  # ty: ignore[unresolved-attribute]
        with pytest.raises(AssertionError):
            _fingerprint(jobs, _time_tree(4))

    def test_same_regions_with_a_different_append_extent_differ(self) -> None:
        jobs = _jobs((0, 2), (2, 4))
        assert _fingerprint(jobs, _time_tree(3)) != _fingerprint(jobs, _time_tree(4))

    def test_different_append_labels_differ(self) -> None:
        shifted = xr.DataTree(
            _time_tree(4)
            .to_dataset()
            .assign_coords(time=pd.date_range("2026-03-31", periods=4, freq="D"))
        )
        jobs = _jobs((0, 2), (2, 4))
        assert _fingerprint(jobs, _time_tree(4)) != _fingerprint(jobs, shifted)

    @pytest.mark.parametrize(
        "other",
        [
            _jobs((2, 4), (0, 2)),
            _jobs((0, 2), (2, 3)),
            _jobs((0, 2), (2, 4), paths=("w",)),
            _jobs((0, 2), (2, 4), paths=("v", "w")),
        ],
    )
    def test_different_work_lists_differ(self, other: list[object]) -> None:
        assert _fingerprint(_jobs((0, 2), (2, 4)), _time_tree(4)) != _fingerprint(
            other, _time_tree(4)
        )

    @pytest.mark.parametrize(
        ("workers_total", "worker_assignment"), [(3, "spread"), (2, "contiguous")]
    )
    def test_worker_partitioning_is_covered(
        self,
        workers_total: int,
        worker_assignment: Literal["spread", "contiguous"],
    ) -> None:
        jobs = _jobs((0, 2))
        assert _fingerprint(jobs, _time_tree(4)) != pc.work_fingerprint(
            jobs,  # ty: ignore[invalid-argument-type]
            _time_tree(4),
            append_dim="time",
            workers_total=workers_total,
            worker_assignment=worker_assignment,
        )


def _main_repo() -> BranchFakeRepo:
    return BranchFakeRepo(initial_main="snap-main")


def _guarded_main_setup(
    factory: FakeStoreFactory,
    tmp_path: Path,
    repos: list[tuple[str, BranchFakeRepo]],
    *,
    is_first: bool = True,
    workers_total: int = 2,
    guard: object = None,
    fingerprint: str = "fp-a",
) -> pc.SetupInfo:
    return pc.parallel_setup(
        factory,  # ty: ignore[invalid-argument-type]
        is_first=is_first,
        workers_total=workers_total,
        reformat_job_name="job",
        branch_name="_job_job",
        template_ds=_template(),
        tmp_store=tmp_path,
        icechunk_repos=repos,  # ty: ignore[invalid-argument-type]
        consolidated=True,
        existing_data_guard=guard if guard is not None else MagicMock(),  # ty: ignore[invalid-argument-type]
        work_fingerprint=fingerprint,
    )


class TestGuardedMainSetup:
    @pytest.mark.parametrize("workers_total", [1, 2])
    def test_guards_every_pinned_main_snapshot_before_branching(
        self,
        tmp_path: Path,
        guarded_reads: list[_ReadonlySession],
        workers_total: int,
    ) -> None:
        factory = FakeStoreFactory()
        primary, replica = _main_repo(), BranchFakeRepo(initial_main="snap-replica")
        events: list[str] = []
        primary.create_branch = lambda name, snap: events.append(f"branch {snap}")  # ty: ignore[invalid-assignment]
        replica.create_branch = lambda name, snap: events.append(f"branch {snap}")  # ty: ignore[invalid-assignment]

        def guard(existing: _ReadonlySession) -> None:
            events.append(f"guard {existing.snapshot_id}")

        result = _guarded_main_setup(
            factory,
            tmp_path,
            [("primary", primary), ("replica-0", replica)],
            workers_total=workers_total,
            guard=guard,
        )

        assert events == [
            "guard snap-main",
            "guard snap-replica",
            "branch snap-main",
            "branch snap-replica",
        ]
        assert [s.branch for s in guarded_reads] == [None, None]
        assert result == {
            "repo_snapshots": {"primary": "snap-main", "replica-0": "snap-replica"},
            "work_fingerprint": "fp-a",
        }
        assert json.loads(factory.files["job"]["setup/ready.json"]) == result
        assert factory.persist_virtual_config_calls == 1

    def test_guard_failure_writes_nothing(
        self,
        tmp_path: Path,
        stub_io: dict[str, MagicMock],
        guarded_reads: list[_ReadonlySession],  # noqa: ARG002
    ) -> None:
        factory = FakeStoreFactory()
        repo = _main_repo()

        def guard(existing: object) -> None:
            raise ValueError("labels do not match")

        with pytest.raises(ValueError, match="labels do not match"):
            _guarded_main_setup(factory, tmp_path, [("primary", repo)], guard=guard)

        assert repo.create_branch_calls == []
        assert repo.sessions == []
        stub_io["copy_zarr_metadata"].assert_not_called()
        assert factory.files == {}

    def test_retry_guards_recorded_snapshot_after_main_moves(
        self,
        tmp_path: Path,
        guarded_reads: list[_ReadonlySession],  # noqa: ARG002
    ) -> None:
        factory = FakeStoreFactory()
        repo = _main_repo()
        repo._branches["main"] = "snap-main-reset"
        ready = {"repo_snapshots": {"primary": "snap-main"}, "work_fingerprint": "fp-a"}
        factory.write_coordination_file(
            "job", "setup/ready.json", json.dumps(ready).encode()
        )

        result = _guarded_main_setup(factory, tmp_path, [("primary", repo)])

        assert result == ready
        assert repo.readonly_calls == [{"branch": None, "snapshot_id": "snap-main"}]
        assert repo.create_branch_calls == [("_job_job", "snap-main")]

    @pytest.mark.parametrize("is_first", [True, False])
    @pytest.mark.parametrize("stored", [None, "fp-b"])
    def test_rejects_recorded_setup_for_different_work(
        self,
        tmp_path: Path,
        guarded_reads: list[_ReadonlySession],  # noqa: ARG002
        is_first: bool,
        stored: str | None,
    ) -> None:
        factory = FakeStoreFactory()
        repo = _main_repo()
        ready: dict[str, object] = {"repo_snapshots": {"primary": "snap-main"}}
        if stored is not None:
            ready["work_fingerprint"] = stored
        factory.write_coordination_file(
            "job", "setup/ready.json", json.dumps(ready).encode()
        )
        guard = MagicMock()

        with pytest.raises(AssertionError, match="work"):
            _guarded_main_setup(
                factory, tmp_path, [("primary", repo)], is_first=is_first, guard=guard
            )

        guard.assert_not_called()
        assert repo.create_branch_calls == []
        assert repo.sessions == []
        assert json.loads(factory.files["job"]["setup/ready.json"]) == ready

    @pytest.mark.parametrize("workers_total", [1, 2])
    def test_rejects_temp_branch_without_recorded_setup(
        self, tmp_path: Path, workers_total: int
    ) -> None:
        factory = FakeStoreFactory()
        repo = _main_repo()
        repo._branches["_job_job"] = "snap-main"
        guard = MagicMock()

        with pytest.raises(AssertionError, match="without recorded setup"):
            _guarded_main_setup(
                factory,
                tmp_path,
                [("primary", repo)],
                workers_total=workers_total,
                guard=guard,
            )

        guard.assert_not_called()
        assert factory.files == {}

    def test_later_worker_checks_its_fingerprint_and_guards_pinned_snapshots(
        self, tmp_path: Path, guarded_reads: list[_ReadonlySession]
    ) -> None:
        factory = FakeStoreFactory()
        primary, replica = _main_repo(), BranchFakeRepo(initial_main="snap-r")
        primary._branches["main"] = "snap-main-moved"
        ready = {
            "repo_snapshots": {"primary": "snap-main", "replica-0": "snap-r"},
            "work_fingerprint": "fp-a",
        }
        factory.write_coordination_file(
            "job", "setup/ready.json", json.dumps(ready).encode()
        )
        guard = MagicMock()

        result = _guarded_main_setup(
            factory,
            tmp_path,
            [("primary", primary), ("replica-0", replica)],
            is_first=False,
            guard=guard,
        )

        assert result == ready
        assert primary.readonly_calls == [{"branch": None, "snapshot_id": "snap-main"}]
        assert replica.readonly_calls == [{"branch": None, "snapshot_id": "snap-r"}]
        assert [call.args[0] for call in guard.call_args_list] == guarded_reads

    def test_later_worker_guard_failure_raises(
        self,
        tmp_path: Path,
        guarded_reads: list[_ReadonlySession],  # noqa: ARG002
    ) -> None:
        factory = FakeStoreFactory()
        ready = {"repo_snapshots": {"primary": "snap-main"}, "work_fingerprint": "fp-a"}
        factory.write_coordination_file(
            "job", "setup/ready.json", json.dumps(ready).encode()
        )

        def guard(existing: object) -> None:
            raise ValueError("labels do not match")

        with pytest.raises(ValueError, match="labels do not match"):
            _guarded_main_setup(
                factory,
                tmp_path,
                [("primary", _main_repo())],
                is_first=False,
                guard=guard,
            )

    def test_guarded_main_requires_a_fingerprint(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError):
            pc.parallel_setup(
                FakeStoreFactory(),  # ty: ignore[invalid-argument-type]
                is_first=True,
                workers_total=1,
                reformat_job_name="job",
                branch_name="_job_job",
                template_ds=_template(),
                tmp_store=tmp_path,
                icechunk_repos=[("primary", _main_repo())],  # ty: ignore[invalid-argument-type]
                consolidated=True,
                existing_data_guard=MagicMock(),
            )


class TestGuardedMainFinalize:
    def _finalize(
        self,
        factory: FakeStoreFactory,
        tmp_path: Path,
        repo: BranchFakeRepo,
        updated: xr.DataTree,
        *,
        workers_total: int = 1,
    ) -> None:
        job = MagicMock()
        job.append_dim = "time"
        job.update_template_with_results.return_value = updated
        factory.set_icechunk_repos([("primary", repo)])
        repo._branches["_job_job"] = "snap-temp"
        pc.finalize(
            factory,  # ty: ignore[invalid-argument-type]
            all_jobs=[job],
            merged_results={},
            reformat_job_name="job",
            branch_name="_job_job",
            template_ds=updated,
            tmp_store=tmp_path,
            setup_info={
                "repo_snapshots": {"primary": "snap-main"},
                "work_fingerprint": "fp-a",
            },
            workers_total=workers_total,
            update_template_with_results=True,
            consolidated=True,
        )

    @pytest.fixture
    def pinned_trees(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, xr.DataTree]:
        trees: dict[str, xr.DataTree] = {}
        monkeypatch.setattr(
            pc, "_open_session", lambda session: trees[session.snapshot_id]
        )
        return trees

    def test_retraction_is_checked_against_pinned_snapshot_not_fresh_main(
        self,
        tmp_path: Path,
        stub_io: dict[str, MagicMock],
        pinned_trees: dict[str, xr.DataTree],
    ) -> None:
        factory = FakeStoreFactory()
        factory.published_tree = _time_tree(2)
        pinned_trees["snap-main"] = _time_tree(5)
        repo = _main_repo()

        with pytest.raises(ValueError, match="would retract already-published data"):
            self._finalize(factory, tmp_path, repo, _time_tree(3))

        assert repo.reset_calls == []
        stub_io["copy_zarr_metadata"].assert_not_called()

    def test_trimmed_template_labels_must_match_pinned_snapshot(
        self,
        tmp_path: Path,
        stub_io: dict[str, MagicMock],
        pinned_trees: dict[str, xr.DataTree],
    ) -> None:
        factory = FakeStoreFactory()
        pinned_trees["snap-main"] = _time_tree(3)
        shifted = xr.DataTree(
            _time_tree(4)
            .to_dataset()
            .assign_coords(time=pd.date_range("2026-03-31", periods=4, freq="D"))
        )
        repo = _main_repo()

        with pytest.raises(ValueError, match="labels do not match"):
            self._finalize(factory, tmp_path, repo, shifted)

        assert repo.reset_calls == []
        stub_io["copy_zarr_metadata"].assert_not_called()

    def test_single_worker_publishes_from_pinned_snapshot_and_clears_setup(
        self,
        tmp_path: Path,
        pinned_trees: dict[str, xr.DataTree],
    ) -> None:
        factory = FakeStoreFactory()
        factory.write_coordination_file("job", "setup/ready.json", b"{}")
        pinned_trees["snap-main"] = _time_tree(3)
        repo = _main_repo()

        self._finalize(factory, tmp_path, repo, _time_tree(4))

        assert [(name, src) for name, _new, src in repo.reset_calls] == [
            ("main", "snap-main")
        ]
        assert factory.files == {}


class TestMissingTempBranchAfterMainMoved:
    def _finalize(
        self,
        tmp_path: Path,
        setup_info: pc.SetupInfo,
        *,
        update_template_with_results: bool,
    ) -> tuple[FakeStoreFactory, BranchFakeRepo]:
        factory = FakeStoreFactory()
        repo = _main_repo()
        repo._branches["main"] = "snap-main-moved"
        factory.set_icechunk_repos([("primary", repo)])
        job = MagicMock()
        job.append_dim = "time"
        job.update_template_with_results.return_value = _time_tree(3)
        pc.finalize(
            factory,  # ty: ignore[invalid-argument-type]
            all_jobs=[job],
            merged_results={},
            reformat_job_name="job",
            branch_name="_job_job",
            template_ds=_time_tree(3),
            tmp_store=tmp_path,
            setup_info=setup_info,
            workers_total=2,
            update_template_with_results=update_template_with_results,
            consolidated=True,
        )
        return factory, repo

    def test_guarded_update_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pc, "_open_session", lambda session: _time_tree(3))

        with pytest.raises(RuntimeError, match="moved during this job"):
            self._finalize(
                tmp_path,
                {"repo_snapshots": {"primary": "snap-main"}, "work_fingerprint": "fp"},
                update_template_with_results=True,
            )

    def test_unguarded_main_backfill_treats_it_as_already_reset(
        self, tmp_path: Path
    ) -> None:
        factory, repo = self._finalize(
            tmp_path,
            {"repo_snapshots": {"primary": "snap-main"}},
            update_template_with_results=False,
        )

        assert repo.reset_calls == []
        assert repo.lookup_branch("main") == "snap-main-moved"
        assert factory.files == {}
