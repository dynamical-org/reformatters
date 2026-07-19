"""Fast unit tests for parallel_coordination.

Exercises the coordination logic in isolation by stubbing out zarr/icechunk
I/O with fakes and monkeypatching. End-to-end coverage — including real
zarr writes, real icechunk sessions, and worker restart semantics —
lives in tests/common/test_parallel_writes.py.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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

    def count_coordination_files(self, job_name: str, prefix: str) -> int:
        return len(self.read_all_coordination_files(job_name, prefix))

    def clear_coordination_files(self, job_name: str) -> None:
        self.files.pop(job_name, None)

    def icechunk_repos(self, *, sort: str) -> list[tuple[str, FakeRepo]]:
        return list(self._icechunk_repos_by_sort[sort])

    def primary_store(self, writable: bool = False) -> object:  # noqa: ARG002
        return self._primary_store

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
        self, monkeypatch: pytest.MonkeyPatch
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

        pc.wait_for_workers(factory, "job", workers_total=3)  # ty: ignore[invalid-argument-type]

        # Started with 1 file, needs 3 → 2 polls.
        assert sleep_calls == [10, 10]


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
