import inspect
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fsspec
import pytest
import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from zarr.storage import LocalStore

from reformatters.common import materialized_region_job, parallel_coordination, storage
from reformatters.common.materialized_region_job import MaterializedRegionJob
from scripts.ifsens_profiling import profiling
from tests.common.failure_injection_test import (
    FailRegionJob,
    _backfill_through,
    _make_dataset,
    _time,
)

SECRET = "CANARY-7f3a9c-SECRET"  # noqa: S105

FAKE_PY_SPY = f"""#!{sys.executable}
import signal, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
if args == ["--version"]:
    print("py-spy 0.4.2")
    sys.exit(0)
out = Path(args[args.index("--output") + 1])
child = subprocess.Popen(args[args.index("--") + 1 :])
signal.signal(signal.SIGTERM, lambda *_: child.terminate())
rc = child.wait()
if args[args.index("--format") + 1] == "raw":
    out.write_text("<module> (x.py:1);busy (x.py:2) 7\\n")
else:
    out.write_text('{{"profiles": [{{"samples": [[0], [0]]}}]}}')
sys.exit(rc)
"""


REPO = Path(__file__).resolve().parents[3]
REAL_PY_SPY = Path(os.environ.get("PY_SPY_TEST_BINARY", "/opt/profiler/bin/py-spy"))


@pytest.fixture(autouse=True)
def importable_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", f"{REPO / 'src'}:{REPO / 'src' / 'reformatters'}")


@pytest.fixture
def fake_py_spy(tmp_path: Path) -> Path:
    path = tmp_path / "py-spy"
    path.write_text(FAKE_PY_SPY)
    path.chmod(0o755)
    return path


def _segments(directory: Path, prefix: str) -> list[dict[str, Any]]:
    return [
        event
        for path in sorted(directory.glob(f"{prefix}-*.json"))
        for event in json.loads(path.read_text())
    ]


def test_hooks_time_phases_without_changing_output(tmp_path: Path) -> None:
    plain = _make_dataset(tmp_path / "plain")
    _backfill_through(plain, 4)
    expected = xr.open_zarr(plain.store_factory.primary_store()).load()
    plain_store = plain.store_factory.primary_store()
    assert isinstance(plain_store, LocalStore)
    shutil.rmtree(plain_store.root)

    hooked = _make_dataset(tmp_path / "hooked")
    original_process = MaterializedRegionJob.process
    with profiling.install_hooks(tmp_path / "evidence"):
        _backfill_through(hooked, 4)
        durable_before_exit = len(_segments(tmp_path / "evidence", "phase"))
    assert MaterializedRegionJob.process is original_process
    assert storage.commit_if_icechunk.__module__ == storage.__name__

    xr.testing.assert_identical(
        expected, xr.open_zarr(hooked.store_factory.primary_store()).load()
    )
    events = _segments(tmp_path / "evidence", "phase")
    assert durable_before_exit == len(events)
    assert {e["phase"] for e in events} == {
        "setup",
        "unit",
        "download",
        "read",
        "transform",
        "local_write",
        "upload",
        "commit",
        "wait",
        "finalize",
    }
    unit = next(e for e in events if e["phase"] == "unit")
    assert unit["data_vars"] == ["var0", "var1"]
    assert {"read_parallelism", "download_parallelism", "cpu_count"} <= unit.keys()
    assert all(e["ok"] and e["pid"] == os.getpid() for e in events)
    assert all(
        int(e["end_monotonic_ns"]) >= int(e["start_monotonic_ns"])
        and str(e["start_utc"]).endswith("+00:00")
        for e in events
    )
    assert {e["data_var"] for e in events if e["phase"] == "read"} == {"var0", "var1"}


_J = MaterializedRegionJob
_PINNED_SIGNATURES = [
    (_J.process, ["self", "primary_store", "replica_stores"]),
    (
        _J._download_processing_group,
        ["self", "source_file_coords", "data_var_names"],
    ),
    (_J._read_into_data_array, ["self", "out", "data_var", "source_file_coords"]),
    (_J.apply_data_transformations, ["self", "data_array", "data_var"]),
    (
        _J._write_shards,
        [
            "self",
            "processing_region_da_template",
            "shared_buffer",
            "output_region_ds",
            "tmp_store",
            "write_executor",
        ],
    ),
    (
        materialized_region_job.copy_data_var,
        [
            "data_var_path",
            "i_slice",
            "template_ds",
            "append_dim",
            "tmp_store",
            "primary_store",
            "replica_stores",
        ],
    ),
    (
        storage.commit_if_icechunk,
        ["message", "primary_store", "replica_stores", "allow_rebase"],
    ),
    (
        parallel_coordination.parallel_setup,
        [
            "store_factory",
            "is_first",
            "workers_total",
            "reformat_job_name",
            "branch_name",
            "template_ds",
            "tmp_store",
            "icechunk_repos",
            "consolidated",
            "exclude_coord_value_chunks",
            "keep_branch",
        ],
    ),
    (
        parallel_coordination.wait_for_workers,
        ["store_factory", "reformat_job_name", "workers_total"],
    ),
    (
        parallel_coordination.collect_results,
        ["store_factory", "reformat_job_name", "workers_total"],
    ),
    (
        parallel_coordination.finalize,
        [
            "store_factory",
            "all_jobs",
            "merged_results",
            "reformat_job_name",
            "branch_name",
            "template_ds",
            "tmp_store",
            "setup_info",
            "workers_total",
            "update_template_with_results",
            "consolidated",
            "publish_zarr3_metadata",
            "exclude_coord_value_chunks",
            "job_uid",
            "keep_branch",
        ],
    ),
]


@pytest.mark.parametrize(("fn", "parameters"), _PINNED_SIGNATURES)
def test_wrapped_boundaries_keep_their_signatures(
    fn: Callable[..., object], parameters: list[str]
) -> None:
    assert list(inspect.signature(fn).parameters) == parameters


def test_every_wrapped_symbol_is_pinned() -> None:
    pinned = {fn.__name__ for fn, _ in _PINNED_SIGNATURES}
    wrapped = set(profiling._JOB_METHOD_PHASES) | {
        name for _, name, _ in profiling._MODULE_FUNCTION_PHASES
    }
    assert pinned == wrapped


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        ({"fail_download_times": frozenset({_time(1)})}, "DownloadFailed"),
        ({"fail_read_var_times": frozenset({("var1", _time(1))})}, "ReadFailed"),
    ],
)
def test_failed_source_file_stops_before_upload_and_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: dict[str, Any],
    status: str,
) -> None:
    for attr, value in failure.items():
        monkeypatch.setattr(FailRegionJob, attr, value)
    dataset = _make_dataset(tmp_path / "data")
    with (
        profiling.install_hooks(tmp_path / "evidence"),
        pytest.raises(profiling.SourceFileFailureError, match=status),
    ):
        _backfill_through(dataset, 4)

    events = _segments(tmp_path / "evidence", "phase")
    phases = {e["phase"] for e in events}
    assert not phases & {"commit", "wait", "finalize"}
    failed_var = "var1" if status == "ReadFailed" else None
    assert not [
        e
        for e in events
        if e["phase"] == "upload"
        and e["region_start"] == 0
        and failed_var in (None, e["data_var"])
    ]
    assert {e["error_type"] for e in events if not e["ok"]} == {
        "SourceFileFailureError"
    }
    text = "".join(p.read_text() for p in (tmp_path / "evidence").iterdir())
    assert "https://" not in text
    assert "simulated" not in text


def test_sample_metrics_reads_cgroup_and_proc(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "cpu.stat").write_text("usage_usec 100\nuser_usec 60\nsystem_usec 40\n")
    (cgroup / "memory.current").write_text("2048\n")
    (cgroup / "cpu.max").write_text("max 100000\n")
    (cgroup / "memory.peak").write_text("4096\n")
    (cgroup / "memory.stat").write_text("anon 10\nfile 20\nshmem 30\nslab 1\n")
    (cgroup / "io.stat").write_text(
        "259:0 rbytes=5 wbytes=7 rios=1 wios=2 dbytes=0 dios=0\n"
        "259:1 rbytes=5 wbytes=3 rios=1 wios=1 dbytes=0 dios=0\n"
    )
    proc = tmp_path / "proc"
    for pid, ppid in ((10, 1), (11, 10), (12, 11), (20, 1)):
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "stat").write_text(
            f"{pid} (py thon) S {ppid} 0 0 0 0 0 0 0 0 0 {pid} 1 0 0\n"
        )
        (proc / str(pid) / "statm").write_text("100 2 0 0 0 0 0\n")
    (proc / "net").mkdir()
    (proc / "net" / "dev").write_text(
        "Inter-|\n face |\n"
        "    lo: 999 0 0 0 0 0 0 0 999 0 0 0 0 0 0 0\n"
        "  eth0: 100 0 0 0 0 0 0 0 50 0 0 0 0 0 0 0\n"
    )

    metrics = profiling.sample_metrics(cgroup, proc, 10, [tmp_path])

    assert metrics["cgroup_cpu"]["usage_usec"] == 100
    assert metrics["cgroup_memory_peak_bytes"] == 4096
    assert metrics["cgroup_cpu_max"] == "max 100000"
    assert metrics["cgroup_memory"] == {"anon": 10, "file": 20, "shmem": 30}
    assert metrics["cgroup_io"]["rbytes"] == 10
    assert metrics["cgroup_io"]["wbytes"] == 10
    assert metrics["descendants"]["processes"] == 3
    assert metrics["descendants"]["cpu_ticks"] == 10 + 1 + 11 + 1 + 12 + 1
    assert metrics["descendants"]["rss_bytes"] == 3 * 2 * os.sysconf("SC_PAGE_SIZE")
    assert metrics["network"] == {"rx_bytes": 100, "tx_bytes": 50}
    assert metrics["disks"][0]["total_bytes"] > 0


def _remote(tmp_path: Path) -> tuple[fsspec.AbstractFileSystem, Path]:
    remote = tmp_path / "remote"
    remote.mkdir()
    return fsspec.filesystem("file"), remote


def test_run_profiled_persists_evidence_without_secrets(
    tmp_path: Path, fake_py_spy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET)
    fs, remote = _remote(tmp_path)
    command = [
        sys.executable,
        "-c",
        f"import time; time.sleep(1.5)  # {SECRET}",
    ]

    rc = profiling.run_profiled(
        command,
        str(remote),
        fs,
        {"run": "test"},
        tmp_path / "out",
        py_spy=fake_py_spy,
        upload_interval_s=0.5,
    )

    assert rc == 0
    names = {p.name for p in remote.iterdir()}
    assert {
        "manifest-initial.json",
        "manifest-final.json",
        profiling.PY_SPY_OUTPUT,
    } <= names
    assert any(n.startswith("resource-") for n in names)
    initial = json.loads((remote / "manifest-initial.json").read_text())
    assert initial["selftest_samples"] == 7
    assert initial["metadata"] == {"run": "test"}
    final = json.loads((remote / "manifest-final.json").read_text())
    assert final["returncode"] == 0
    assert final["evidence_error"] is None
    assert final["sha256"].keys() == names - {
        "manifest-initial.json",
        "manifest-final.json",
    }
    for path in remote.iterdir():
        assert SECRET not in path.read_text()


def test_run_profiled_fails_fast_when_initial_upload_fails(
    tmp_path: Path, fake_py_spy: Path
) -> None:
    class BrokenFs(LocalFileSystem):
        def pipe_file(self, *_: object, **__: object) -> None:
            raise OSError("denied")

    marker = tmp_path / "started"
    with pytest.raises(OSError, match="denied"):
        profiling.run_profiled(
            [sys.executable, "-c", f"open({str(marker)!r}, 'w')"],
            str(tmp_path / "remote"),
            BrokenFs(),
            {},
            tmp_path / "out",
            py_spy=fake_py_spy,
        )
    assert not marker.exists()


def test_run_profiled_stops_workload_on_upload_failure(
    tmp_path: Path, fake_py_spy: Path
) -> None:
    class FailingPutFs(LocalFileSystem):
        def put_file(self, *_: object, **__: object) -> None:
            raise OSError("upload refused")

    fs = FailingPutFs()
    remote = tmp_path / "remote"
    remote.mkdir()
    start = time.monotonic()
    rc = profiling.run_profiled(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        str(remote),
        fs,
        {},
        tmp_path / "out",
        py_spy=fake_py_spy,
        upload_interval_s=0.0,
        drain_s=5,
        flush_s=2,
    )
    assert rc == 1
    assert time.monotonic() - start < 20
    assert (remote / "manifest-initial.json").exists()


def test_sigterm_drains_and_kills_process_group(
    tmp_path: Path, fake_py_spy: Path
) -> None:
    pid_file = tmp_path / "grandchild.pid"
    stubborn = (
        "import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(600)"
    )
    supervisor = f"""
import sys
from pathlib import Path
import fsspec
from scripts.ifsens_profiling import profiling
sys.exit(profiling.run_profiled(
    [sys.executable, "-c", {stubborn!r}],
    {str(tmp_path / "remote")!r},
    fsspec.filesystem("file"),
    {{}},
    Path({str(tmp_path / "out")!r}),
    py_spy=Path({str(fake_py_spy)!r}),
    drain_s=1.0,
    flush_s=5.0,
))
"""
    (tmp_path / "remote").mkdir()
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", supervisor],
    )
    deadline = time.monotonic() + 30
    while not pid_file.exists() or not pid_file.read_text():
        assert time.monotonic() < deadline
        time.sleep(0.1)
    grandchild = int(pid_file.read_text())

    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=30)

    with pytest.raises(ProcessLookupError):
        os.kill(grandchild, 0)
    final = json.loads((tmp_path / "remote" / "manifest-final.json").read_text())
    assert final["signal"] == "SIGTERM"
    assert final["killed_after_drain"] is True
    assert any(p.name.startswith("resource-") for p in (tmp_path / "remote").iterdir())
    assert final["returncode"] is None
    assert proc.returncode == 1


def test_default_disk_paths_cover_pvc_and_shm() -> None:
    default = inspect.signature(profiling.run_profiled).parameters["disk_paths"].default
    assert default == (Path("/app/data"), Path("/dev/shm"))  # noqa: S108


def test_final_flush_failure_fails_closed(tmp_path: Path, fake_py_spy: Path) -> None:
    class FailingPutFs(LocalFileSystem):
        def put_file(self, *_: object, **__: object) -> None:
            raise OSError("upload refused")

    remote = tmp_path / "remote"
    remote.mkdir()
    rc = profiling.run_profiled(
        [sys.executable, "-c", "pass"],
        str(remote),
        FailingPutFs(),
        {},
        tmp_path / "out",
        py_spy=fake_py_spy,
        upload_interval_s=3600,
        disk_paths=[tmp_path],
    )
    assert rc == 1
    final = json.loads((remote / "manifest-final.json").read_text())
    assert final["returncode"] == 0
    assert final["evidence_error"] == "OSError"


def test_unexpected_supervisor_error_drains_group(
    tmp_path: Path, fake_py_spy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_metrics(*_: object) -> dict[str, Any]:
        raise RuntimeError("cgroup unreadable")

    monkeypatch.setattr(profiling, "sample_metrics", broken_metrics)
    pid_file = tmp_path / "child.pid"
    remote = tmp_path / "remote"
    remote.mkdir()
    rc = profiling.run_profiled(
        [
            sys.executable,
            "-c",
            (
                f"import os, time; open({str(pid_file)!r}, 'w')"
                ".write(str(os.getpid())); time.sleep(600)"
            ),
        ],
        str(remote),
        fsspec.filesystem("file"),
        {},
        tmp_path / "out",
        py_spy=fake_py_spy,
        drain_s=5,
    )
    assert rc == 1
    final = json.loads((remote / "manifest-final.json").read_text())
    assert final["evidence_error"] == "RuntimeError"
    if pid_file.exists() and pid_file.read_text():
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)


def test_missing_receipt_fails(tmp_path: Path, fake_py_spy: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    rc = profiling.run_profiled(
        [
            sys.executable,
            "-c",
            "import os, signal; os.kill(os.getppid(), signal.SIGKILL)",
        ],
        str(remote),
        fsspec.filesystem("file"),
        {},
        tmp_path / "out",
        py_spy=fake_py_spy,
        disk_paths=[tmp_path],
    )
    assert rc == 1
    assert not (remote / profiling.WORKLOAD_RECEIPT).exists()


@pytest.mark.skipif(not REAL_PY_SPY.exists(), reason="real py-spy binary not present")
@pytest.mark.parametrize("exit_code", [0, 7])
def test_real_py_spy_receipt_carries_workload_exit(
    tmp_path: Path, exit_code: int
) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    busy = (
        "import sys, time\n"
        "t = time.monotonic()\n"
        "while time.monotonic() - t < 2: pass\n"
        f"sys.exit({exit_code})"
    )
    rc = profiling.run_profiled(
        [sys.executable, "-c", busy],
        str(remote),
        fsspec.filesystem("file"),
        {},
        tmp_path / "out",
        py_spy=REAL_PY_SPY,
        disk_paths=[tmp_path],
    )
    assert rc == exit_code
    receipt = json.loads((remote / profiling.WORKLOAD_RECEIPT).read_text())
    assert receipt["returncode"] == exit_code
    final = json.loads((remote / "manifest-final.json").read_text())
    assert final["profile_samples"] > 0


def test_profiler_failure_cannot_mask_workload_success(
    tmp_path: Path, fake_py_spy: Path
) -> None:
    fake_py_spy.write_text(
        FAKE_PY_SPY.replace(
            "sys.exit(rc)",
            "sys.exit(rc if args[args.index('--format') + 1] == 'raw' else 11)",
        )
    )
    fs, remote = _remote(tmp_path)
    result = profiling.run_profiled(
        [sys.executable, "-c", "pass"],
        str(remote),
        fs,
        {},
        tmp_path / "out",
        py_spy=fake_py_spy,
    )
    assert result == 1
    final = json.loads((remote / "manifest-final.json").read_text())
    assert final["returncode"] == 0
    assert final["py_spy_returncode"] == 11
