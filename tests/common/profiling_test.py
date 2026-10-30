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
import s3fs
import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from zarr.storage import LocalStore

from reformatters.common import (
    kubernetes,
    materialized_region_job,
    parallel_coordination,
    profiling,
    storage,
)
from reformatters.common.materialized_region_job import MaterializedRegionJob
from tests.common.failure_injection_test import (
    FailRegionJob,
    _backfill_through,
    _make_dataset,
    _time,
)

SECRET = "CANARY-7f3a9c-SECRET"  # noqa: S105

FAKE_PY_SPY = f"""#!{sys.executable}
import json, signal, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
with Path(sys.argv[0]).with_suffix(".argv").open("a") as log:
    log.write(json.dumps(args) + "\\n")
if args == ["--version"]:
    print("py-spy 0.4.2")
    sys.exit(0)
assert "--nonblocking" in args, "sampling must not pause the workload"
out = Path(args[args.index("--output") + 1])
child = subprocess.Popen(args[args.index("--") + 1 :], close_fds=False)
signal.signal(signal.SIGTERM, lambda *_: child.terminate())
rc = child.wait()
if args[args.index("--format") + 1] == "raw":
    out.write_text("<module> (x.py:1);busy (x.py:2) 7\\n")
else:
    out.write_text(
        '{{"profiles": [{{"name": "Process %d Thread 1", "samples": [[0], [0]]}}]}}'
        % child.pid
    )
print(f"py-spy> Wrote file to '{{out}}'. Samples: 3 Errors: 1")
print("{SECRET}")
sys.exit(rc)
"""


REPO = Path(__file__).resolve().parents[2]
DEFAULT_REAL_PY_SPY = Path("/opt/profiler/bin/py-spy")


@pytest.fixture(autouse=True)
def importable_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", f"{REPO / 'src'}:{REPO / 'src' / 'reformatters'}")


@pytest.fixture(autouse=True)
def isolated_from_cluster_and_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Unexpected credential, cluster, or remote storage access")

    monkeypatch.setattr(kubernetes, "load_secret", forbidden)
    monkeypatch.setattr(kubernetes.config, "load_kube_config", forbidden)
    monkeypatch.setattr(kubernetes.config, "load_incluster_config", forbidden)
    monkeypatch.setattr(s3fs.S3FileSystem, "__init__", forbidden)
    monkeypatch.setattr(s3fs.S3FileSystem, "_cache", {})
    monkeypatch.setattr(storage, "_get_icechunk_storage", forbidden)
    for name in [n for n in os.environ if n.startswith("AWS_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")


REAL_TIMED_RUN = profiling._timed_run


@pytest.fixture(autouse=True)
def instant_overhead_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profiling, "_timed_run", lambda *_: 1.0)


@pytest.fixture
def fake_py_spy(tmp_path: Path) -> Path:
    path = tmp_path / "py-spy"
    path.write_text(FAKE_PY_SPY)
    path.chmod(0o755)
    return path


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _resolve_real_py_spy(configured: str | None, default: Path) -> Path | None:
    if configured is None:
        return default if _is_executable(default) else None
    if not _is_executable(Path(configured)):
        pytest.fail(f"PY_SPY_TEST_BINARY is not an executable file: {configured!r}")
    return Path(configured)


@pytest.fixture
def real_py_spy() -> Path:
    path = _resolve_real_py_spy(
        os.environ.get("PY_SPY_TEST_BINARY"), DEFAULT_REAL_PY_SPY
    )
    if path is None:
        pytest.skip("real py-spy binary not present")
    return path


def _segments(directory: Path, prefix: str) -> list[dict[str, Any]]:
    return [
        event
        for path in sorted(directory.glob(f"{prefix}-*.json"))
        for event in json.loads(path.read_text())
    ]


def test_cluster_and_cloud_access_is_guarded() -> None:
    for attempt in (
        lambda: kubernetes.load_secret("any"),
        kubernetes.config.load_kube_config,
        kubernetes.config.load_incluster_config,
        s3fs.S3FileSystem,
    ):
        with pytest.raises(pytest.fail.Exception):
            attempt()
    assert {n for n in os.environ if n.startswith("AWS_")} == {
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
    }
    assert isinstance(fsspec.filesystem("file"), LocalFileSystem)


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
    assert metrics["descendants"]["sampler_cpu_ticks"] == 11
    assert metrics["descendants"]["sampler_rss_bytes"] == 2 * os.sysconf("SC_PAGE_SIZE")
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
        "import time; time.sleep(1.5)",
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
    assert initial["selftest_samples"] == 2
    assert initial["sampler_overhead_warning"] is False
    assert initial["metadata"] == {"run": "test"}
    assert initial["sampler_overhead_warning_ratio"] == profiling.OVERHEAD_WARNING_RATIO
    assert initial["sampler_overhead"].keys() == {"cpu", "import"}
    for probe in initial["sampler_overhead"].values():
        assert probe == {
            "plain_seconds": [1.0] * 5,
            "sampled_seconds": [1.0] * 5,
            "median_ratio": 1.0,
        }
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
from reformatters.common import profiling
profiling._timed_run = lambda *_: 1.0
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


@pytest.mark.parametrize("exit_code", [0, 7])
def test_real_py_spy_receipt_carries_workload_exit(
    tmp_path: Path,
    real_py_spy: Path,
    exit_code: int,
    capfd: pytest.CaptureFixture[str],
) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    busy = (
        "import sys, time\n"
        "print('workload-stdout-marker', flush=True)\n"
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
        py_spy=real_py_spy,
        disk_paths=[tmp_path],
    )
    assert rc == exit_code
    initial = json.loads((remote / "manifest-initial.json").read_text())
    assert initial["selftest_samples"] > 0
    receipt = json.loads((remote / profiling.WORKLOAD_RECEIPT).read_text())
    assert receipt["returncode"] == exit_code
    final = json.loads((remote / "manifest-final.json").read_text())
    assert final["profile_samples"] > 0
    assert final["py_spy_returncode"] == 0
    assert final["py_spy_samples"] > 0
    assert final["py_spy_error_fraction"] is not None
    assert final["profile_process_samples"]
    assert "workload-stdout-marker" in capfd.readouterr().out
    assert "workload-" not in (tmp_path / "out" / profiling.PY_SPY_LOG).read_text()


def test_real_py_spy_overhead_probes_run_nonblocking(
    tmp_path: Path, real_py_spy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(profiling, "_timed_run", REAL_TIMED_RUN)
    monkeypatch.setattr(profiling, "OVERHEAD_PAIRS", 1)
    monkeypatch.setattr(
        profiling,
        "_OVERHEAD_PROBES",
        {
            "cpu": "import time\nend = time.monotonic() + 1.0\nwhile time.monotonic() < end: pass"
        },
    )

    overhead = profiling._sampler_overhead(real_py_spy, tmp_path / "probe.raw")

    assert overhead["cpu"]["plain_seconds"][0] > 0
    assert overhead["cpu"]["sampled_seconds"][0] > 0


@pytest.mark.parametrize("configured", ["missing", "not-executable", ""])
def test_configured_real_py_spy_must_be_executable(
    tmp_path: Path, configured: str
) -> None:
    (tmp_path / "not-executable").write_text("")
    path = str(tmp_path / configured) if configured else configured
    with pytest.raises(pytest.fail.Exception, match="PY_SPY_TEST_BINARY"):
        _resolve_real_py_spy(path, tmp_path / "absent")


def test_real_py_spy_is_optional_when_not_configured(tmp_path: Path) -> None:
    assert _resolve_real_py_spy(None, tmp_path / "absent") is None
    default = tmp_path / "py-spy"
    default.write_text("")
    default.chmod(0o755)
    assert _resolve_real_py_spy(None, default) == default


def test_profiler_failure_cannot_mask_workload_success(
    tmp_path: Path, fake_py_spy: Path
) -> None:
    fake_py_spy.write_text(
        FAKE_PY_SPY.replace(
            "sys.exit(rc)",
            "sys.exit(11 if out.name == 'py-spy.speedscope.json' else rc)",
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


def _fake_with(fake_py_spy: Path, old: str, new: str) -> None:
    assert old in fake_py_spy.read_text()
    fake_py_spy.write_text(fake_py_spy.read_text().replace(old, new))


def _run_passing_workload(tmp_path: Path, fake_py_spy: Path) -> tuple[int, Any]:
    fs, remote = _remote(tmp_path)
    rc = profiling.run_profiled(
        [sys.executable, "-c", "pass"],
        str(remote),
        fs,
        {},
        tmp_path / "out",
        py_spy=fake_py_spy,
    )
    return rc, json.loads((remote / "manifest-final.json").read_text())


def test_selftest_and_record_sample_without_pausing_the_workload(
    tmp_path: Path, fake_py_spy: Path
) -> None:
    _run_passing_workload(tmp_path, fake_py_spy)

    calls = [
        json.loads(line)
        for line in fake_py_spy.with_suffix(".argv").read_text().splitlines()
    ]
    records = [c for c in calls if c[0] == "record"]
    assert len(records) == 2
    for argv in records:
        assert argv[argv.index("--format") + 1] == "speedscope"
        assert "--nonblocking" in argv
        assert "--subprocesses" in argv
        assert argv[argv.index("--rate") + 1] == "20"


def _scripted_overhead(
    monkeypatch: pytest.MonkeyPatch,
    py_spy: Path,
    sampled_seconds: dict[str, list[float]],
) -> list[tuple[list[str], float]]:
    calls: list[tuple[list[str], float]] = []
    remaining = {k: iter(v) for k, v in sampled_seconds.items()}

    def timed_run(command: list[str], deadline: float) -> float:
        calls.append((list(command), deadline))
        if command[0] != str(py_spy):
            return 1.0
        probe = next(
            k for k, code in profiling._OVERHEAD_PROBES.items() if command[-1] == code
        )
        return next(remaining[probe])

    monkeypatch.setattr(profiling, "_timed_run", timed_run)
    return calls


def test_overhead_probes_are_paired_nonblocking_and_share_a_deadline(
    tmp_path: Path, fake_py_spy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _scripted_overhead(
        monkeypatch,
        fake_py_spy,
        {"cpu": [1.0] * 5, "import": [1.0] * 5},
    )
    start = time.monotonic()

    overhead = profiling._sampler_overhead(fake_py_spy, tmp_path / "probe.raw")

    assert len(calls) == 20
    assert [command[0] == str(fake_py_spy) for command, _ in calls] == [
        False,
        True,
    ] * 10
    sampled = [command for command, _ in calls if command[0] == str(fake_py_spy)]
    assert all("--nonblocking" in command for command in sampled)
    assert len({deadline for _, deadline in calls}) == 1
    assert calls[0][1] <= start + profiling.OVERHEAD_TIMEOUT_S + 1
    assert overhead.keys() == {"cpu", "import"}


def test_overhead_ratio_is_the_median_of_the_pairs(
    tmp_path: Path, fake_py_spy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scripted_overhead(
        monkeypatch,
        fake_py_spy,
        {"cpu": [1.0, 1.1, 1.05, 9.0, 1.0], "import": [1.1] * 5},
    )

    overhead = profiling._sampler_overhead(fake_py_spy, tmp_path / "probe.raw")

    assert overhead["cpu"]["median_ratio"] == 1.05
    assert overhead["import"]["median_ratio"] == 1.1


@pytest.mark.parametrize(
    "sampled_seconds",
    [
        {"cpu": [1.2] * 5, "import": [1.0] * 5},
        {"cpu": [1.0] * 5, "import": [1.2] * 5},
    ],
)
def test_excessive_overhead_flags_measurements_and_runs_work(
    tmp_path: Path,
    fake_py_spy: Path,
    monkeypatch: pytest.MonkeyPatch,
    sampled_seconds: dict[str, list[float]],
) -> None:
    _scripted_overhead(monkeypatch, fake_py_spy, sampled_seconds)
    fs, remote = _remote(tmp_path)
    marker = tmp_path / "started"

    rc = profiling.run_profiled(
        [sys.executable, "-c", f"open({str(marker)!r}, 'w')"],
        str(remote),
        fs,
        {},
        tmp_path / "out",
        py_spy=fake_py_spy,
    )

    assert rc == 0
    assert marker.exists()
    final = json.loads((remote / "manifest-final.json").read_text())
    assert final["sampler_overhead_warning"] is True
    initial = json.loads((remote / "manifest-initial.json").read_text())
    measured = {
        name: probe["sampled_seconds"]
        for name, probe in initial["sampler_overhead"].items()
    }
    assert measured == sampled_seconds
    assert initial["sampler_overhead_warning"] is True


def test_process_group_ignores_zombies_but_waits_for_live_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "killpg", lambda *_: None)
    child = tmp_path / "12"
    child.mkdir()
    stat = child / "stat"
    stat.write_text("12 (worker) Z 1 10 0 0\n")
    assert not profiling._group_alive(10, proc_root=tmp_path)
    stat.write_text("12 (worker) S 1 10 0 0\n")
    assert profiling._group_alive(10, proc_root=tmp_path)
    stat.write_text("12 (worker) R 1 11 0 0\n")
    assert not profiling._group_alive(10, proc_root=tmp_path)


def test_final_manifest_records_sampler_quality_without_raw_output(
    tmp_path: Path, fake_py_spy: Path
) -> None:
    rc, final = _run_passing_workload(tmp_path, fake_py_spy)

    assert rc == 0
    assert final["sampler_nonblocking"] is True
    assert final["py_spy_samples"] == 3
    assert final["py_spy_errors"] == 1
    assert final["py_spy_error_fraction"] == 0.25
    assert final["py_spy_lag_warnings"] == 0
    assert final["py_spy_max_lag_seconds"] is None
    assert final["py_spy_low_confidence"] is False
    for path in (tmp_path / "remote").iterdir():
        assert SECRET not in path.read_text()


def test_high_sampling_error_fraction_is_low_confidence_not_failure(
    tmp_path: Path, fake_py_spy: Path
) -> None:
    _fake_with(fake_py_spy, "Samples: 3 Errors: 1", "Samples: 3 Errors: 2")

    rc, final = _run_passing_workload(tmp_path, fake_py_spy)

    assert rc == 0
    assert final["py_spy_error_fraction"] == 0.4
    assert final["py_spy_low_confidence"] is True


def test_missing_sampler_totals_are_low_confidence_not_failure(
    tmp_path: Path, fake_py_spy: Path
) -> None:
    _fake_with(fake_py_spy, "Samples: 3 Errors: 1", "no totals")

    rc, final = _run_passing_workload(tmp_path, fake_py_spy)

    assert rc == 0
    assert final["py_spy_samples"] is None
    assert final["py_spy_error_fraction"] is None
    assert final["py_spy_low_confidence"] is True


def test_empty_profile_still_fails_a_successful_workload(
    tmp_path: Path, fake_py_spy: Path
) -> None:
    _fake_with(
        fake_py_spy,
        "sys.exit(rc)",
        "if out.name == 'py-spy.speedscope.json':\n"
        "    out.write_text('{\"profiles\": []}')\n"
        "sys.exit(rc)",
    )

    rc, final = _run_passing_workload(tmp_path, fake_py_spy)

    assert rc == 1
    assert final["returncode"] == 0
    assert final["profile_samples"] == 0


def test_empty_selftest_profile_stops_before_workload(
    tmp_path: Path, fake_py_spy: Path
) -> None:
    _fake_with(fake_py_spy, '"samples": [[0], [0]]', '"samples": []')
    with pytest.raises(AssertionError, match="self-test recorded no samples"):
        _run_passing_workload(tmp_path, fake_py_spy)
    assert not (tmp_path / "out" / profiling.WORKLOAD_RECEIPT).exists()


def test_sampler_stats_keep_only_py_spy_numbers() -> None:
    stats = profiling._sampler_stats(
        "py-spy> Sampling process 20 times a second.\n"
        "1.50s behind in sampling, results may be inaccurate.\n"
        "py-spy> 9.48s behind in sampling, results may be inaccurate.\n"
        "print from a module: Samples: 999 Errors: 999\n"
        "print from a module: 500.00s behind in sampling\n"
        "py-spy> Wrote speedscope file to '/a-1/b'. Samples: 19 Errors: 5\n"
    )

    assert stats == {
        "py_spy_samples": 19,
        "py_spy_errors": 5,
        "py_spy_error_fraction": 5 / 24,
        "py_spy_lag_warnings": 2,
        "py_spy_max_lag_seconds": 9.48,
        "py_spy_low_confidence": False,
    }


def test_coverage_shows_a_profile_with_only_parent_processes(
    tmp_path: Path, fake_py_spy: Path
) -> None:
    rc, final = _run_passing_workload(tmp_path, fake_py_spy)

    assert rc == 0
    assert list(final["profile_process_samples"].values()) == [2]
    assert final["profile_child_samples"] == 0
    assert final["profile_parent_only"] is True


def test_coverage_counts_samples_outside_the_parent_processes(
    tmp_path: Path, fake_py_spy: Path
) -> None:
    _fake_with(
        fake_py_spy,
        '"samples": [[0], [0]]}]',
        '"samples": [[0], [0]]}, '
        '{"name": "Process 999999 Thread 2", "samples": [[0]]}]',
    )

    rc, final = _run_passing_workload(tmp_path, fake_py_spy)

    assert rc == 0
    assert final["profile_process_samples"]["999999"] == 1
    assert final["profile_samples"] == 3
    assert final["profile_child_samples"] == 1
    assert final["profile_parent_only"] is False


def test_workload_output_is_inherited_and_kept_from_sampler_log(
    tmp_path: Path, fake_py_spy: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    fs, remote = _remote(tmp_path)
    workload = (
        "import sys; print('workload-stdout-marker'); "
        "print('workload-stderr-marker', file=sys.stderr)"
    )

    rc = profiling.run_profiled(
        [sys.executable, "-c", workload],
        str(remote),
        fs,
        {},
        tmp_path / "out",
        py_spy=fake_py_spy,
    )

    assert rc == 0
    captured = capfd.readouterr()
    assert "workload-stdout-marker" in captured.out
    assert "workload-stderr-marker" in captured.err
    assert "Samples: 3 Errors: 1" in captured.err
    sampler_log = (tmp_path / "out" / profiling.PY_SPY_LOG).read_text()
    assert "Samples: 3 Errors: 1" in sampler_log
    assert "workload-" not in sampler_log
    assert not (remote / profiling.PY_SPY_LOG).exists()


def test_run_profiled_checks_the_injected_proc_root(
    tmp_path: Path, fake_py_spy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    seen: list[Path] = []

    def group_alive(_pgid: int, root: Path) -> bool:
        seen.append(root)
        return False

    monkeypatch.setattr(profiling, "_group_alive", group_alive)
    fs, remote = _remote(tmp_path)

    profiling.run_profiled(
        [sys.executable, "-c", "pass"],
        str(remote),
        fs,
        {},
        tmp_path / "out",
        py_spy=fake_py_spy,
        proc_root=proc_root,
        disk_paths=[tmp_path],
    )

    assert seen == [proc_root, proc_root]


def test_timed_run_stops_at_the_deadline() -> None:
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        REAL_TIMED_RUN(
            [sys.executable, "-c", "import time; time.sleep(30)"], start + 0.2
        )
    assert time.monotonic() - start < 10


def test_failing_overhead_probe_raises() -> None:
    with pytest.raises(subprocess.CalledProcessError):
        REAL_TIMED_RUN(
            [sys.executable, "-c", "raise SystemExit(3)"], time.monotonic() + 30
        )
