import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

import fsspec

from reformatters.common import materialized_region_job, parallel_coordination, storage
from reformatters.common.materialized_region_job import MaterializedRegionJob
from reformatters.common.region_job import SourceFileStatus

PY_SPY = Path("/opt/profiler/bin/py-spy")
SAMPLE_RATE_HZ = 20
METRICS_INTERVAL_S = 1.0
UPLOAD_INTERVAL_S = 30.0
DRAIN_S = 25.0
FLUSH_S = 15.0
PY_SPY_OUTPUT = "py-spy.speedscope.json"
WORKLOAD_RECEIPT = "workload-exit.json"

_JOB_METHOD_PHASES = {
    "process": "unit",
    "_download_processing_group": "download",
    "_read_into_data_array": "read",
    "apply_data_transformations": "transform",
    "_write_shards": "local_write",
}
_MODULE_FUNCTION_PHASES: tuple[tuple[Any, str, str], ...] = (
    (materialized_region_job, "copy_data_var", "upload"),
    (storage, "commit_if_icechunk", "commit"),
    (parallel_coordination, "parallel_setup", "setup"),
    (parallel_coordination, "wait_for_workers", "wait"),
    (parallel_coordination, "collect_results", "wait"),
    (parallel_coordination, "finalize", "finalize"),
)


class SourceFileFailureError(RuntimeError):
    pass


_active_phases = threading.local()


def _now() -> tuple[str, int]:
    return datetime.now(UTC).isoformat(), time.monotonic_ns()


def _write_immutable(path: Path, payload: object) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True))
    tmp.rename(path)


class _PhaseLog:
    def __init__(self, output_dir: Path) -> None:
        self._dir = output_dir
        self._lock = threading.Lock()
        self._seq = 0

    def record(self, event: dict[str, Any]) -> None:
        with self._lock:
            path = self._dir / f"phase-{os.getpid()}-{self._seq:06d}.json"
            self._seq += 1
        _write_immutable(path, [event])


def _job_labels(job: MaterializedRegionJob[Any, Any]) -> dict[str, Any]:
    return {
        "region_start": job.region.start,
        "region_stop": job.region.stop,
        "data_vars": [v.path for v in job.data_vars],
        "download_parallelism": job.download_parallelism,
        "read_parallelism": job.read_parallelism,
        "max_vars_per_download_group": job.max_vars_per_download_group,
        "cpu_count": os.cpu_count(),
    }


def _phase_labels(phase: str, args: tuple[Any, ...]) -> dict[str, Any]:
    match phase:
        case "unit":
            return _job_labels(args[0])
        case "download":
            return {**_unit_ref(args[0]), "data_vars": list(args[2])}
        case "read" | "transform":
            return {**_unit_ref(args[0]), "data_var": args[2].path}
        case "local_write":
            return {**_unit_ref(args[0]), "data_var": args[1].name}
        case "upload":
            return {
                "data_var": args[0],
                "region_start": args[1].start,
                "region_stop": args[1].stop,
            }
        case _:
            return {}


def _unit_ref(job: MaterializedRegionJob[Any, Any]) -> dict[str, Any]:
    return {"region_start": job.region.start, "region_stop": job.region.stop}


def _check_statuses(label: str, coords: Sequence[Any], ok: SourceFileStatus) -> None:
    failed = [c.status.name for c in coords if c.status != ok]
    if failed:
        raise SourceFileFailureError(
            f"{len(failed)} of {len(coords)} source files not {ok.name} for "
            f"{label}: {sorted(set(failed))}"
        )


def _check_result(phase: str, args: tuple[Any, ...], result: Any) -> None:  # noqa: ANN401
    match phase:
        case "download":
            _check_statuses(str(args[2]), result, SourceFileStatus.Processing)
        case "read":
            _check_statuses(args[2].path, result, SourceFileStatus.Succeeded)
        case "unit":
            for var_path, coords in result.items():
                _check_statuses(var_path, coords, SourceFileStatus.Succeeded)


def _timed[**P, R](fn: Callable[P, R], phase: str, log: _PhaseLog) -> Callable[P, R]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        active = _active_phases.__dict__.setdefault("phases", set())
        if phase in active:
            return fn(*args, **kwargs)
        active.add(phase)
        start_utc, start_ns = _now()
        error_type = None
        try:
            result = fn(*args, **kwargs)
            _check_result(phase, args, result)
            return result
        except BaseException as e:
            error_type = type(e).__name__
            raise
        finally:
            active.discard(phase)
            end_utc, end_ns = _now()
            thread = threading.current_thread()
            log.record(
                {
                    "kind": "phase",
                    "phase": phase,
                    "start_utc": start_utc,
                    "end_utc": end_utc,
                    "start_monotonic_ns": start_ns,
                    "end_monotonic_ns": end_ns,
                    "pid": os.getpid(),
                    "thread_id": thread.ident,
                    "thread_name": thread.name,
                    "ok": error_type is None,
                    "error_type": error_type,
                    **_phase_labels(phase, args),
                }
            )

    return wrapper


def _classes_defining(name: str) -> list[type]:
    found: list[type] = []
    pending: list[type] = [MaterializedRegionJob]
    while pending:
        cls = pending.pop()
        if name in vars(cls):
            found.append(cls)
        pending.extend(cls.__subclasses__())
    return found


@contextlib.contextmanager
def install_hooks(output_dir: Path) -> Iterator[None]:
    """Time the materialized processing boundaries of this process into immutable
    ``phase-<pid>-<seq>.json`` segments in ``output_dir``.

    Region job classes must be imported before entering. Any source file that fails
    to download or read raises ``SourceFileFailureError``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log = _PhaseLog(output_dir)
    originals: list[tuple[Any, str, Any]] = [
        (cls, name, vars(cls)[name])
        for name in _JOB_METHOD_PHASES
        for cls in _classes_defining(name)
    ]
    originals += [(m, name, getattr(m, name)) for m, name, _ in _MODULE_FUNCTION_PHASES]
    phases = {**_JOB_METHOD_PHASES, **{n: p for _, n, p in _MODULE_FUNCTION_PHASES}}
    try:
        for owner, name, original in originals:
            setattr(owner, name, _timed(original, phases[name], log))
        yield
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def _read_kv(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    return {
        k: int(v) for k, v in (line.split() for line in path.read_text().splitlines())
    }


def _read_text(path: Path) -> str | None:
    return path.read_text().strip() if path.exists() else None


def _read_int(path: Path) -> int | None:
    return int(path.read_text().split()[0]) if path.exists() else None


def _io_totals(path: Path) -> dict[str, int]:
    totals: dict[str, int] = {}
    if not path.exists():
        return totals
    for line in path.read_text().splitlines():
        for field in line.split()[1:]:
            key, _, value = field.partition("=")
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def _proc_tree(proc_root: Path, root_pid: int) -> dict[str, int]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    parents: dict[int, int] = {}
    stats: dict[int, tuple[int, int]] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        with contextlib.suppress(OSError):
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            rss_pages = int((entry / "statm").read_text().split()[1])
            pid = int(entry.name)
            parents[pid] = int(fields[1])
            stats[pid] = (rss_pages * page_size, int(fields[11]) + int(fields[12]))
    tree = {root_pid}
    changed = True
    while changed:
        children = {p for p, pp in parents.items() if pp in tree} - tree
        changed = bool(children)
        tree |= children
    present = [stats[p] for p in tree if p in stats]
    return {
        "processes": len(present),
        "rss_bytes": sum(r for r, _ in present),
        "cpu_ticks": sum(t for _, t in present),
    }


def _net_totals(path: Path) -> dict[str, int]:
    rx = tx = 0
    if path.exists():
        for line in path.read_text().splitlines()[2:]:
            iface, _, counters = line.partition(":")
            if iface.strip() == "lo":
                continue
            values = counters.split()
            rx += int(values[0])
            tx += int(values[8])
    return {"rx_bytes": rx, "tx_bytes": tx}


def _disk(path: Path) -> dict[str, Any]:
    st = os.statvfs(path)
    return {
        "path": str(path),
        "total_bytes": st.f_blocks * st.f_frsize,
        "used_bytes": (st.f_blocks - st.f_bfree) * st.f_frsize,
    }


def sample_metrics(
    cgroup_root: Path, proc_root: Path, root_pid: int, disk_paths: Sequence[Path]
) -> dict[str, Any]:
    utc, mono = _now()
    memory_stat = _read_kv(cgroup_root / "memory.stat")
    return {
        "kind": "resource",
        "utc": utc,
        "monotonic_ns": mono,
        "cgroup_cpu": _read_kv(cgroup_root / "cpu.stat"),
        "cgroup_cpu_max": _read_text(cgroup_root / "cpu.max"),
        "cgroup_memory_current_bytes": _read_int(cgroup_root / "memory.current"),
        "cgroup_memory_peak_bytes": _read_int(cgroup_root / "memory.peak"),
        "cgroup_memory": {
            k: memory_stat[k] for k in ("anon", "file", "shmem") if k in memory_stat
        },
        "cgroup_io": _io_totals(cgroup_root / "io.stat"),
        "descendants": _proc_tree(proc_root, root_pid),
        "network": _net_totals(proc_root / "net" / "dev"),
        "disks": [_disk(p) for p in disk_paths if p.exists()],
    }


def py_spy_selftest(py_spy: Path, work_dir: Path) -> dict[str, Any]:
    output = work_dir / "py-spy-selftest.raw"
    busy = "import time\nt = time.monotonic()\nwhile time.monotonic() - t < 2: pass"
    version = subprocess.run(  # noqa: S603
        [str(py_spy), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(  # noqa: S603
        [
            str(py_spy),
            "record",
            "--subprocesses",
            "--rate",
            str(SAMPLE_RATE_HZ),
            "--format",
            "raw",
            "--output",
            str(output),
            "--",
            sys.executable,
            "-c",
            busy,
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    samples = sum(
        int(line.rsplit(" ", 1)[1])
        for line in output.read_text().splitlines()
        if line.strip()
    )
    assert samples > 0, "py-spy self-test recorded no samples"
    return {"py_spy_version": version, "selftest_samples": samples}


def _speedscope_samples(path: Path) -> int:
    try:
        profile = json.loads(path.read_text())
        return sum(len(p.get("samples", [])) for p in profile["profiles"])
    except OSError, ValueError, KeyError, TypeError:
        return 0


def _run_workload(receipt: Path, command: list[str]) -> int:
    child = subprocess.Popen(command)  # noqa: S603
    signal.signal(signal.SIGTERM, lambda *_: child.terminate())
    signal.signal(signal.SIGINT, lambda *_: child.terminate())
    returncode = child.wait()
    _write_immutable(receipt, {"returncode": returncode, "ended_utc": _now()[0]})
    return returncode


def _exit_code(returncode: int) -> int:
    return 128 - returncode if returncode < 0 else returncode


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Evidence:
    def __init__(
        self, fs: fsspec.AbstractFileSystem, artifact_url: str, output_dir: Path
    ) -> None:
        self.fs = fs
        self.url = artifact_url.rstrip("/")
        self.dir = output_dir
        self.uploaded: dict[str, str] = {}

    def put(self, path: Path) -> None:
        self.fs.put_file(str(path), f"{self.url}/{path.name}")
        self.uploaded[path.name] = _sha256(path)

    def put_closed_segments(self) -> None:
        for path in sorted(self.dir.glob("*-*.json")):
            if path.name not in self.uploaded and path.name != PY_SPY_OUTPUT:
                self.put(path)

    def pipe(self, name: str, payload: object) -> None:
        self.fs.pipe_file(
            f"{self.url}/{name}", json.dumps(payload, sort_keys=True).encode()
        )


def run_profiled(  # noqa: PLR0915
    command: list[str],
    artifact_url: str,
    fs: fsspec.AbstractFileSystem,
    metadata: dict[str, Any],
    output_dir: Path,
    *,
    py_spy: Path = PY_SPY,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_root: Path = Path("/proc"),
    disk_paths: Sequence[Path] = (Path("/app/data"), Path("/dev/shm")),  # noqa: S108
    upload_interval_s: float = UPLOAD_INTERVAL_S,
    drain_s: float = DRAIN_S,
    flush_s: float = FLUSH_S,
) -> int:
    """Run ``command`` under py-spy in its own process group, persisting resource
    metrics, phase segments and manifests to ``artifact_url``. Remote evidence is
    never deleted. Returns the workload's exit code from its receipt; a missing
    receipt, failed evidence or an empty profile turns success into 1.

    ``metadata`` is recorded as given; the command line, environment and exception
    messages are never recorded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = _Evidence(fs, artifact_url, output_dir)
    selftest = py_spy_selftest(py_spy, output_dir)
    started_utc, _ = _now()
    evidence.pipe(
        "manifest-initial.json",
        {"metadata": metadata, "started_utc": started_utc, **selftest},
    )

    stop = threading.Event()
    signal_name: list[str] = []

    def on_signal(signum: int, _frame: object) -> None:
        signal_name.append(signal.Signals(signum).name)
        stop.set()

    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
    proc = subprocess.Popen(  # noqa: S603
        [
            str(py_spy),
            "record",
            "--subprocesses",
            "--rate",
            str(SAMPLE_RATE_HZ),
            "--format",
            "speedscope",
            "--output",
            str(output_dir / PY_SPY_OUTPUT),
            "--",
            sys.executable,
            "-m",
            __name__,
            str(output_dir / WORKLOAD_RECEIPT),
            *command,
        ],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
    )
    pgid = proc.pid
    sampler_started = time.monotonic()
    samples: list[dict[str, Any]] = []
    segment = 0
    evidence_error: str | None = None
    last_upload = time.monotonic()

    def close_resource_segment() -> None:
        nonlocal segment, samples
        if samples:
            _write_immutable(output_dir / f"resource-{segment:06d}.json", samples)
            segment += 1
            samples = []

    killed = False
    try:
        while proc.poll() is None and not stop.is_set():
            samples.append(sample_metrics(cgroup_root, proc_root, pgid, disk_paths))
            if time.monotonic() - last_upload >= upload_interval_s:
                close_resource_segment()
                evidence.put_closed_segments()
                last_upload = time.monotonic()
            stop.wait(METRICS_INTERVAL_S)
    except Exception as e:  # noqa: BLE001
        evidence_error = type(e).__name__
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + drain_s
        while _group_alive(pgid) and time.monotonic() < deadline:
            proc.poll()
            time.sleep(0.1)
        if _group_alive(pgid):
            killed = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
        py_spy_returncode = proc.wait()
        for s, handler in previous.items():
            signal.signal(s, handler)

    with contextlib.suppress(Exception):
        samples.append(sample_metrics(cgroup_root, proc_root, pgid, disk_paths))
    close_resource_segment()
    py_spy_file = output_dir / PY_SPY_OUTPUT
    profile_samples = _speedscope_samples(py_spy_file)
    receipt = output_dir / WORKLOAD_RECEIPT
    returncode = (
        json.loads(receipt.read_text())["returncode"] if receipt.exists() else None
    )
    final = {
        "sampler_rate_hz": SAMPLE_RATE_HZ,
        "sampler_elapsed_seconds": time.monotonic() - sampler_started,
        "py_spy_returncode": py_spy_returncode,
        "profile_samples": profile_samples,
        "metadata": metadata,
        "started_utc": started_utc,
        "ended_utc": _now()[0],
        "returncode": returncode,
        "signal": signal_name[0] if signal_name else None,
        "killed_after_drain": killed,
        "evidence_error": evidence_error,
    }

    flushed = threading.Event()

    def flush() -> None:
        try:
            evidence.put_closed_segments()
            if py_spy_file.exists():
                evidence.put(py_spy_file)
        except Exception as e:  # noqa: BLE001
            final["evidence_error"] = final["evidence_error"] or type(e).__name__
        with contextlib.suppress(Exception):
            evidence.pipe("manifest-final.json", {**final, "sha256": evidence.uploaded})
            if final["evidence_error"] is None:
                flushed.set()

    flusher = threading.Thread(target=flush, daemon=True)
    flusher.start()
    flusher.join(flush_s)
    exit_code = 1 if returncode is None else _exit_code(returncode)
    if (
        not flushed.is_set()
        or not profile_samples
        or py_spy_returncode != 0
        or killed
        or signal_name
    ):
        return exit_code or 1
    return exit_code


if __name__ == "__main__":
    sys.exit(_run_workload(Path(sys.argv[1]), sys.argv[2:]))
