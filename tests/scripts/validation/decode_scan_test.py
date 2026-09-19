import os
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from scripts.validation import decode_scan
from scripts.validation.utils import RunContext


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        output_dir=tmp_path,
        validation_url="s3://bucket/noaa-test/v1.icechunk",
        reference_url=None,
        validation_ds=xr.Dataset(),
        reference_ds=None,
        started_at=pd.Timestamp.now(tz="UTC"),
        point1_sel={},
        point2_sel={},
        point1_lat=0.0,
        point1_lon=0.0,
        point2_lat=0.0,
        point2_lon=0.0,
        ensemble_member=None,
        variables=[],
        is_virtual=True,
    )


def test_decode_summary_lines_pass(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.decode_sample_desc = (
        "20 of 100 append-dim regions, 5 leads and 3 levels per group variable"
    )
    ctx.decode_checked_count = 1234
    ctx.decode_failures = []

    lines = decode_scan.decode_summary_lines(ctx)

    assert lines == [
        (
            "1234 references decoded successfully, sampled across "
            "20 of 100 append-dim regions, 5 leads and 3 levels per group variable."
        )
    ]


def test_decode_summary_lines_failures(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.decode_sample_desc = (
        "20 of 100 append-dim regions, 5 leads and 3 levels per group variable"
    )
    ctx.decode_checked_count = 1234
    ctx.decode_failures = ["temperature_2m all-NaN at 2024-01-01T00"]

    lines = decode_scan.decode_summary_lines(ctx)

    assert lines[0].startswith("Decode health failures, sampled across")
    assert lines[-1] == "- FAIL: temperature_2m all-NaN at 2024-01-01T00"


def test_decode_checker_preserves_configured_all_nan_allowlist() -> None:
    configured = validation.CheckVirtualDecodeHealth(
        allow_all_nan_vars=("legitimate_all_nan",)
    )
    dataset = cast(
        "DynamicalDataset[Any, Any]",
        SimpleNamespace(validators=lambda: (configured,)),
    )

    def reference_exists(var_path: str, out_loc: Mapping[str, object]) -> bool:
        return bool(var_path or out_loc)

    checker = decode_scan._decode_checker(dataset, reference_exists)

    assert checker.allow_all_nan_vars == ("legitimate_all_nan",)
    assert checker.reference_exists is reference_exists
    assert (
        checker.positions,
        checker.sampled_leads,
        checker.sampled_levels,
    ) == (
        1,
        decode_scan.SAMPLED_LEADS,
        decode_scan.SAMPLED_LEVELS,
    )


_DATASET_ID = "noaa-test-forecast-virtual"
_START = pd.Timestamp("2024-01-01")
_END = pd.Timestamp("2024-02-01")


def _result(
    passed: bool, checked_count: int, message: str
) -> validation.ValidationResult:
    return validation.ValidationResult(
        passed=passed, message=message, checked_count=checked_count
    )


class _Job:
    """Stub VirtualRegionJob exposing only what the decode scan uses."""

    def __init__(
        self, start: int, var_paths: tuple[str, ...] = ("temperature_2m",)
    ) -> None:
        self.region = slice(start, start + 2)
        self.data_vars = [SimpleNamespace(path=path) for path in var_paths]


class _Checker:
    """Stub CheckVirtualDecodeHealth recording which jobs it actually decoded."""

    def __init__(self, results: dict[int, validation.ValidationResult]) -> None:
        self._results = results
        self.checked: list[int] = []

    def model_dump(self, **_kwargs: object) -> dict[str, Any]:
        return {"sampled_leads": 5, "sampled_levels": 3}

    def check(
        self, context: validation.ValidationContext
    ) -> validation.ValidationResult:
        job = cast("Any", context.region_job)
        self.checked.append(job.region.start)
        return self._results[job.region.start]


def _run_scan_with(
    monkeypatch: pytest.MonkeyPatch,
    ctx: RunContext,
    jobs: list[_Job],
    checker: _Checker,
) -> None:
    """Drive run_decode_scan against stub jobs and a stub checker (no store access)."""
    dataset = SimpleNamespace(
        dataset_id=_DATASET_ID,
        template_config=SimpleNamespace(
            append_dim="init_time",
            data_vars=[],
            get_template=lambda end: None,
        ),
    )
    monkeypatch.setattr(
        decode_scan,
        "resolve_scan_window",
        lambda ctx: (dataset, None, _START, _END),
    )
    monkeypatch.setattr(
        decode_scan.validation,
        "open_flattened_dataset",
        lambda store, consolidated: xr.Dataset(),
    )
    monkeypatch.setattr(decode_scan.zarr, "open_group", lambda store, mode: None)
    monkeypatch.setattr(decode_scan, "_var_keys", lambda *args: None)
    monkeypatch.setattr(decode_scan, "build_virtual_jobs", lambda *args, **kwargs: jobs)
    monkeypatch.setattr(decode_scan, "_decode_checker", lambda *args: checker)
    decode_scan.run_decode_scan(ctx)


def _stub_checkpoint_path(ctx: RunContext, job: _Job) -> Path:
    assert ctx.checkpoint_dir is not None
    key = decode_scan._checkpoint_key(
        _DATASET_ID,
        _START,
        _END,
        ctx.variables,
        decode_scan.MAX_SAMPLED_REGIONS,
        cast("Any", _Checker({})),
    )
    return decode_scan._job_checkpoint_path(
        ctx.checkpoint_dir, _DATASET_ID, key, cast("Any", job)
    )


def test_job_checkpoint_round_trips_a_failing_job(tmp_path: Path) -> None:
    result = _result(
        False, 128, "temperature_2m: every sampled chunk decoded entirely NaN"
    )
    path = tmp_path / "job.json"

    decode_scan._write_job_checkpoint(path, result)

    assert decode_scan._read_job_checkpoint(path) == result


def test_interrupted_checkpoint_write_is_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write that dies before the rename leaves nothing at the path a resume reads."""
    path = tmp_path / "job.json"

    def explode(self: Path, target: Path) -> None:
        raise OSError("interrupted")

    monkeypatch.setattr(Path, "rename", explode)
    with pytest.raises(OSError, match="interrupted"):
        decode_scan._write_job_checkpoint(path, _result(True, 10, "ok"))
    monkeypatch.undo()

    assert not path.exists()
    assert [p.name for p in tmp_path.iterdir()] == [f"job.json.{os.getpid()}.partial"]


def test_resume_decodes_only_the_jobs_without_a_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = [_Job(0), _Job(2), _Job(4)]
    failure = "temperature_2m: every sampled chunk decoded entirely NaN"
    results = {
        0: _result(True, 10, "ok"),
        2: _result(False, 5, failure),
        4: _result(True, 7, "ok"),
    }
    ctx = _ctx(tmp_path)
    ctx.checkpoint_dir = tmp_path / "checkpoints"
    ctx.variables = ["temperature_2m"]

    first = _Checker(results)
    _run_scan_with(monkeypatch, ctx, jobs, first)

    assert sorted(first.checked) == [0, 2, 4]
    assert ctx.decode_checked_count == 22
    assert ctx.decode_failures == [failure]

    # An interruption before the last job: its checkpoint is missing, the others remain.
    assert len(list(ctx.checkpoint_dir.iterdir())) == 3
    _stub_checkpoint_path(ctx, jobs[2]).unlink()

    second = _Checker(results)
    _run_scan_with(monkeypatch, ctx, jobs, second)

    assert second.checked == [4]
    assert ctx.decode_checked_count == 22
    assert ctx.decode_failures == [failure]


def test_no_checkpoint_dir_writes_nothing_and_decodes_every_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = [_Job(0), _Job(2)]
    results = {0: _result(True, 10, "ok"), 2: _result(True, 7, "ok")}
    ctx = _ctx(tmp_path)

    for _ in range(2):
        checker = _Checker(results)
        _run_scan_with(monkeypatch, ctx, jobs, checker)
        assert sorted(checker.checked) == [0, 2]

    assert ctx.decode_checked_count == 17
    assert list(tmp_path.iterdir()) == []


def _key(**overrides: object) -> str:
    kwargs: dict[str, Any] = {
        "dataset_id": _DATASET_ID,
        "start": _START,
        "end": _END,
        "variables": ["temperature_2m", "pressure_surface"],
        "max_samples": decode_scan.MAX_SAMPLED_REGIONS,
        "checker": validation.CheckVirtualDecodeHealth(),
    }
    kwargs.update(overrides)
    return decode_scan._checkpoint_key(**kwargs)


def test_checkpoint_key_separates_everything_that_changes_the_sample() -> None:
    baseline = _key()

    assert _key(variables=["pressure_surface", "temperature_2m"]) == baseline
    assert _key(checker=validation.CheckVirtualDecodeHealth(max_workers=64)) == baseline
    assert _key(max_samples=40) != baseline
    assert _key(variables=["temperature_2m"]) != baseline
    assert _key(start=pd.Timestamp("2024-01-15")) != baseline
    assert _key(end=pd.Timestamp("2024-03-01")) != baseline
    assert _key(dataset_id="other-virtual") != baseline
    assert (
        _key(checker=validation.CheckVirtualDecodeHealth(sampled_levels=5)) != baseline
    )
    assert (
        _key(checker=validation.CheckVirtualDecodeHealth(sampled_leads=7)) != baseline
    )
    assert (
        _key(
            checker=validation.CheckVirtualDecodeHealth(
                allow_all_nan_vars=("temperature_2m",)
            )
        )
        != baseline
    )


def test_job_checkpoint_paths_are_unique_per_job_and_per_scan(tmp_path: Path) -> None:
    key = _key()
    paths = {
        decode_scan._job_checkpoint_path(tmp_path, _DATASET_ID, key, cast("Any", job))
        for job in (_Job(0), _Job(2), _Job(0, ("pressure_surface",)))
    }

    assert len(paths) == 3
    other_scan = decode_scan._job_checkpoint_path(
        tmp_path, _DATASET_ID, _key(max_samples=40), cast("Any", _Job(0))
    )
    assert other_scan not in paths
    assert all(path.name.startswith(f"{_DATASET_ID}_decode_") for path in paths)
