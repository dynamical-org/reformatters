from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
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
