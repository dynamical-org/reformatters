from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from scripts.validation import decode_scan
from scripts.validation.decode_scan import _decode_concurrency
from scripts.validation.utils import READ_BUDGET_BYTES, RunContext


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

    checker = decode_scan._decode_checker(dataset, reference_exists, max_workers=3)

    assert checker.allow_all_nan_vars == ("legitimate_all_nan",)
    assert checker.reference_exists is reference_exists
    assert (
        checker.positions,
        checker.sampled_leads,
        checker.sampled_levels,
        checker.max_workers,
    ) == (
        1,
        decode_scan.SAMPLED_LEADS,
        decode_scan.SAMPLED_LEVELS,
        3,
    )


def _decode_ds(shape: tuple[int, ...], chunks: tuple[int, ...]) -> xr.Dataset:
    dims = ("init_time", "ensemble_member", "lead_time", "y", "x")
    var = xr.DataArray(np.zeros(shape, dtype="float32"), dims=dims)
    var.encoding["chunks"] = chunks
    return xr.Dataset({"var": var})


def test_decode_chunk_span_ignores_the_dims_a_decode_pins() -> None:
    # 100 inits and 60 leads, one chunk each: a decode pins both, so it spans one chunk.
    ds = _decode_ds((100, 1, 60, 4, 4), (1, 1, 1, 4, 4))
    assert decode_scan._decode_chunk_span(ds) == 1

    # 64 members in chunks of 4: a decode covers every member, so it spans 16 chunks.
    ds = _decode_ds((100, 64, 60, 4, 4), (1, 4, 1, 4, 4))
    assert decode_scan._decode_chunk_span(ds) == 16


def test_decode_concurrency_falls_to_one_when_a_single_decode_fills_the_budget() -> (
    None
):
    assert _decode_concurrency(chunk_span=1, chunk_nbytes=2**10) == (
        decode_scan.MAX_JOB_CONCURRENCY,
        decode_scan.MAX_DECODE_CONCURRENCY,
    )
    assert _decode_concurrency(chunk_span=16, chunk_nbytes=READ_BUDGET_BYTES) == (1, 1)
