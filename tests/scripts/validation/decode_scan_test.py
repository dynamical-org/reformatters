from pathlib import Path

import pandas as pd
import xarray as xr

from scripts.validation.decode_scan import decode_summary_lines
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
    ctx.decode_note = "Decode health is sampled: 20 of 100 regions."
    ctx.decode_failures = []

    lines = decode_summary_lines(ctx)

    assert lines[0] == ctx.decode_note
    assert lines[-1] == "All sampled references decoded successfully."


def test_decode_summary_lines_failures(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.decode_note = "Decode health is sampled: 20 of 100 regions."
    ctx.decode_failures = ["temperature_2m all-NaN at 2024-01-01T00"]

    lines = decode_summary_lines(ctx)

    assert lines[-1] == "- FAIL: temperature_2m all-NaN at 2024-01-01T00"
