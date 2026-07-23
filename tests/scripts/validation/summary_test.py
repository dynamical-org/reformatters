from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scripts.validation.summary import _availability_line, _run_parameters_table
from scripts.validation.utils import RunContext, VariableStats


def _dataset(with_vertical_dim: bool) -> xr.Dataset:
    dims = ("init_time", "lead_time", "latitude", "longitude")
    coords = {
        "init_time": pd.date_range("2020-01-01", periods=2, freq="D"),
        "lead_time": pd.to_timedelta([0, 6], unit="h"),
        "latitude": np.array([10.0, 20.0]),
        "longitude": np.array([30.0, 40.0]),
    }
    if with_vertical_dim:
        dims = (*dims, "pressure_level")
        coords["pressure_level"] = np.array([500.0, 850.0])
    shape = tuple(len(coords[d]) for d in dims)
    return xr.Dataset(
        {"temperature": (dims, np.zeros(shape))},
        coords=coords,
    )


def _ctx(ds: xr.Dataset, tmp_path: Path) -> RunContext:
    return RunContext(
        output_dir=tmp_path,
        validation_url="s3://bucket/test/v1.zarr",
        reference_url=None,
        validation_ds=ds,
        reference_ds=None,
        started_at=pd.Timestamp.now(tz="UTC"),
        point1_sel={"latitude": 0, "longitude": 0},
        point2_sel={"latitude": 1, "longitude": 1},
        point1_lat=10.0,
        point1_lon=30.0,
        point2_lat=20.0,
        point2_lon=40.0,
        ensemble_member=None,
        variables=["temperature"],
    )


def test_run_parameters_table_omits_vertical_level_without_vertical_dim(
    tmp_path: Path,
) -> None:
    table = _run_parameters_table(_ctx(_dataset(with_vertical_dim=False), tmp_path))
    assert not any("Vertical level" in line for line in table)


def test_run_parameters_table_includes_vertical_level_with_vertical_dim(
    tmp_path: Path,
) -> None:
    table = _run_parameters_table(_ctx(_dataset(with_vertical_dim=True), tmp_path))
    assert any("Vertical level" in line for line in table)


def test_availability_line_includes_method_and_handles_unmeasured() -> None:
    measured = VariableStats(
        name="v",
        positions_total=6,
        positions_complete=6,
        availability_method="via co-ingested variables",
    )
    assert (
        _availability_line(measured)
        == "**Availability** — 6 of 6 positions complete (via co-ingested variables)"
    )

    unmeasured = VariableStats(name="v", availability_method="not measured — reason")
    assert _availability_line(unmeasured) == (
        "**Availability** — n/a (not measured — reason)"
    )
