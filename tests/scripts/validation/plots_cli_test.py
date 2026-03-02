from pathlib import Path

import numpy as np
import xarray as xr
from _pytest.monkeypatch import MonkeyPatch
from typer.testing import CliRunner

from scripts.validation import utils
from scripts.validation.plots import app


def test_report_nulls_accepts_start_alias(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    ds = xr.Dataset(
        data_vars={
            "precipitation_surface": xr.DataArray(
                np.full((2, 8, 8), 1.0, dtype=np.float32),
                dims=("time", "latitude", "longitude"),
            )
        },
        coords={
            "time": np.array(
                ["2023-01-01T00:00:00", "2023-01-01T01:00:00"], dtype="datetime64[ns]"
            ),
            "latitude": np.linspace(40.0, 41.0, 8, dtype=np.float32),
            "longitude": np.linspace(-120.0, -119.0, 8, dtype=np.float32),
        },
        attrs={"dataset_id": "test-dataset"},
    )
    dataset_path = tmp_path / "test.zarr"
    ds.to_zarr(dataset_path)

    monkeypatch.setattr(utils, "OUTPUT_DIR", str(tmp_path / "output"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "report-nulls",
            str(dataset_path),
            "--start",
            "2023-01-01",
            "--variable",
            "precipitation_surface",
        ],
    )

    assert result.exit_code == 0, result.output
