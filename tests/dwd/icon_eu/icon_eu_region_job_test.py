from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from zarr.storage import LocalStore

from reformatters.common import template_utils
from reformatters.common.config_models import Encoding
from reformatters.common.iterating import item
from reformatters.common.types import ArrayFloat32
from reformatters.dwd.icon_eu.icon_eu_region_job import (
    DwdIconEuRegionJob,
    DwdIconEuSourceFileCoord,
)
from reformatters.dwd.icon_eu.icon_eu_template_config import (
    DwdIconEuDataVar,
    derive_forecast_coordinates,
    forecast_coordinate_configs,
    forecast_dimension_coordinates,
    icon_eu_data_vars,
)


class CellSourceFileCoord(DwdIconEuSourceFileCoord):
    def get_url(self) -> str:
        return f"https://archive.example/{self.variable_name_in_filename}/{self.lead_time.total_seconds():.0f}"

    def get_fallback_url(self) -> str:
        return self.get_url().replace("archive.example", "source.example")


class CellRegionJob(DwdIconEuRegionJob[CellSourceFileCoord]):
    source_file_coord_class: ClassVar[type[DwdIconEuSourceFileCoord]] = (
        CellSourceFileCoord
    )

    def read_data(
        self, coord: CellSourceFileCoord, data_var: DwdIconEuDataVar
    ) -> ArrayFloat32:
        assert coord.downloaded_path is not None
        return np.load(coord.downloaded_path).astype(data_var.encoding.dtype)


def test_another_grid_writes_scaled_and_deaccumulated_values_after_archive_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_values = {
        "snow_water_equivalent_surface": [1000, 2000, 3000],
        "precipitation_surface": [0, 280800, 302400],
        "downward_direct_short_wave_radiation_flux_surface": [0, 100, 8400 / 81],
    }
    encoding = Encoding(
        dtype="float32", fill_value=np.nan, chunks=(1, 3, 1), shards=(1, 3, 2)
    )
    data_vars = [v for v in icon_eu_data_vars(encoding) if v.name in source_values]
    init_times = pd.date_range("2026-01-01", periods=1, freq="6h")
    coords = {**forecast_dimension_coordinates(init_times), "cell": [0, 1]}
    lead_times = pd.to_timedelta([0, 78, 81], unit="h")
    coords["lead_time"] = lead_times
    ds = xr.Dataset(
        {
            v.name: template_utils.make_empty_variable(
                ("init_time", "lead_time", "cell"), coords, v.encoding.dtype
            )
            for v in data_vars
        },
        coords,
        attrs={"dataset_id": "test-icon-eu-cells"},
    )
    ds = ds.assign_coords(derive_forecast_coordinates(ds))
    for v in data_vars:
        template_utils.assign_var_metadata(ds[v.name], v)
    for c in forecast_coordinate_configs(init_times, 10):
        template_utils.assign_var_metadata(ds[c.name], c)
    ds.cell.encoding = {"dtype": "int64", "fill_value": -1, "chunks": (2,)}
    template = xr.DataTree(ds)
    store = LocalStore(tmp_path / "output.zarr")
    tmp_store = tmp_path / "tmp.zarr"
    template_utils.write_metadata(template, tmp_path / "output.zarr")
    template_utils.write_metadata(template, tmp_store)

    def download(url: str, dataset_id: str) -> Path:
        parsed = urlparse(url)
        if parsed.hostname == "archive.example":
            raise FileNotFoundError(url)
        assert parsed.hostname == "source.example"
        assert dataset_id == "test-icon-eu-cells"
        filename_var, seconds = parsed.path.strip("/").split("/")
        var = item(
            v
            for v in data_vars
            if v.internal_attrs.variable_name_in_filename == filename_var
        )
        step = list(lead_times.total_seconds()).index(int(seconds))
        path = tmp_path / f"{filename_var}-{seconds}.npy"
        np.save(path, np.full(2, source_values[var.name][step], dtype=np.float32))
        return path

    monkeypatch.setattr(
        "reformatters.dwd.icon_eu.icon_eu_region_job.http_download_to_disk", download
    )
    jobs = CellRegionJob.get_jobs(tmp_store, template, "init_time", data_vars, "test")
    for job in jobs:
        job.process(primary_store=store, replica_stores=[])

    result = xr.open_zarr(store, chunks=None, decode_timedelta=True)
    for name, expected in {
        "snow_water_equivalent_surface": [1, 2, 3],
        "precipitation_surface": [np.nan, 1, 2],
        "downward_direct_short_wave_radiation_flux_surface": [np.nan, 100, 200],
    }.items():
        for cell in result.cell:
            np.testing.assert_allclose(
                result[name].sel(cell=cell).values[0], expected, rtol=0.005
            )
