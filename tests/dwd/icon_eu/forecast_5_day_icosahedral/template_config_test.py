from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from reformatters.common.config_models import ROOT
from reformatters.common.template_utils import write_metadata
from reformatters.dwd.icon_eu.forecast_5_day.template_config import (
    DwdIconEuForecast5DayTemplateConfig,
)
from reformatters.dwd.icon_eu.forecast_5_day_icosahedral.template_config import (
    DwdIconEuForecast5DayIcosahedralTemplateConfig,
)


def test_native_template_preserves_cell_order_and_main_run_forecast_contract() -> None:
    config = DwdIconEuForecast5DayIcosahedralTemplateConfig()
    template = config.get_template(pd.Timestamp("2026-10-07T12")).to_dataset()
    assert_array_equal(
        template.init_time, pd.to_datetime(["2026-10-07T00", "2026-10-07T06"])
    )
    expected_hours = [*range(79), *range(81, 121, 3)]
    assert_array_equal(template.lead_time, pd.to_timedelta(expected_hours, unit="h"))
    assert_array_equal(template.cell, np.arange(768608))
    assert_allclose(
        template.latitude[:5],
        [71.62097247, 71.57862857, 71.57837093, 71.53595393, 71.53545041],
        atol=2e-5,
        rtol=0,
    )
    assert_allclose(
        template.longitude[:5],
        [36.09586398, 36.16467596, 36.35590902, 36.4240284, 36.61488361],
        atol=2e-5,
        rtol=0,
    )
    assert_array_equal(template.valid_time, template.init_time + template.lead_time)
    assert config.dims[ROOT] == ("init_time", "lead_time", "cell")
    regular = DwdIconEuForecast5DayTemplateConfig().get_template(
        pd.Timestamp("2026-02-10T06")
    )
    assert template.spatial_ref.attrs == regular.spatial_ref.attrs
    assert template.wind_gust_10m.attrs["comment"]
    assert "comment" not in regular.wind_gust_10m.attrs


def test_mesh_does_not_multiply_point_forecast_rows() -> None:
    config = DwdIconEuForecast5DayIcosahedralTemplateConfig()
    tree = config.get_template("2026-10-07T12")
    ds = tree.to_dataset()
    assert set(ds.dims) == {"init_time", "lead_time", "cell"}
    frame = ds.isel(cell=0).to_dataframe()
    assert len(frame) == ds.sizes["init_time"] * ds.sizes["lead_time"]
    assert ds.temperature_2m.attrs["mesh"] == "/mesh/mesh"
    grid = tree[ds.temperature_2m.attrs["mesh"].rsplit("/", 1)[0]]
    assert "node" not in ds.dims
    assert grid["mesh"].attrs["face_dimension"] == "cell"


def test_written_mesh_preserves_triangle_and_area_access(tmp_path: Path) -> None:
    config = DwdIconEuForecast5DayIcosahedralTemplateConfig()
    template = config.get_template("2026-10-07T06")
    store = tmp_path / "native.zarr"
    write_metadata(template, store)
    with (
        xr.open_zarr(store, decode_timedelta=True) as ds,
        xr.open_zarr(store, group="mesh") as grid,
    ):
        measure = ds.temperature_2m.attrs["cell_measures"].split()[1]
        assert measure == "/mesh/cell_area"
        triangles = grid.face_node_connectivity.values
        assert triangles.min() == 0
        assert triangles.max() < grid.sizes["node"]
        assert_array_equal(triangles[0], [0, 2500, 1])
        assert_array_equal(
            triangles[-1], template["mesh"].to_dataset().face_node_connectivity[-1]
        )
        assert_allclose(
            grid.node_latitude.isel(node=triangles[0]),
            [71.663330078125, 71.57864379882812, 71.62088012695312],
            atol=2e-5,
            rtol=0,
        )
        assert grid.cell_area.attrs["units"] == "m2"
        assert_allclose(grid.cell_area[0], 43068914.0242895, rtol=1e-7)
        assert ds.attrs["source_grid_id"] == "832c1507-fbc7-3eb3-8413-9890094d2c60"
        assert ds.cell.dtype == np.int32
