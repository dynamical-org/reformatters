from typing import Any

import numpy as np
import pyproj
import pytest

from reformatters.common.projection import latitude_longitude_grids
from reformatters.noaa.rtma.analysis_virtual import template_config
from reformatters.noaa.rtma.analysis_virtual.template_config import (
    NoaaRtmaAnalysisVirtualTemplateConfig,
    spatial_coordinates,
)
from reformatters.noaa.rtma.grib_header import GRID_PROJ


def test_coordinate_statistics_ignore_projection_roundoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = NoaaRtmaAnalysisVirtualTemplateConfig().coords
    y, x = spatial_coordinates()
    monkeypatch.setattr(
        template_config, "spatial_coordinates", lambda: (y + 1e-9, x - 1e-9)
    )
    actual = NoaaRtmaAnalysisVirtualTemplateConfig().coords
    assert [coord.attrs for coord in actual] == [coord.attrs for coord in expected]


def test_grid_coordinates_match_independent_eccodes_points(
    source_fixture: dict[str, Any],
) -> None:
    y, x = spatial_coordinates()
    latitude, longitude = latitude_longitude_grids(GRID_PROJ, x, y)
    assert len(y) == 1597
    assert len(x) == 2345
    assert np.all(np.diff(y) < 0)
    assert np.all(np.diff(x) > 0)
    for point in source_fixture["points"]:
        np.testing.assert_allclose(
            latitude[point["y"], point["x"]], point["latitude"], rtol=0, atol=4e-6
        )
        np.testing.assert_allclose(
            longitude[point["y"], point["x"]], point["longitude"], rtol=0, atol=8e-6
        )
    np.testing.assert_allclose(y[[0, -1]], [3789568.4852873166, -263797.50271268317])
    np.testing.assert_allclose(x[[0, -1]], [-3271147.288294264, 2681916.5437057363])


def test_projection_and_virtual_encodings() -> None:
    config = NoaaRtmaAnalysisVirtualTemplateConfig()
    spatial_ref = next(c for c in config.coords if c.name == "spatial_ref")
    assert pyproj.CRS(spatial_ref.attrs.crs_wkt) == pyproj.CRS(GRID_PROJ)
    assert spatial_ref.attrs.semi_major_axis == 6371200
    for var in config.data_vars:
        assert var.encoding.dtype == "float64"
        assert var.encoding.chunks == (1, 1597, 2345)
        assert var.encoding.shards is None
        assert np.isnan(var.encoding.fill_value)
        assert var.encoding.serializer is not None
        assert var.encoding.serializer["configuration"]["north_up"]
        assert var.encoding.serializer["configuration"]["adjust_longitude_range"]
    for name in ("cloud_ceiling_height", "visibility_surface"):
        var = next(v for v in config.data_vars if v.name == name)
        assert var.attrs.comment is None
        assert not var.encoding.filters
    assert next(
        v for v in config.data_vars if v.name == "wind_gust_10m"
    ).internal_attrs.grib_identity == (0, 2, 22, 103, 10)
