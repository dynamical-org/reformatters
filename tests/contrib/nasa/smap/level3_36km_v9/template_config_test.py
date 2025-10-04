from copy import deepcopy

import numpy as np
import pandas as pd

from reformatters.contrib.nasa.smap.level3_36km_v9.template_config import (
    NasaSmapLevel336KmV9TemplateConfig,
)


def test_get_template_spatial_ref() -> None:
    """Ensure the spatial reference system in the template matched our expectation."""
    template_config = NasaSmapLevel336KmV9TemplateConfig()
    ds = template_config.get_template(
        template_config.append_dim_start + pd.Timedelta(days=10)
    )
    original_attrs = deepcopy(ds.spatial_ref.attrs)

    expected_crs = "EPSG:6933"
    calculated_spatial_ref_attrs = ds.rio.write_crs(expected_crs).spatial_ref.attrs
    assert set(original_attrs) - set(calculated_spatial_ref_attrs) == {"comment"}
    original_attrs.pop("comment")
    assert original_attrs == calculated_spatial_ref_attrs


def test_dimension_coordinates_x_y() -> None:
    """Verify x and y coordinates match EASE-Grid 2.0 specification."""
    template_config = NasaSmapLevel336KmV9TemplateConfig()
    coords = template_config.dimension_coordinates()

    # Expected grid parameters from GDAL info and documentation
    x_size = 964
    y_size = 406
    cell_size = 36032.22  # meters
    ulxmap = -17367530.0  # outer edge of upper-left pixel
    ulymap = 7314540.0  # outer edge of upper-left pixel

    x = coords["x"]
    y = coords["y"]

    # Check array sizes
    assert len(x) == x_size
    assert len(y) == y_size

    # Check x coordinates (west to east)
    # First x should be at center of first cell
    expected_first_x = ulxmap + cell_size / 2
    assert np.isclose(x[0], expected_first_x)

    # Last x should be at center of last cell
    expected_last_x = ulxmap + (x_size - 1) * cell_size + cell_size / 2
    assert np.isclose(x[-1], expected_last_x)

    # Check spacing
    x_spacing = np.diff(x)
    assert np.allclose(x_spacing, cell_size)

    # Check y coordinates (north to south, decreasing)
    # First y should be at center of first cell
    expected_first_y = ulymap - cell_size / 2
    assert np.isclose(y[0], expected_first_y)

    # Last y should be at center of last cell
    expected_last_y = ulymap - (y_size - 1) * cell_size - cell_size / 2
    assert np.isclose(y[-1], expected_last_y)

    # Check spacing (should be negative since decreasing)
    y_spacing = np.diff(y)
    assert np.allclose(y_spacing, -cell_size)

    # Verify symmetry (grid should be centered at origin)
    assert np.isclose(x[0], -x[-1])
    assert np.isclose(y[0], -y[-1])
