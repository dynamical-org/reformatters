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
    assert template_config.epsg == expected_crs
    calculated_spatial_ref_attrs = ds.rio.write_crs(expected_crs).spatial_ref.attrs
    assert set(original_attrs) - set(calculated_spatial_ref_attrs) == {"comment"}
    original_attrs.pop("comment")
    assert original_attrs == calculated_spatial_ref_attrs


def test_dimension_coordinates_x_y() -> None:
    """Verify x and y coordinates match EASE-Grid 2.0 specification."""
    template_config = NasaSmapLevel336KmV9TemplateConfig()
    coords = template_config.dimension_coordinates()

    x = coords["x"]
    y = coords["y"]

    assert len(x) == template_config.x_size
    assert len(y) == template_config.y_size

    expected_first_x = template_config.upper_left_x + template_config.cell_size / 2
    assert np.isclose(x[0], expected_first_x)

    expected_last_x = (
        template_config.upper_left_x
        + (template_config.x_size - 1) * template_config.cell_size
        + template_config.cell_size / 2
    )
    assert np.isclose(x[-1], expected_last_x)

    x_spacing = np.diff(x)
    assert np.allclose(x_spacing, template_config.cell_size)

    expected_first_y = template_config.upper_left_y - template_config.cell_size / 2
    assert np.isclose(y[0], expected_first_y)

    expected_last_y = (
        template_config.upper_left_y
        - (template_config.y_size - 1) * template_config.cell_size
        - template_config.cell_size / 2
    )
    assert np.isclose(y[-1], expected_last_y)

    y_spacing = np.diff(y)
    assert np.allclose(y_spacing, -template_config.cell_size)

    assert np.isclose(x[0], -x[-1])
    assert np.isclose(y[0], -y[-1])


def test_latitude_longitude_coordinates() -> None:
    """Verify latitude and longitude 2D coordinates are computed correctly."""
    template_config = NasaSmapLevel336KmV9TemplateConfig()
    ds = template_config.get_template(
        template_config.append_dim_start + pd.Timedelta(days=10)
    )

    lat = ds.latitude.values
    lon = ds.longitude.values

    assert lat.shape == (406, 964)
    assert lon.shape == (406, 964)

    assert np.min(lat) >= -85.044502
    assert np.max(lat) <= 85.044502

    assert np.min(lon) >= -180.0
    assert np.max(lon) <= 180.0

    assert lat[0, 0] > lat[-1, 0]

    mid_y = lat.shape[0] // 2
    assert lon[mid_y, 0] < lon[mid_y, -1]

    assert np.all(np.isfinite(lat))
    assert np.all(np.isfinite(lon))
