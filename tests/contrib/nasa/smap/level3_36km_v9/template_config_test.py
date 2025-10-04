from copy import deepcopy

import pandas as pd
from reformatters.nasa.smap.level3_36km_v9.template_config import (
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
