from copy import deepcopy

import pandas as pd

from reformatters.contrib.noaa.ndvi_cdr.analysis.template_config import (
    NoaaNdviCdrAnalysisTemplateConfig,
)


def test_get_template_spatial_ref() -> None:
    """Ensure the spatial reference system in the template matched our expectation."""
    template_config = NoaaNdviCdrAnalysisTemplateConfig()
    ds = template_config.get_template(
        template_config.append_dim_start + pd.Timedelta(days=10)
    )
    original_attrs = deepcopy(ds.spatial_ref.attrs)

    expected_crs = "EPSG:4326"
    calculated_spatial_ref_attrs = ds.rio.write_crs(expected_crs).spatial_ref.attrs
    original_attrs.pop("comment")
    assert original_attrs == calculated_spatial_ref_attrs
