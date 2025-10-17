# import json
# from copy import deepcopy
# from pathlib import Path

# import pandas as pd
# import pytest

# from reformatters.example.template_config import ExampleTemplateConfig

# def test_get_template_spatial_ref() -> None:
#     """Ensure the spatial reference system in the template matched our expectation."""
#     template_config = ExampleTemplateConfig()
#     ds = template_config.get_template(
#         template_config.append_dim_start + pd.Timedelta(days=10)
#     )
#     original_attrs = deepcopy(ds.spatial_ref.attrs)

#     expected_crs = "Your dataset's proj4 or EPSG:XXXX string"
#     calculated_spatial_ref_attrs = ds.rio.write_crs(expected_crs).spatial_ref.attrs
#     original_attrs.pop("comment")
#     assert original_attrs == calculated_spatial_ref_attrs
