from copy import deepcopy

import pandas as pd

from reformatters.contrib.uarizona.swann.analysis.template_config import (
    UarizonaSwannAnalysisTemplateConfig,
)


def test_get_template_spatial_ref() -> None:
    template_config = UarizonaSwannAnalysisTemplateConfig()
    ds = template_config.get_template(
        template_config.append_dim_start + pd.Timedelta(days=5)
    )
    original_attrs = deepcopy(ds.spatial_ref.attrs)

    # See https://nsidc.org/sites/default/files/documents/user-guide/nsidc-0719-v001-userguide.pdf
    # which indicates EPSG:4269 is the CRS for the source data.
    calculated_spatial_ref_attrs = ds.rio.write_crs("EPSG:4269").spatial_ref.attrs
    original_attrs.pop("comment")
    assert original_attrs == calculated_spatial_ref_attrs
