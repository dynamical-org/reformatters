import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from reformatters.contrib.u_arizona.swann.analysis.template_config import (
    UarizonaSwannAnalysisTemplateConfig,
)


def test_update_template(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    template_config = UarizonaSwannAnalysisTemplateConfig()
    with open(template_config.template_path() / "zarr.json") as f:
        existing_template = json.load(f)

    test_template_path = tmp_path / "latest.zarr"
    monkeypatch.setattr(
        UarizonaSwannAnalysisTemplateConfig,
        "template_path",
        lambda _self: test_template_path,
    )

    template_config.update_template()

    with open(template_config.template_path() / "zarr.json") as f:
        updated_template = json.load(f)

    assert existing_template == updated_template


def test_get_template_spatial_ref() -> None:
    template_config = UarizonaSwannAnalysisTemplateConfig()
    ds = template_config.get_template(
        template_config.append_dim_start + pd.Timedelta(days=5)
    )
    original_attrs = deepcopy(ds.spatial_ref.attrs)

    # See https://nsidc.org/sites/default/files/documents/user-guide/nsidc-0719-v001-userguide.pdf
    # which indicates EPSG:4269 is the CRS for the source data.
    calculated_spatial_ref_attrs = ds.rio.write_crs("EPSG:4269").spatial_ref.attrs
    assert set(original_attrs) - set(calculated_spatial_ref_attrs) == {"comment"}
    original_attrs.pop("comment")
    assert original_attrs == calculated_spatial_ref_attrs
