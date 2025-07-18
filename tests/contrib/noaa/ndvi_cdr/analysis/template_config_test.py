import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from reformatters.contrib.noaa.ndvi_cdr.analysis.template_config import (
    NoaaNdviCdrAnalysisTemplateConfig,
)


def test_update_template(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Ensure that `uv run main <dataset-id> update-template` has been run and
    all changes to NoaaNdviCdrAnalysisTemplateConfig are reflected in the on-disk Zarr template.
    """
    template_config = NoaaNdviCdrAnalysisTemplateConfig()
    with open(template_config.template_path() / "zarr.json") as f:
        existing_template = json.load(f)

    test_template_path = tmp_path / "latest.zarr"
    monkeypatch.setattr(
        NoaaNdviCdrAnalysisTemplateConfig,
        "template_path",
        lambda _self: test_template_path,
    )

    template_config.update_template()

    with open(template_config.template_path() / "zarr.json") as f:
        updated_template = json.load(f)

    assert existing_template == updated_template


def test_get_template_spatial_ref() -> None:
    """Ensure the spatial reference system in the template matched our expectation."""
    template_config = NoaaNdviCdrAnalysisTemplateConfig()
    ds = template_config.get_template(
        template_config.append_dim_start + pd.Timedelta(days=10)
    )
    original_attrs = deepcopy(ds.spatial_ref.attrs)

    expected_crs = "EPSG:4326"
    calculated_spatial_ref_attrs = ds.rio.write_crs(expected_crs).spatial_ref.attrs
    assert set(original_attrs) - set(calculated_spatial_ref_attrs) == {"comment"}
    original_attrs.pop("comment")
    assert original_attrs == calculated_spatial_ref_attrs
