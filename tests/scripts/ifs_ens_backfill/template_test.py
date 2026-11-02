import json

import numpy as np
import pandas as pd

from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.deprecated_zarr_v3.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3TemplateConfig,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
)


def test_origin_preserves_array_encodings_and_independent_zarr() -> None:
    cfg = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    zarr_cfg = EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3TemplateConfig()
    assert cfg.append_dim_start == pd.Timestamp("2020-10-15")
    assert zarr_cfg.append_dim_start == pd.Timestamp("2024-04-01")
    assert cfg.append_dim_coordinate_chunk_size() == 5840
    assert zarr_cfg.append_dim_coordinate_chunk_size() == 5840
    for path in cfg.template_path().glob("*/zarr.json"):
        actual = json.loads(path.read_text())
        published = json.loads(
            (
                zarr_cfg.template_path() / path.relative_to(cfg.template_path())
            ).read_text()
        )
        actual.pop("attributes")
        published.pop("attributes")
        assert actual == published, path.parent.name


def test_regenerated_template_covers_prefix_and_original_boundary() -> None:
    cfg = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    ds = cfg.get_template(pd.Timestamp("2024-04-03")).dataset
    assert ds.sizes["init_time"] == 1266
    assert len(ds.data_vars) == 19
    np.testing.assert_array_equal(
        ds.init_time.values[[0, 1263, 1264, 1265]],
        pd.to_datetime(
            ["2020-10-15", "2024-03-31", "2024-04-01", "2024-04-02"]
        ).to_numpy(),
    )
    np.testing.assert_array_equal(
        ds.valid_time.values, ds.init_time.values[:, None] + ds.lead_time.values
    )
