import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from reformatters.common.config_models import ROOT
from reformatters.dwd.icon_eu.forecast_5_day.template_config import (
    DwdIconEuForecast5DayTemplateConfig,
)
from reformatters.dwd.icon_eu.forecast_5_day_icosahedral.template_config import (
    DwdIconEuForecast5DayIcosahedralTemplateConfig,
)


def test_native_template_preserves_cell_order_and_main_run_forecast_contract() -> None:
    config = DwdIconEuForecast5DayIcosahedralTemplateConfig()
    template = config.get_template(pd.Timestamp("2026-10-07T12")).to_dataset()
    assert_array_equal(
        template.init_time, pd.to_datetime(["2026-10-07T00", "2026-10-07T06"])
    )
    expected_hours = [*range(79), *range(81, 121, 3)]
    assert_array_equal(template.lead_time, pd.to_timedelta(expected_hours, unit="h"))
    assert_array_equal(template.cell, np.arange(768608))
    assert_allclose(
        template.latitude[:5],
        [71.62097247, 71.57862857, 71.57837093, 71.53595393, 71.53545041],
        atol=2e-5,
        rtol=0,
    )
    assert_allclose(
        template.longitude[:5],
        [36.09586398, 36.16467596, 36.35590902, 36.4240284, 36.61488361],
        atol=2e-5,
        rtol=0,
    )
    assert_array_equal(template.valid_time, template.init_time + template.lead_time)
    assert config.dims[ROOT] == ("init_time", "lead_time", "cell")
    for var in config.data_vars:
        assert template[var.name].encoding["chunks"] == (1, 93, 24019)
        assert template[var.name].encoding["shards"] == (1, 93, 768608)
    assert template.cell.encoding["dtype"] == np.dtype("int64")
    assert template.cell.encoding["compressors"][0].shuffle == "bitshuffle"
    regular = DwdIconEuForecast5DayTemplateConfig().get_template(
        pd.Timestamp("2026-02-10T06")
    )
    assert template.spatial_ref.attrs == regular.spatial_ref.attrs


def test_native_variable_metadata_and_transformations_match_regular_forecasts() -> None:
    regular = DwdIconEuForecast5DayTemplateConfig()
    native = DwdIconEuForecast5DayIcosahedralTemplateConfig()
    assert {v.name for v in native.data_vars} == {v.name for v in regular.data_vars}
    for actual, expected in zip(native.data_vars, regular.data_vars, strict=True):
        actual_attrs = actual.attrs.model_dump()
        expected_attrs = expected.attrs.model_dump()
        if actual.name == "wind_gust_10m":
            actual_attrs.pop("comment")
            expected_attrs.pop("comment")
        assert actual_attrs == expected_attrs
        assert actual.internal_attrs == expected.internal_attrs
        assert actual.encoding.dtype == expected.encoding.dtype
        assert actual.encoding.compressors == expected.encoding.compressors
