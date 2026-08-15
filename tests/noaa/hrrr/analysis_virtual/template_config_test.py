import numpy as np
import pandas as pd

from reformatters.common.config_models import ROOT
from reformatters.noaa.hrrr.analysis_virtual.template_config import (
    NoaaHrrrAnalysisVirtualTemplateConfig,
)
from reformatters.noaa.hrrr.forecast_48_hour_virtual.template_config import (
    NoaaHrrrForecast48HourVirtualTemplateConfig,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar

CONFIG = NoaaHrrrAnalysisVirtualTemplateConfig()


def get_var(path: str) -> NoaaHrrrDataVar:
    return next(v for v in CONFIG.data_vars if v.path == path)


def test_hourly_time_structure() -> None:
    assert CONFIG.append_dim == "time"
    assert CONFIG.append_dim_frequency == pd.Timedelta("1h")
    dims = CONFIG.dims
    assert dims[ROOT] == ("time", "y", "x")
    assert dims["pressure_level"] == ("time", "y", "x", "pressure_level")
    assert dims["model_level"] == ("time", "y", "x", "model_level")


def test_serves_the_whole_forecast_catalog() -> None:
    forecast_vars = NoaaHrrrForecast48HourVirtualTemplateConfig().data_vars
    assert len(CONFIG.data_vars) == 176
    assert [v.path for v in CONFIG.data_vars] == [v.path for v in forecast_vars]


def test_run_total_variables_carry_the_one_hour_equivalence_comment() -> None:
    run_totals = [
        v
        for v in CONFIG.data_vars
        if v.internal_attrs.window_reset_frequency == pd.Timedelta.max
    ]
    assert {v.name for v in run_totals} == {
        "total_precipitation_run_total_surface",
        "snowfall_water_equivalent_run_total_surface",
        "frozen_precipitation_run_total_surface",
        "total_snowfall_run_total_surface",
        "freezing_rain_run_total_surface",
    }
    for var in run_totals:
        assert var.attrs.comment is not None
        assert "one hour" in var.attrs.comment
    assert get_var("total_precipitation_run_total_surface").attrs.comment == (
        "Identical to the one hour accumulated total_precipitation_surface in this "
        "analysis dataset."
    )
    # ASNOW and FRZR have no hourly variant in the catalog to point at.
    assert get_var("total_snowfall_run_total_surface").attrs.comment == (
        "Accumulated over the one hour ending at this time."
    )

    # Variables that are not run totals keep the shared catalog's attrs untouched.
    forecast_attrs = {
        v.path: v.attrs for v in NoaaHrrrForecast48HourVirtualTemplateConfig().data_vars
    }
    run_total_paths = {v.path for v in run_totals}
    assert all(
        var.attrs == forecast_attrs[var.path]
        for var in CONFIG.data_vars
        if var.path not in run_total_paths
    )


def test_one_chunk_per_message_encoding() -> None:
    root_var = get_var("temperature_2m")
    assert root_var.encoding.chunks == (1, 1059, 1799)
    assert root_var.encoding.shards is None
    pressure_var = get_var("pressure_level/temperature")
    assert pressure_var.encoding.chunks == (1, 1059, 1799, 1)
    model_var = get_var("model_level/temperature")
    assert model_var.encoding.chunks == (1, 1059, 1799, 1)


def test_dataset_attributes() -> None:
    attrs = CONFIG.dataset_attributes
    assert attrs.dataset_id == "noaa-hrrr-analysis-virtual"
    assert attrs.dataset_version == "0.1.0"
    assert attrs.name == "NOAA HRRR analysis, virtual"
    assert attrs.license == "CC-BY-4.0"
    assert attrs.time_resolution == "1 hour"
    assert attrs.forecast_domain is None


def test_template_starts_at_hrrr_operational_start() -> None:
    # Matches the materialized noaa-hrrr-analysis rather than the virtual forecast
    # datasets' HRRR v3 start; the whole NODD archive is reachable virtually.
    assert CONFIG.append_dim_start == pd.Timestamp("2014-10-01T00:00")
    template = CONFIG.get_template(pd.Timestamp("2014-10-01T03:00"))
    times = template.to_dataset().get_index("time")
    assert list(times) == [
        pd.Timestamp("2014-10-01T00:00"),
        pd.Timestamp("2014-10-01T01:00"),
        pd.Timestamp("2014-10-01T02:00"),
    ]
    assert np.all(np.diff(times) == pd.Timedelta("1h").to_timedelta64())
