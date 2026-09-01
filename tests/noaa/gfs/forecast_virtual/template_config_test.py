import numpy as np
import pandas as pd

from reformatters.common.config_models import ROOT
from reformatters.noaa.gfs.forecast_virtual.template_config import (
    NoaaGfsForecastVirtualTemplateConfig,
)
from reformatters.noaa.gfs.virtual_template_config import NoaaGfsVirtualTemplateConfig
from reformatters.noaa.models import NoaaDataVar

CONFIG = NoaaGfsForecastVirtualTemplateConfig()


def get_var(path: str) -> NoaaDataVar:
    return next(v for v in CONFIG.data_vars if v.path == path)


def test_forecast_time_structure() -> None:
    assert CONFIG.append_dim == "init_time"
    assert CONFIG.append_dim_frequency == pd.Timedelta("6h")
    # The first cycle of the 0.25 degree archive. Prepending to an append dim later is a
    # breaking change, so the inclusive start is the only one that stays available.
    assert CONFIG.append_dim_start == pd.Timestamp("2021-03-22T12:00")
    assert CONFIG.dims[ROOT] == ("init_time", "lead_time", "latitude", "longitude")
    assert CONFIG.dims["pressure_level"] == (
        "init_time",
        "lead_time",
        "latitude",
        "longitude",
        "pressure_level",
    )


def test_lead_times_are_hourly_then_three_hourly() -> None:
    lead_times = CONFIG.dimension_coordinates()["lead_time"]
    assert len(lead_times) == 209
    assert lead_times[0] == pd.Timedelta("0h")
    assert lead_times[-1] == pd.Timedelta("384h")
    # The source publishes no f121 or f122; the spacing changes inside the one dim.
    hourly = lead_times[lead_times <= pd.Timedelta("120h")]
    assert len(hourly) == 121
    assert (hourly.diff()[1:] == pd.Timedelta("1h")).all()
    three_hourly = lead_times[lead_times >= pd.Timedelta("120h")]
    assert (three_hourly.diff()[1:] == pd.Timedelta("3h")).all()


def test_serves_the_whole_shared_catalog() -> None:
    """The running totals are forecast-only: only here does the source render them a
    window string that differs from the 6 hour bucket's."""
    shared = {v.path for v in NoaaGfsVirtualTemplateConfig._catalog_data_vars(CONFIG)}
    served = {v.path for v in CONFIG.data_vars}
    assert served == shared
    assert len(served) == 295
    assert {
        "total_precipitation_run_total_surface",
        "convective_precipitation_run_total_surface",
    } <= served


def test_one_chunk_holds_one_grib_message() -> None:
    assert get_var("temperature_2m").encoding.chunks == (1, 1, 721, 1440)
    assert get_var("pressure_level/temperature").encoding.chunks == (
        1,
        1,
        721,
        1440,
        1,
    )
    for var in CONFIG.data_vars:
        assert var.encoding.shards is None, var.path
        assert list(var.encoding.compressors or []) == [], var.path
        assert var.internal_attrs.keep_mantissa_bits == "no-rounding", var.path


def test_expected_forecast_length_is_the_whole_lead_set() -> None:
    template = CONFIG.get_template(pd.Timestamp("2021-03-23T00:00")).to_dataset()
    assert (
        template["expected_forecast_length"].values == np.timedelta64(384, "h")
    ).all()
    assert template["valid_time"].dims == ("init_time", "lead_time")


def test_windowed_variables_describe_their_window_in_forecast_lead_time() -> None:
    """The analysis sibling phrases the same windows in UTC times because it has no
    lead_time dim; here the window is a property of the step."""
    windowed = [v for v in CONFIG.data_vars if v.attrs.step_type != "instant"]
    assert len(windowed) == 44
    for var in windowed:
        comment = var.attrs.comment
        assert comment is not None, var.name
        assert comment.startswith(
            (
                "Accumulated over",
                "Accumulated from",
                "Averaged over",
                "Maximum over",
                "Minimum over",
            )
        ), var.name

    assert get_var("total_precipitation_surface").attrs.comment == (
        "Accumulated over the 6 hour window containing this step: the window opens at "
        "the most recent multiple of 6 hours of forecast lead time and closes at this "
        "step, so it lengthens from 1 to 6 hours and restarts every 6 hours rather "
        "than covering a fixed interval. Subtracting the value at an earlier step in "
        "the same window gives the exact total between those two steps."
    )
    assert get_var("total_precipitation_run_total_surface").attrs.comment == (
        "Accumulated from the forecast initialization time to this step, so the window "
        "lengthens with lead time and never resets. Subtracting the value at an "
        "earlier step gives the exact total between those two steps."
    )
    assert get_var("albedo_surface").attrs.comment == (
        "Averaged over the 6 hour window containing this step: the window opens at the "
        "most recent multiple of 6 hours of forecast lead time and closes at this "
        "step, so it lengthens from 1 to 6 hours and restarts every 6 hours rather "
        "than covering a fixed interval."
    )
    assert str(get_var("maximum_temperature_2m").attrs.comment).startswith(
        "Maximum over the 6 hour window"
    )
    assert str(get_var("minimum_temperature_2m").attrs.comment).startswith(
        "Minimum over the 6 hour window"
    )
    # Only accumulations can be differenced; an average over a lengthening window cannot.
    assert "Subtracting" not in str(get_var("albedo_surface").attrs.comment)


def test_a_windowed_variables_own_comment_survives_the_window_sentence() -> None:
    comment = str(get_var("water_runoff_surface").attrs.comment)
    assert comment.startswith("Accumulated over the 6 hour window")
    assert comment.endswith("NaN over water, where this quantity does not apply.")


def test_instantaneous_variables_take_no_window_comment() -> None:
    assert get_var("temperature_2m").attrs.comment is None
    assert "window" not in str(get_var("pressure_level/temperature").attrs.comment)
