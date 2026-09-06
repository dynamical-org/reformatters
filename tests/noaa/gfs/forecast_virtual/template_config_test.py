import numpy as np
import pandas as pd

from reformatters.common.config_models import ROOT
from reformatters.noaa.gfs.forecast_virtual.template_config import (
    NoaaGfsForecastVirtualTemplateConfig,
)
from reformatters.noaa.gfs.virtual_template_config import NoaaGfsVirtualTemplateConfig
from reformatters.noaa.models import NoaaDataVar
from reformatters.noaa.noaa_grib_index import grib_index_window_str

CONFIG = NoaaGfsForecastVirtualTemplateConfig()


def get_var(path: str) -> NoaaDataVar:
    return next(v for v in CONFIG.data_vars if v.path == path)


def test_forecast_time_structure() -> None:
    assert CONFIG.append_dim == "init_time"
    assert CONFIG.append_dim_frequency == pd.Timedelta("6h")
    assert CONFIG.append_dim_start == pd.Timestamp("2021-05-01T00:00")
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
    assert len(served) == 274
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
    template = CONFIG.get_template(pd.Timestamp("2021-05-02T00:00")).to_dataset()
    assert (
        template["expected_forecast_length"].values == np.timedelta64(384, "h")
    ).all()
    assert template["valid_time"].dims == ("init_time", "lead_time")


def test_windowed_variables_describe_their_window_in_forecast_lead_time() -> None:
    """The window is a property of the forecast step, so it is phrased in lead time."""
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
                "Maximum value over",
                "Minimum value over",
            )
        ), var.name

    assert get_var("total_precipitation_surface").attrs.comment == (
        "Accumulated over the preceding 1-6 hours of forecast lead time, since the "
        "last lead time divisible by 6 before this step."
    )
    assert get_var("total_precipitation_run_total_surface").attrs.comment == (
        "Accumulated from the forecast initialization time to this step."
    )
    assert get_var("albedo_surface").attrs.comment == (
        "Averaged over the preceding 1-6 hours of forecast lead time, since the last "
        "lead time divisible by 6 before this step. Exactly 0 wherever the averaging "
        "window received no sunlight; exclude those zeros from a time mean albedo."
    )
    assert str(get_var("maximum_temperature_2m").attrs.comment).startswith(
        "Maximum value over the preceding 1-6 hours"
    )
    assert str(get_var("minimum_temperature_2m").attrs.comment).startswith(
        "Minimum value over the preceding 1-6 hours"
    )


def _window_hours(var: NoaaDataVar, lead_hours: int) -> tuple[int, int]:
    """The window start and end the source's own index string declares for this step."""
    text = grib_index_window_str(var, lead_hours)
    head, unit = text.split(" ")[0], text.split(" ")[1]
    start, _, end = head.partition("-")
    scale = 24 if unit == "day" else 1
    assert int(end) * scale == lead_hours
    return int(start) * scale, int(end) * scale


def _published_lead_hours() -> list[int]:
    return [
        int(lead / pd.Timedelta("1h"))
        for lead in CONFIG.dimension_coordinates()["lead_time"]
        if lead > pd.Timedelta(0)
    ]


def test_the_bucket_comment_describes_every_window_length_the_source_produces() -> None:
    """The comment claims 1-6 hours; these are the lengths the source really produces.

    The hourly leads produce every length from 1 to 6 and the 3-hourly tail only 3 and
    6, so the range the comment gives is exact rather than an upper bound.
    """
    bucket = get_var("total_precipitation_surface")
    lead_hours = _published_lead_hours()
    assert len(lead_hours) == 208
    lengths = {
        lead: end - start
        for lead in lead_hours
        for start, end in [_window_hours(bucket, lead)]
    }

    hourly = [lead for lead in lead_hours if lead <= 120]
    three_hourly = [lead for lead in lead_hours if lead > 120]
    assert (len(hourly), len(three_hourly)) == (120, 88)
    assert {lengths[lead] for lead in hourly} == {1, 2, 3, 4, 5, 6}
    assert {lengths[lead] for lead in three_hourly} == {3, 6}
    for lead in lead_hours:
        assert (lengths[lead] == 6) == (lead % 6 == 0), lead
    for lead in three_hourly:
        assert (lengths[lead] == 3) == (lead % 6 == 3), lead


def test_a_windowed_variables_own_comment_survives_the_window_sentence() -> None:
    comment = str(get_var("water_runoff_surface").attrs.comment)
    assert comment.startswith("Accumulated over the preceding 1-6 hours")
    assert comment.endswith("NaN over water.")


def test_instantaneous_variables_take_no_window_comment() -> None:
    assert get_var("temperature_2m").attrs.comment is None
    assert "window" not in str(get_var("pressure_level/temperature").attrs.comment)
