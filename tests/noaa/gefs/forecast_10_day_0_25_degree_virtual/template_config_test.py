import re

import numpy as np
import pandas as pd

from reformatters.common.config_models import ROOT
from reformatters.noaa.gefs.forecast_10_day_0_25_degree_virtual.template_config import (
    NoaaGefsForecast10Day025DegreeVirtualTemplateConfig,
)
from reformatters.noaa.gefs.gefs_config_models import NoaaGefsVirtualDataVar
from reformatters.noaa.noaa_grib_index import _lead_time_str

CONFIG = NoaaGefsForecast10Day025DegreeVirtualTemplateConfig()


def get_var(name: str) -> NoaaGefsVirtualDataVar:
    return next(v for v in CONFIG.data_vars if v.name == name)


def test_forecast_time_structure() -> None:
    assert CONFIG.append_dim == "init_time"
    assert CONFIG.append_dim_frequency == pd.Timedelta("6h")
    assert CONFIG.append_dim_start == pd.Timestamp("2020-09-23T12:00")
    assert CONFIG.dims == {
        ROOT: ("init_time", "ensemble_member", "lead_time", "latitude", "longitude")
    }


def test_lead_times_cover_the_s_file() -> None:
    """The 0.25 degree s file publishes 3 hourly and stops at 240 hours."""
    lead_times = CONFIG.dimension_coordinates()["lead_time"]
    assert len(lead_times) == 81
    assert lead_times[0] == pd.Timedelta("0h")
    assert lead_times[-1] == pd.Timedelta("240h")
    assert set(np.diff(lead_times)) == {pd.Timedelta("3h").to_timedelta64()}


def test_all_thirty_one_ensemble_members() -> None:
    """gec00 plus gep01..gep30, the members GEFS v12 publishes at every cycle."""
    members = CONFIG.dimension_coordinates()["ensemble_member"]
    assert list(members) == list(range(31))
    coord = next(c for c in CONFIG.coords if c.name == "ensemble_member")
    assert coord.encoding.dtype == "int16"
    assert coord.encoding.chunks == 31
    assert coord.attrs.standard_name == "realization"


def test_one_chunk_per_message_spans_only_the_grid() -> None:
    """A chunk holds one whole GRIB message: the full grid, one init, member and lead."""
    var = get_var("temperature_2m")
    assert var.encoding.chunks == (1, 1, 1, 721, 1440)
    assert var.encoding.shards is None
    assert var.encoding.compressors == ()
    assert var.encoding.serializer is not None
    assert var.encoding.serializer["name"] == "gribberish"


def test_serves_the_s_file_inventory_except_surface_geopotential_height() -> None:
    assert len(CONFIG.data_vars) == 38
    assert "geopotential_height_surface" not in {v.name for v in CONFIG.data_vars}


def test_window_comments_are_phrased_in_lead_time() -> None:
    """An analysis names UTC clock hours; a forecast's window is set by lead time, and
    the same wall clock hour carries a different window in each init."""
    windowed = [v for v in CONFIG.data_vars if v.attrs.step_type != "instant"]
    assert len(windowed) == 15
    for var in windowed:
        assert var.internal_attrs.window_reset_frequency == pd.Timedelta("6h"), var.name
        if var.attrs.flag_values is None:
            assert var.attrs.comment is not None, var.name
            assert "UTC" not in var.attrs.comment, var.name
            assert (
                "(lead times 6, 12, 18, ... hours) or 3 hour period "
                "(lead times 3, 9, 15, ... hours)"
            ) in var.attrs.comment, var.name

    assert get_var("total_cloud_cover_atmosphere").attrs.comment == (
        "Average value in the last 6 hour period (lead times 6, 12, 18, ... hours) "
        "or 3 hour period (lead times 3, 9, 15, ... hours)."
    )
    assert get_var("total_precipitation_surface").attrs.comment == (
        "Total accumulated in the last 6 hour period (lead times 6, 12, 18, ... hours) "
        "or 3 hour period (lead times 3, 9, 15, ... hours). Subtracting the value at an "
        "earlier lead time with the same window start gives the exact total between "
        "those two lead times."
    )


def test_every_lead_time_carries_the_window_its_comment_promises() -> None:
    """The comment claims a 6 hour window at lead times 6, 12, 18, ... and a 3 hour one
    at 3, 9, 15, ...; the idx window string the region job matches on is what decides.
    Enumerated over every windowed variable and every lead, because a reset-frequency
    or day-form-window slip shows up only at particular leads (240 hours is a whole
    number of days, where a running total would render "0-10 day acc fcst")."""
    lead_hours = [int(t.total_seconds() // 3600) for t in CONFIG.lead_times()]
    windowed = [v for v in CONFIG.data_vars if v.attrs.step_type != "instant"]
    assert len(windowed) == 15
    assert 240 in lead_hours

    for var in windowed:
        for lead in lead_hours:
            if lead == 0 and not var.has_hour_0_values():
                continue
            window = _lead_time_str(var, lead)
            match = re.fullmatch(r"(\d+)-(\d+) hour \w+ fcst", window)
            assert match is not None, (var.name, lead, window)
            start, end = int(match.group(1)), int(match.group(2))
            assert end == lead, (var.name, lead, window)
            assert end - start == (6 if lead % 6 == 0 else 3), (var.name, lead, window)


def test_differencing_only_holds_within_a_window_start() -> None:
    """The accumulation comment qualifies differencing by "the same window start". The
    qualification is load bearing: consecutive lead times usually reset in between, so
    an unqualified subtraction would be wrong far more often than right."""
    accumulated = get_var("total_precipitation_surface")
    lead_hours = [int(t.total_seconds() // 3600) for t in CONFIG.lead_times() if t > pd.Timedelta(0)]  # fmt: skip
    starts = {lead: int(_lead_time_str(accumulated, lead).split("-")[0]) for lead in lead_hours}  # fmt: skip

    assert starts[6] == 0, starts
    assert starts[9] == 6, starts
    shared = [(a, b) for a in lead_hours for b in lead_hours if a < b and starts[a] == starts[b]]  # fmt: skip
    assert len(shared) == 40
    assert all(starts[a] == starts[b] < a for a, b in shared)


def test_flag_variables_carry_only_their_codes() -> None:
    """flag_values and flag_meanings are the whole meaning of a categorical variable, so
    it carries no comment: a window sentence would contradict them by describing a
    fraction, and restating the codes in prose would let the two representations drift.
    """
    for name in (
        "categorical_snow_surface",
        "categorical_ice_pellets_surface",
        "categorical_freezing_rain_surface",
        "categorical_rain_surface",
    ):
        var = get_var(name)
        assert var.attrs.flag_values == (0, 1), name
        assert var.attrs.flag_meanings == "no yes", name
        assert var.attrs.comment is None, name


def test_dataset_attributes() -> None:
    attrs = CONFIG.dataset_attributes
    assert attrs.dataset_id == "noaa-gefs-forecast-10-day-0-25-degree-virtual"
    assert attrs.name == "NOAA GEFS forecast, 10 day, 0.25 degree, virtual"
    assert attrs.dataset_version == "0.1.0"
    assert attrs.spatial_resolution == "0.25 degrees (~20km)"
    assert attrs.time_domain == "Forecasts initialized 2020-09-23 12:00:00 UTC to Present"  # fmt: skip
    assert attrs.time_resolution == "Forecasts initialized every 6 hours"
    assert attrs.forecast_domain == "Forecast lead time 0-240 hours ahead"
    assert attrs.forecast_resolution == "Forecast step 3 hourly"


def test_template_carries_the_forecast_coordinates() -> None:
    template = CONFIG.get_template(pd.Timestamp("2020-09-24T00:00")).to_dataset()
    assert list(template.get_index("init_time")) == [
        pd.Timestamp("2020-09-23T12:00"),
        pd.Timestamp("2020-09-23T18:00"),
    ]
    assert template["valid_time"].dims == ("init_time", "lead_time")
    assert template["valid_time"].isel(init_time=0, lead_time=-1) == pd.Timestamp(
        "2020-10-03T12:00"
    )
    expected_length = template["expected_forecast_length"]
    assert expected_length.dims == ("init_time",)
    assert set(np.unique(expected_length.values)) == {
        pd.Timedelta("240h").to_timedelta64()
    }
