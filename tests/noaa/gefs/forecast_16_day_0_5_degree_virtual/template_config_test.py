import re
from collections import Counter

import numpy as np
import pandas as pd

from reformatters.common.config_models import ROOT
from reformatters.noaa.gefs.forecast_16_day_0_5_degree_virtual.template_config import (
    NoaaGefsForecast16Day05DegreeVirtualTemplateConfig,
)
from reformatters.noaa.gefs.gefs_config_models import NoaaGefsVirtualDataVar
from reformatters.noaa.noaa_grib_index import grib_index_window_str

CONFIG = NoaaGefsForecast16Day05DegreeVirtualTemplateConfig()


def get_var(path: str) -> NoaaGefsVirtualDataVar:
    return next(v for v in CONFIG.data_vars if v.path == path)


def test_forecast_time_structure() -> None:
    assert CONFIG.append_dim == "init_time"
    assert CONFIG.append_dim_frequency == pd.Timedelta("6h")
    assert CONFIG.append_dim_start == pd.Timestamp("2020-10-01T00:00")


def test_lead_times_step_up_where_the_a_and_b_files_coarsen() -> None:
    """The a and b files publish 3 hourly through 240 hours and 6 hourly to 384."""
    lead_times = CONFIG.dimension_coordinates()["lead_time"]
    assert len(lead_times) == 105
    assert lead_times[0] == pd.Timedelta("0h")
    assert lead_times[-1] == pd.Timedelta("384h")
    fine = lead_times[lead_times <= pd.Timedelta("240h")]
    coarse = lead_times[lead_times >= pd.Timedelta("240h")]
    assert set(np.diff(fine)) == {pd.Timedelta("3h").to_timedelta64()}
    assert set(np.diff(coarse)) == {pd.Timedelta("6h").to_timedelta64()}
    assert pd.Timedelta("243h") not in set(lead_times)


def test_all_thirty_one_ensemble_members() -> None:
    """gec00 plus gep01..gep30, the members GEFS v12 publishes at every cycle."""
    members = CONFIG.dimension_coordinates()["ensemble_member"]
    assert list(members) == list(range(31))


def test_half_degree_grid() -> None:
    dim_coords = CONFIG.dimension_coordinates()
    assert len(dim_coords["latitude"]) == 361
    assert len(dim_coords["longitude"]) == 720
    assert CONFIG.resolution_degrees == 0.5
    assert CONFIG.dataset_attributes.spatial_resolution == "0.5 degrees (~40km)"


def test_one_chunk_per_message_spans_only_the_grid() -> None:
    """A chunk holds one whole GRIB message: the full grid, one init, member, lead and
    -- for a vertical group variable -- one level."""
    assert get_var("temperature_2m").encoding.chunks == (1, 1, 1, 361, 720)
    for path in (
        "pressure_level/temperature",
        "model_level/temperature",
        "height_above_mean_sea_level/temperature",
    ):
        var = get_var(path)
        assert var.encoding.chunks == (1, 1, 1, 361, 720, 1), path
        assert var.encoding.shards is None, path
        assert var.encoding.serializer is not None, path
        assert var.encoding.serializer["name"] == "gribberish", path


def test_group_membership_and_shape() -> None:
    """The three vertical families the a and b files carry on a dense, comparable set
    of levels; everything else keeps its level in the variable name at the root."""
    assert Counter(
        ROOT if v.group is ROOT else v.group for v in CONFIG.data_vars
    ) == Counter({ROOT: 247, "pressure_level": 12, "model_level": 6, "height_above_mean_sea_level": 3})  # fmt: skip

    dim_coords = CONFIG.dimension_coordinates()
    assert len(dim_coords["pressure_level"]) == 31
    assert list(dim_coords["model_level"]) == [1, 2, 3, 4]
    assert len(dim_coords["height_above_mean_sea_level"]) == 8

    for group in ("pressure_level", "model_level", "height_above_mean_sea_level"):
        assert CONFIG.dims[group][-1] == group, group
        assert CONFIG.dims[group][:-1] == CONFIG.dims[ROOT], group


def test_potential_vorticity_stays_at_the_root() -> None:
    """The source's potential vorticity surfaces are a hemisphere convention, not a
    dense comparable set, so each keeps its level in the variable name."""
    pv_vars = [v for v in CONFIG.data_vars if "pvu" in v.name or v.name.endswith("k")]
    assert pv_vars, "expected the PV and isentropic families"
    assert all(v.group is ROOT for v in pv_vars)


def test_surface_geopotential_height_is_not_served() -> None:
    """The a file carries it only at lead 0 and the b file only from lead 3, so no one
    source file supplies the whole lead axis."""
    assert "geopotential_height_surface" not in {v.name for v in CONFIG.data_vars}


def test_a_and_b_files_partition_the_pressure_levels() -> None:
    """Six elements are split across both products level by level; the rest come from
    the b file alone. A variable must name every file it has a message in or
    _check_refs_complete rejects the file that carries the other half."""
    by_files = {
        v.name: sorted(v.internal_attrs.source_file_types)
        for v in CONFIG.data_vars
        if v.group == "pressure_level"
    }
    assert by_files == {
        "geopotential_height": ["a", "b"],
        "temperature": ["a", "b"],
        "relative_humidity": ["a", "b"],
        "wind_u": ["a", "b"],
        "wind_v": ["a", "b"],
        "vertical_velocity": ["a", "b"],
        "specific_humidity": ["b"],
        "ozone_mixing_ratio": ["b"],
        "absolute_vorticity": ["b"],
        "cloud_mixing_ratio": ["b"],
        "icing_probability": ["b"],
        "icing_severity": ["b"],
    }
    assert all(
        v.internal_attrs.source_file_types == frozenset({"b"})
        for v in CONFIG.data_vars
        if v.group in ("model_level", "height_above_mean_sea_level")
    )


def test_the_cloud_mixing_ratio_element_was_respelled_mid_archive() -> None:
    """The index spells it CLWMR before 2025-12-19 and CLMR after; a file carries one
    spelling, so both must be matched."""
    var = get_var("pressure_level/cloud_mixing_ratio")
    assert var.internal_attrs.grib_element == "CLMR"
    assert var.internal_attrs.grib_element_alternatives == ("CLWMR",)


def test_window_comments_are_phrased_in_lead_time() -> None:
    """An analysis names UTC clock hours; a forecast's window is set by lead time, and
    the same wall clock hour carries a different window in each init."""
    windowed = [v for v in CONFIG.data_vars if v.attrs.step_type != "instant"]
    assert len(windowed) == 45
    for var in windowed:
        assert var.internal_attrs.window_reset_frequency == pd.Timedelta("6h"), var.name
        if var.attrs.flag_values is None:
            assert var.attrs.comment is not None, var.name
            assert "UTC" not in var.attrs.comment, var.name
            assert (
                "(lead times 6, 12, 18, ... hours) or 3 hour period "
                "(lead times 3, 9, 15, ... hours)"
            ) in var.attrs.comment, var.name

    assert get_var("total_precipitation_surface").attrs.comment == (
        "Total accumulated in the last 6 hour period (lead times 6, 12, 18, ... hours) "
        "or 3 hour period (lead times 3, 9, 15, ... hours)."
    )


def test_every_lead_time_carries_the_window_its_comment_promises() -> None:
    """The comment claims a 6 hour window at lead times 6, 12, 18, ... and a 3 hour one
    at 3, 9, 15, ...; the idx window string the region job matches on is what decides.
    Enumerated over every windowed variable and every lead, because a reset-frequency
    or day-form-window slip shows up only at particular leads (240 and 384 hours are a
    whole number of days, where a running total would render "0-16 day acc fcst")."""
    lead_hours = [int(t.total_seconds() // 3600) for t in CONFIG.lead_times()]
    windowed = [v for v in CONFIG.data_vars if v.attrs.step_type != "instant"]
    assert {240, 384} <= set(lead_hours)

    for var in windowed:
        for lead in lead_hours:
            if lead == 0 and not var.has_hour_0_values():
                continue
            window = grib_index_window_str(var, lead)
            match = re.fullmatch(r"(\d+)-(\d+) hour \w+ fcst", window)
            assert match is not None, (var.name, lead, window)
            start, end = int(match.group(1)), int(match.group(2))
            assert end == lead, (var.name, lead, window)
            assert end - start == (6 if lead % 6 == 0 else 3), (var.name, lead, window)


def test_flag_variables_carry_only_their_codes() -> None:
    """flag_values and flag_meanings are the whole meaning of a categorical variable, so
    it carries no comment: a window sentence would contradict them by describing a
    fraction, and restating the codes in prose would let the two representations drift.
    """
    for path in (
        "categorical_snow_surface",
        "categorical_ice_pellets_surface",
        "categorical_freezing_rain_surface",
        "categorical_rain_surface",
    ):
        var = get_var(path)
        assert var.attrs.flag_values == (0, 1), path
        assert var.attrs.flag_meanings == "no yes", path
        assert var.attrs.comment is None, path

    # GRIB2 code table 4.207, whose figures are not in severity order.
    severity = get_var("pressure_level/icing_severity")
    assert severity.attrs.flag_values == (0, 1, 2, 3, 4, 5)
    assert severity.attrs.flag_meanings == "none light moderate severe trace heavy"
    assert severity.attrs.comment is None


def test_instant_variables_the_source_omits_at_lead_zero() -> None:
    """Three convective cloud fields are instantaneous yet absent from the lead 0 file,
    so the default step_type rule would request a message that is not there."""
    omitted = {
        v.name for v in CONFIG.data_vars
        if v.attrs.step_type == "instant" and not v.has_hour_0_values()
    }  # fmt: skip
    assert omitted == {
        "pressure_convective_cloud_bottom",
        "pressure_convective_cloud_top",
        "convective_cloud_cover",
    }


def test_dataset_attributes() -> None:
    attrs = CONFIG.dataset_attributes
    assert attrs.dataset_id == "noaa-gefs-forecast-16-day-0-5-degree-virtual"
    assert attrs.name == "NOAA GEFS forecast 16 day 0.5 degree, virtual"
    assert attrs.dataset_version == "0.1.0"
    assert attrs.time_domain == "Forecasts initialized 2020-10-01 00:00:00 UTC to Present"  # fmt: skip
    assert attrs.time_resolution == "Forecasts initialized every 6 hours"
    assert attrs.forecast_domain == "Forecast lead time 0-384 hours ahead"
    # The axis coarsens at 240 hours, so naming only the finer step would be false
    # about the 246-384 hour leads.
    assert attrs.forecast_resolution == (
        "Forecast step 0-240 hours: 3 hourly, 246-384 hours: 6 hourly"
    )


def test_template_carries_the_forecast_coordinates_in_every_group() -> None:
    """A group is opened on its own, so it repeats the shared coordinates."""
    template = CONFIG.get_template(pd.Timestamp("2020-10-01T12:00"))
    for node in template.subtree:
        ds = node.to_dataset()
        assert list(ds.get_index("init_time")) == [
            pd.Timestamp("2020-10-01T00:00"),
            pd.Timestamp("2020-10-01T06:00"),
        ], node.path
        assert ds["valid_time"].dims == ("init_time", "lead_time"), node.path
        assert ds["valid_time"].isel(init_time=0, lead_time=-1) == pd.Timestamp(
            "2020-10-17T00:00"
        ), node.path
        assert set(np.unique(ds["expected_forecast_length"].values)) == {
            pd.Timedelta("384h").to_timedelta64()
        }, node.path
