import re
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from reformatters.common.config_models import ROOT
from reformatters.common.time_utils import whole_hours
from reformatters.noaa.gefs.forecast_10_day_0_25_degree_virtual.template_config import (
    NoaaGefsForecast10Day025DegreeVirtualTemplateConfig,
)
from reformatters.noaa.gefs.forecast_16_day_0_5_degree_virtual.template_config import (
    NoaaGefsForecast16Day05DegreeVirtualTemplateConfig,
)
from reformatters.noaa.gefs.forecast_35_day_0_5_degree_virtual.template_config import (
    NoaaGefsForecast35Day05DegreeVirtualTemplateConfig,
)
from reformatters.noaa.gefs.gefs_config_models import (
    GEFSSourceFileType,
    NoaaGefsVirtualDataVar,
)
from reformatters.noaa.gefs.virtual_template_config import (
    NoaaGefsForecastABVirtualTemplateConfig,
    NoaaGefsForecastVirtualTemplateConfig,
)
from reformatters.noaa.noaa_grib_index import grib_index_window_str

# Every config serving the 0.5 degree a and b products. They share one catalog, so a
# variable's identity, group, metadata and geometry must not depend on which forecast
# length a reader opens.
AB_CONFIGS = [
    NoaaGefsForecast16Day05DegreeVirtualTemplateConfig(),
    NoaaGefsForecast35Day05DegreeVirtualTemplateConfig(),
]
AB_CONFIG_PARAMS = pytest.mark.parametrize(
    "config", AB_CONFIGS, ids=[c.dataset_attributes.dataset_id for c in AB_CONFIGS]
)

THREE_HOURLY_TO_240 = pd.timedelta_range("0h", "240h", freq="3h")


def tail(start: str, end: str, freq: str) -> pd.TimedeltaIndex:
    """THREE_HOURLY_TO_240 continued by lead times from `start` to `end` every `freq`."""
    return pd.TimedeltaIndex(
        THREE_HOURLY_TO_240.append(pd.timedelta_range(start, end, freq=freq))
    )


def get_var(
    config: NoaaGefsForecastVirtualTemplateConfig, path: str
) -> NoaaGefsVirtualDataVar:
    return next(v for v in config.data_vars if v.path == path)


def config_with_lead_times(
    leads: pd.TimedeltaIndex,
) -> NoaaGefsForecastVirtualTemplateConfig:
    class Config(NoaaGefsForecastVirtualTemplateConfig):
        source_file_types: tuple[GEFSSourceFileType, ...] = ("s",)

        def lead_times(self) -> pd.TimedeltaIndex:
            return leads

    return Config(forecast_length=leads[-1])


def test_ten_day_lead_times_are_described() -> None:
    """3 hourly through the 240 hour lead where the 0.25 degree s file ends."""
    assert len(THREE_HOURLY_TO_240) == 81
    config_with_lead_times(THREE_HOURLY_TO_240)


def test_sixteen_day_lead_times_are_described() -> None:
    """3 hourly to 240 hours, then 6 hourly to the 384 hour lead."""
    leads = tail("246h", "384h", "6h")
    assert len(leads) == 105
    config_with_lead_times(leads)


def test_thirty_five_day_lead_times_are_described() -> None:
    """3 hourly to 240 hours, then 6 hourly to the 840 hour lead."""
    leads = tail("246h", "840h", "6h")
    assert len(leads) == 181
    config_with_lead_times(leads)


def test_lead_times_the_window_comments_cannot_describe_are_rejected() -> None:
    """A 4 hourly tail puts lead times off both sequences window_comments enumerates."""
    leads = tail("244h", "384h", "4h")
    assert len(leads) == 117
    with pytest.raises(
        ValidationError,
        match="window_comments does not describe lead time 244 hours, whose window is 4 hours",
    ):
        config_with_lead_times(leads)


def test_forecast_length_shorter_than_the_first_window_is_described() -> None:
    """A single 3 hour lead time, the shortest domain the wording has to cover."""
    config = NoaaGefsForecastVirtualTemplateConfig(
        source_file_types=("s",), forecast_length=pd.Timedelta("3h")
    )
    assert list(config.lead_times()) == [pd.Timedelta("0h"), pd.Timedelta("3h")]


def test_window_comments_name_the_lead_times_that_carry_each_window() -> None:
    """Every "N hour period (lead times a, b, c, ... hours)" clause the wording emits,
    read back and checked against the window string the source's own idx line carries
    at each named lead. A clause naming the wrong leads mislabels every windowed value
    a reader averages or differences."""
    config = config_with_lead_times(THREE_HOURLY_TO_240)
    windowed = next(var for var in config.data_vars if var.attrs.step_type == "accum")

    clauses = re.findall(
        r"(\d+) hour period \(lead times ([\d, ]+), \.\.\. hours\)",
        config.window_comments["accum"],
    )
    assert len(clauses) == 2, config.window_comments["accum"]

    for window_hours, named_leads in clauses:
        for lead in [int(hours) for hours in named_leads.split(", ")]:
            source_window = grib_index_window_str(windowed, lead)
            match = re.fullmatch(r"(\d+)-(\d+) hour acc fcst", source_window)
            assert match is not None, (lead, source_window)
            start, end = int(match.group(1)), int(match.group(2))
            assert end == lead, (lead, source_window)
            assert end - start == int(window_hours), (lead, source_window)


@AB_CONFIG_PARAMS
def test_all_thirty_one_ensemble_members(
    config: NoaaGefsForecastABVirtualTemplateConfig,
) -> None:
    """gec00 plus gep01..gep30, the members GEFS v12 publishes at every cycle."""
    assert list(config.dimension_coordinates()["ensemble_member"]) == list(range(31))


@AB_CONFIG_PARAMS
def test_half_degree_grid(config: NoaaGefsForecastABVirtualTemplateConfig) -> None:
    dim_coords = config.dimension_coordinates()
    assert len(dim_coords["latitude"]) == 361
    assert len(dim_coords["longitude"]) == 720
    assert config.resolution_degrees == 0.5
    assert config.dataset_attributes.spatial_resolution == "0.5 degrees (~40km)"


@AB_CONFIG_PARAMS
def test_one_chunk_per_message_spans_only_the_grid(
    config: NoaaGefsForecastABVirtualTemplateConfig,
) -> None:
    """A chunk holds one whole GRIB message: the full grid, one init, member, lead and
    -- for a vertical group variable -- one level."""
    assert get_var(config, "temperature_2m").encoding.chunks == (1, 1, 1, 361, 720)
    for path in (
        "pressure_level/temperature",
        "model_level/temperature",
        "height_above_mean_sea_level/temperature",
    ):
        var = get_var(config, path)
        assert var.encoding.chunks == (1, 1, 1, 361, 720, 1), path
        assert var.encoding.shards is None, path
        assert var.encoding.serializer is not None, path
        assert var.encoding.serializer["name"] == "gribberish", path


@AB_CONFIG_PARAMS
def test_group_membership_and_shape(
    config: NoaaGefsForecastABVirtualTemplateConfig,
) -> None:
    """The three vertical families the a and b files carry on a dense, comparable set
    of levels; everything else keeps its level in the variable name at the root."""
    assert Counter(
        ROOT if v.group is ROOT else v.group for v in config.data_vars
    ) == Counter({ROOT: 247, "pressure_level": 12, "model_level": 6, "height_above_mean_sea_level": 3})  # fmt: skip

    dim_coords = config.dimension_coordinates()
    assert len(dim_coords["pressure_level"]) == 31
    assert list(dim_coords["model_level"]) == [1, 2, 3, 4]
    assert len(dim_coords["height_above_mean_sea_level"]) == 8

    for group in ("pressure_level", "model_level", "height_above_mean_sea_level"):
        assert config.dims[group][-1] == group, group
        assert config.dims[group][:-1] == config.dims[ROOT], group


@AB_CONFIG_PARAMS
def test_potential_vorticity_stays_at_the_root(
    config: NoaaGefsForecastABVirtualTemplateConfig,
) -> None:
    """The source's potential vorticity surfaces are a hemisphere convention, not a
    dense comparable set, so each keeps its level in the variable name."""
    pv_vars = [v for v in config.data_vars if "pvu" in v.name or v.name.endswith("k")]
    assert pv_vars, "expected the PV and isentropic families"
    assert all(v.group is ROOT for v in pv_vars)


@AB_CONFIG_PARAMS
def test_surface_geopotential_height_is_not_served(
    config: NoaaGefsForecastABVirtualTemplateConfig,
) -> None:
    """The a file carries it only at lead 0 and the b file only from lead 3, so no one
    source file supplies the whole lead axis."""
    assert "geopotential_height_surface" not in {v.name for v in config.data_vars}


@AB_CONFIG_PARAMS
def test_a_and_b_files_partition_the_pressure_levels(
    config: NoaaGefsForecastABVirtualTemplateConfig,
) -> None:
    """Six elements are split across both products level by level; the rest come from
    the b file alone. A variable must name every file it has a message in or
    _check_refs_complete rejects the file that carries the other half."""
    by_files = {
        v.name: sorted(v.internal_attrs.source_file_types)
        for v in config.data_vars
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
        for v in config.data_vars
        if v.group in ("model_level", "height_above_mean_sea_level")
    )


@AB_CONFIG_PARAMS
def test_the_cloud_mixing_ratio_element_was_respelled_mid_archive(
    config: NoaaGefsForecastABVirtualTemplateConfig,
) -> None:
    """The index spells it CLWMR before 2025-12-19 and CLMR after; a file carries one
    spelling, so both must be matched."""
    var = get_var(config, "pressure_level/cloud_mixing_ratio")
    assert var.internal_attrs.grib_element == "CLMR"
    assert var.internal_attrs.grib_element_alternatives == ("CLWMR",)


@AB_CONFIG_PARAMS
def test_window_comments_are_phrased_in_lead_time(
    config: NoaaGefsForecastABVirtualTemplateConfig,
) -> None:
    """An analysis names UTC clock hours; a forecast's window is set by lead time, and
    the same wall clock hour carries a different window in each init."""
    windowed = [v for v in config.data_vars if v.attrs.step_type != "instant"]
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

    assert get_var(config, "total_precipitation_surface").attrs.comment == (
        "Total accumulated in the last 6 hour period (lead times 6, 12, 18, ... hours) "
        "or 3 hour period (lead times 3, 9, 15, ... hours). Subtracting the value at an "
        "earlier lead time with the same window start gives the exact total between "
        "those two lead times."
    )


@AB_CONFIG_PARAMS
def test_every_lead_time_carries_the_window_its_comment_promises(
    config: NoaaGefsForecastABVirtualTemplateConfig,
) -> None:
    """The comment claims a 6 hour window at lead times 6, 12, 18, ... and a 3 hour one
    at 3, 9, 15, ...; the idx window string the region job matches on is what decides.
    Enumerated over every windowed variable and every lead, because a reset-frequency
    or day-form-window slip shows up only at particular leads (240, 384 and 840 hours
    are a whole number of days, where a running total would render "0-16 day acc fcst").
    """
    lead_hours = [int(t.total_seconds() // 3600) for t in config.lead_times()]
    windowed = [v for v in config.data_vars if v.attrs.step_type != "instant"]
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


@AB_CONFIG_PARAMS
def test_flag_variables_carry_only_their_codes(
    config: NoaaGefsForecastABVirtualTemplateConfig,
) -> None:
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
        var = get_var(config, path)
        assert var.attrs.flag_values == (0, 1), path
        assert var.attrs.flag_meanings == "no yes", path
        assert var.attrs.comment is None, path

    # GRIB2 code table 4.207, whose figures are not in severity order.
    severity = get_var(config, "pressure_level/icing_severity")
    assert severity.attrs.flag_values == (0, 1, 2, 3, 4, 5)
    assert severity.attrs.flag_meanings == "none light moderate severe trace heavy"
    assert severity.attrs.comment is None


@AB_CONFIG_PARAMS
def test_instant_variables_the_source_omits_at_lead_zero(
    config: NoaaGefsForecastABVirtualTemplateConfig,
) -> None:
    """Three convective cloud fields are instantaneous yet absent from the lead 0 file,
    so the default step_type rule would request a message that is not there."""
    omitted = {
        v.name for v in config.data_vars
        if v.attrs.step_type == "instant" and not v.has_hour_0_values()
    }  # fmt: skip
    assert omitted == {
        "pressure_convective_cloud_bottom",
        "pressure_convective_cloud_top",
        "convective_cloud_cover",
    }


@AB_CONFIG_PARAMS
def test_the_lead_axis_steps_up_where_the_a_and_b_files_coarsen(
    config: NoaaGefsForecastABVirtualTemplateConfig,
) -> None:
    """The a and b files publish 3 hourly through 240 hours and 6 hourly beyond it, and
    the attributes readers see name both spans rather than only the finer one."""
    lead_times = config.dimension_coordinates()["lead_time"]
    assert lead_times[0] == pd.Timedelta("0h")
    assert lead_times[-1] == config.forecast_length
    fine = lead_times[lead_times <= pd.Timedelta("240h")]
    coarse = lead_times[lead_times >= pd.Timedelta("240h")]
    assert set(np.diff(fine)) == {pd.Timedelta("3h").to_timedelta64()}
    assert set(np.diff(coarse)) == {pd.Timedelta("6h").to_timedelta64()}
    assert pd.Timedelta("243h") not in set(lead_times)

    hours = whole_hours(config.forecast_length)
    assert config.dataset_attributes.forecast_resolution == (
        f"Forecast step 0-240 hours: 3 hourly, 246-{hours} hours: 6 hourly"
    )


def test_forecast_resolution_names_one_span_for_an_even_lead_axis() -> None:
    """A uniformly spaced axis gets the plain wording, not a span it does not need."""
    config = NoaaGefsForecast10Day025DegreeVirtualTemplateConfig()
    assert config.dataset_attributes.forecast_resolution == "Forecast step 3 hourly"
