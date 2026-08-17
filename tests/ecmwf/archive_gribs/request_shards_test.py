import pandas as pd
import pytest

from reformatters.ecmwf.archive_gribs.request_shards import (
    CONTROL_ONLY_VARIABLES,
    DAILY_LEAD_TIMES,
    DAILY_MEAN_LEAD_TIMES,
    ENSEMBLE_SIZE,
    ISENTROPIC_VARIABLES,
    PRESSURE_LEVEL_VARIABLES,
    SINGLE_LEVEL_LEAD_TIMES,
    SIX_HOURLY_LEAD_TIMES,
    SIX_HOURLY_LEAD_TIMES_FROM_6H,
    EcdsSelection,
    initialization_selections,
    split_by_estimated_size,
)

INIT_TIME = pd.Timestamp("2026-08-10T00:00", tz="UTC")
ALL_VARIABLES = (
    *SINGLE_LEVEL_LEAD_TIMES,
    *PRESSURE_LEVEL_VARIABLES,
    *ISENTROPIC_VARIABLES,
)


BASE_SELECTION = EcdsSelection(
    level_type="pressure",
    forecast_type="perturbed_forecast",
    variables=("temperature",),
    level_values=("500_hpa", "850_hpa"),
    lead_time_labels=("0", "24"),
)


def selection(**overrides: str | tuple[str, ...]) -> EcdsSelection:
    return BASE_SELECTION.model_copy(update=dict(overrides))


def test_manifest_matches_the_ecmwf_origin_catalogue() -> None:
    assert len(SINGLE_LEVEL_LEAD_TIMES) == 38
    assert len(PRESSURE_LEVEL_VARIABLES) == 6
    assert CONTROL_ONLY_VARIABLES < set(SINGLE_LEVEL_LEAD_TIMES)
    assert len(SIX_HOURLY_LEAD_TIMES) == 185
    assert len(SIX_HOURLY_LEAD_TIMES_FROM_6H) == 184
    assert len(DAILY_LEAD_TIMES) == 47
    assert len(DAILY_MEAN_LEAD_TIMES) == 46
    assert SIX_HOURLY_LEAD_TIMES[-1] == DAILY_LEAD_TIMES[-1] == "1104"
    assert DAILY_MEAN_LEAD_TIMES[0] == "0_24"
    assert DAILY_MEAN_LEAD_TIMES[-1] == "1080_1104"
    assert len(PRESSURE_LEVEL_VARIABLES["specific_humidity"]) == 7
    assert len(PRESSURE_LEVEL_VARIABLES["temperature"]) == 10


def test_cost_matches_the_ecds_size_formula() -> None:
    # cost = 101 x n_variable x n_level x n_leadtime x n_date, for one date.
    assert selection().cost == ENSEMBLE_SIZE * 1 * 2 * 2
    assert selection(forecast_type="control_forecast").cost == ENSEMBLE_SIZE * 1 * 2 * 2
    assert selection(level_values=()).cost == ENSEMBLE_SIZE * 1 * 1 * 2


def test_message_count_excludes_members_not_requested() -> None:
    assert selection().message_count == 2 * 100 * 2
    assert selection(forecast_type="control_forecast").message_count == 2 * 1 * 2


def test_inputs_carry_the_initialization_date_and_selection() -> None:
    assert selection().inputs(INIT_TIME) == {
        "origin": "ecmwf",
        "forecast_type": "perturbed_forecast",
        "level_type": "pressure",
        "variable": ["temperature"],
        "year": ["2026"],
        "month": ["08"],
        "day": ["10"],
        "time": ["00:00"],
        "leadtime_hour": ["0", "24"],
        "level_value": ["500_hpa", "850_hpa"],
        "data_format": "grib",
    }
    assert "level_value" not in selection(level_values=()).inputs(INIT_TIME)


def test_file_name_changes_with_the_selection_it_holds() -> None:
    assert selection().file_name.startswith("pressure-perturbed_forecast-temperature-")
    assert selection().file_name.endswith(".grib2")
    assert selection().file_name != selection(level_values=("500_hpa",)).file_name
    assert selection().file_name != selection(lead_time_labels=("0",)).file_name
    assert selection().file_name != selection(variables=("temperature", "u")).file_name


def test_split_keeps_every_variable_exactly_once() -> None:
    wide = selection(variables=("a", "b", "c", "d", "e"))

    shards = split_by_estimated_size(wide, wide.estimated_bytes // 2)

    assert [variable for shard in shards for variable in shard.variables] == list(
        wide.variables
    )
    assert all(shard.estimated_bytes <= wide.estimated_bytes // 2 for shard in shards)


def test_split_never_produces_an_empty_request() -> None:
    single = selection(variables=("a", "b"))

    shards = split_by_estimated_size(single, maximum_shard_bytes=1)

    assert [shard.variables for shard in shards] == [("a",), ("b",)]


def test_selections_group_variables_that_share_levels_and_lead_times() -> None:
    selections = initialization_selections(
        ["10_m_u_component_of_wind", "total_precipitation", "2_m_temperature"]
    )

    by_forecast_type = {
        (shard.forecast_type, shard.lead_time_labels): shard.variables
        for shard in selections
    }
    assert by_forecast_type[("perturbed_forecast", SIX_HOURLY_LEAD_TIMES)] == (
        "10_m_u_component_of_wind",
        "total_precipitation",
    )
    assert by_forecast_type[("perturbed_forecast", DAILY_MEAN_LEAD_TIMES)] == (
        "2_m_temperature",
    )


def test_static_variables_are_requested_as_control_only() -> None:
    selections = initialization_selections(sorted(CONTROL_ONLY_VARIABLES))

    assert len(selections) == 1
    assert selections[0].forecast_type == "control_forecast"
    assert selections[0].variables == tuple(sorted(CONTROL_ONLY_VARIABLES))


def test_specific_humidity_is_requested_apart_from_the_ten_level_variables() -> None:
    selections = initialization_selections(["specific_humidity", "temperature"])

    levels_by_variable = {
        shard.variables: shard.level_values
        for shard in selections
        if shard.forecast_type == "perturbed_forecast"
    }
    assert (
        levels_by_variable[("specific_humidity",)]
        == PRESSURE_LEVEL_VARIABLES["specific_humidity"]
    )
    assert (
        levels_by_variable[("temperature",)] == PRESSURE_LEVEL_VARIABLES["temperature"]
    )


def test_whole_initialization_shards_are_uniquely_named_and_within_the_cost_cap() -> (
    None
):
    selections = initialization_selections(ALL_VARIABLES)

    staged_variables = {
        (shard.forecast_type, variable)
        for shard in selections
        for variable in shard.variables
    }
    assert len({shard.file_name for shard in selections}) == len(selections)
    assert all(shard.cost <= 1_000_000 for shard in selections)
    assert all(
        shard.estimated_bytes <= 4_000_000_000
        for shard in selections
        if len(shard.variables) > 1
    )
    assert {variable for _, variable in staged_variables} == set(ALL_VARIABLES)
    assert (
        not {
            variable
            for forecast_type, variable in staged_variables
            if forecast_type == "perturbed_forecast"
        }
        & CONTROL_ONLY_VARIABLES
    )


def test_unknown_variables_are_rejected() -> None:
    with pytest.raises(ValueError, match="not an ECMWF-origin S2S variable"):
        initialization_selections(["relative_humidity_2m"])
