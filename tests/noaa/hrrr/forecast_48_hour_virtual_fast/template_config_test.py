from reformatters.common.config_models import ROOT
from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    NoaaHrrrForecast48HourTemplateConfig,
)
from reformatters.noaa.hrrr.forecast_48_hour_virtual_fast.template_config import (
    _DEACCUMULATED_VAR_SUBSTITUTES,
    NoaaHrrrForecast48HourVirtualFastTemplateConfig,
)

CONFIG = NoaaHrrrForecast48HourVirtualFastTemplateConfig()
MATERIALIZED = NoaaHrrrForecast48HourTemplateConfig()


def test_variable_set_matches_materialized_with_substitutions() -> None:
    """The point of this dataset: its variables are the materialized product's, with
    only the deaccumulated ones replaced by the raw quantities a codec can produce."""
    expected = {
        name
        for var in MATERIALIZED.data_vars
        for name in _DEACCUMULATED_VAR_SUBSTITUTES.get(var.name, (var.name,))
    }
    assert {var.name for var in CONFIG.data_vars} == expected
    assert len(CONFIG.data_vars) == 28


def test_substitutes_cover_exactly_the_deaccumulated_variables() -> None:
    deaccumulated = {
        var.name
        for var in MATERIALIZED.data_vars
        if var.internal_attrs.deaccumulate_to_rate
    }
    assert deaccumulated == _DEACCUMULATED_VAR_SUBSTITUTES.keys()
    assert deaccumulated == {"precipitation_surface", "snowfall_surface"}


def test_every_variable_is_root_and_sfc() -> None:
    """All single-level, so an update polls only wrfsfc files."""
    assert {var.group for var in CONFIG.data_vars} == {ROOT}
    assert {var.internal_attrs.hrrr_file_type for var in CONFIG.data_vars} == {"sfc"}


def test_no_vertical_groups() -> None:
    assert set(CONFIG.dims) == {ROOT}
    assert CONFIG.dims[ROOT] == ("init_time", "lead_time", "y", "x")


def test_one_chunk_per_message() -> None:
    var = next(v for v in CONFIG.data_vars if v.name == "temperature_2m")
    assert var.encoding.chunks == (1, 1, 1059, 1799)
    assert var.encoding.shards is None
    assert var.encoding.compressors == ()
    assert var.encoding.serializer is not None
    assert var.internal_attrs.keep_mantissa_bits == "no-rounding"


def test_shared_variables_keep_materialized_metadata() -> None:
    """A name shared with the materialized product must mean the same thing."""
    materialized_by_name = {var.name: var for var in MATERIALIZED.data_vars}
    shared = [var for var in CONFIG.data_vars if var.name in materialized_by_name]
    assert len(shared) == 25  # 28 minus the 3 substitutes
    for var in shared:
        assert var.attrs == materialized_by_name[var.name].attrs, var.name


def test_dataset_identity() -> None:
    attrs = CONFIG.dataset_attributes
    assert attrs.dataset_id == "noaa-hrrr-forecast-48-hour-virtual-fast"
    assert attrs.name == "NOAA HRRR forecast, 48 hour, virtual, fast"
