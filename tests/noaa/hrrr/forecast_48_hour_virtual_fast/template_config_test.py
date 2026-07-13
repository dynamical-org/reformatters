from reformatters.common.config_models import ROOT
from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    NoaaHrrrForecast48HourTemplateConfig,
)
from reformatters.noaa.hrrr.forecast_48_hour_virtual.template_config import (
    NoaaHrrrForecast48HourVirtualTemplateConfig,
)
from reformatters.noaa.hrrr.forecast_48_hour_virtual_fast.template_config import (
    NoaaHrrrForecast48HourVirtualFastTemplateConfig,
)

FAST_CONFIG = NoaaHrrrForecast48HourVirtualFastTemplateConfig()
FULL_CONFIG = NoaaHrrrForecast48HourVirtualTemplateConfig()
MATERIALIZED_CONFIG = NoaaHrrrForecast48HourTemplateConfig()


def test_mirrors_materialized_variable_set() -> None:
    materialized_names = {v.name for v in MATERIALIZED_CONFIG.data_vars}
    fast_names = {v.name for v in FAST_CONFIG.data_vars}
    # Same-named variables carry over; the two deaccumulated rates are replaced by
    # the raw source quantities a read-time codec can serve.
    assert materialized_names - fast_names == {
        "precipitation_surface",
        "snowfall_surface",
    }
    assert fast_names - materialized_names == {
        "total_precipitation_surface",
        "precipitation_rate_surface",
        "total_snowfall_run_total_surface",
    }
    assert len(FAST_CONFIG.data_vars) == len(materialized_names) + 1


def test_variables_identical_to_full_virtual_dataset() -> None:
    full_by_name = {v.name: v for v in FULL_CONFIG.data_vars if v.group is ROOT}
    for var in FAST_CONFIG.data_vars:
        assert var == full_by_name[var.name]


def test_root_only_sfc_only_structure() -> None:
    assert set(FAST_CONFIG.dims) == {ROOT}
    assert all(v.group is ROOT for v in FAST_CONFIG.data_vars)
    assert {v.internal_attrs.hrrr_file_type for v in FAST_CONFIG.data_vars} == {"sfc"}
    coord_names = {c.name for c in FAST_CONFIG.coords}
    assert "pressure_level" not in coord_names
    assert "model_level" not in coord_names


def test_dataset_attributes_differ_only_in_identity() -> None:
    attrs = FAST_CONFIG.dataset_attributes
    assert attrs.dataset_id == "noaa-hrrr-forecast-48-hour-virtual-fast"
    assert attrs.name == "NOAA HRRR forecast, 48 hour, virtual, fast"
    identity_fields = {"dataset_id", "dataset_version", "name"}
    assert attrs.model_dump(exclude=identity_fields) == (
        FULL_CONFIG.dataset_attributes.model_dump(exclude=identity_fields)
    )
