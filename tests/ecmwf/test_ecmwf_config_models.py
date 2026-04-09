from typing import Literal

import pandas as pd
import pytest

from reformatters.common.config_models import DataVarAttrs, Encoding
from reformatters.common.zarr import BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE
from reformatters.ecmwf.ecmwf_config_models import (
    EcmwfDataVar,
    EcmwfInternalAttrs,
    has_hour_0_values,
    vars_available,
)

StepType = Literal["instant", "accum", "avg", "min", "max"]


def _make_data_var(
    step_type: StepType = "instant",
    hour_0_values_override: bool | None = None,
    open_data_date_available: pd.Timestamp | None = None,
) -> EcmwfDataVar:
    return EcmwfDataVar(
        name="test_var",
        encoding=Encoding(
            dtype="float32",
            fill_value=float("nan"),
            chunks=(1, 85, 51, 32, 32),
            shards=None,
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        ),
        attrs=DataVarAttrs(
            short_name="test",
            long_name="Test variable",
            units="1",
            step_type=step_type,
        ),
        internal_attrs=EcmwfInternalAttrs(
            grib_comment="test [unit]",
            grib_description='0[-] SFC="Ground or water surface"',
            grib_element="TEST",
            grib_index_param="test",
            keep_mantissa_bits=7,
            hour_0_values_override=hour_0_values_override,
            open_data_date_available=open_data_date_available,
        ),
    )


# --- vars_available ---


def test_vars_available_true_when_no_open_data_date_available() -> None:
    group = [_make_data_var(), _make_data_var()]
    assert vars_available(group, pd.Timestamp("2020-01-01")) is True


def test_vars_available_true_when_init_time_on_open_data_date_available() -> None:
    date = pd.Timestamp("2020-06-01")
    group = [_make_data_var(open_data_date_available=date)]
    assert vars_available(group, date) is True


def test_vars_available_true_when_init_time_after_open_data_date_available() -> None:
    group = [_make_data_var(open_data_date_available=pd.Timestamp("2020-06-01"))]
    assert vars_available(group, pd.Timestamp("2021-01-01")) is True


def test_vars_available_false_when_init_time_before_open_data_date_available() -> None:
    group = [_make_data_var(open_data_date_available=pd.Timestamp("2020-06-01"))]
    assert vars_available(group, pd.Timestamp("2020-01-01")) is False


def test_vars_available_raises_on_mixed_open_data_date_available() -> None:
    group = [
        _make_data_var(open_data_date_available=pd.Timestamp("2020-01-01")),
        _make_data_var(open_data_date_available=pd.Timestamp("2020-06-01")),
    ]
    with pytest.raises(ValueError, match="multiple"):
        vars_available(group, pd.Timestamp("2021-01-01"))


# --- has_hour_0_values ---


@pytest.mark.parametrize("step_type", ["instant", "avg", "accum"])
def test_has_hour_0_values_true_for_non_extremum_step_types(
    step_type: StepType,
) -> None:
    assert has_hour_0_values(_make_data_var(step_type)) is True


@pytest.mark.parametrize("step_type", ["max", "min"])
def test_has_hour_0_values_false_for_extremum_step_types(step_type: StepType) -> None:
    assert has_hour_0_values(_make_data_var(step_type)) is False


def test_has_hour_0_values_override_true_overrides_step_type() -> None:
    assert has_hour_0_values(_make_data_var("max", hour_0_values_override=True)) is True


def test_has_hour_0_values_override_false_overrides_step_type() -> None:
    assert (
        has_hour_0_values(_make_data_var("instant", hour_0_values_override=False))
        is False
    )
