"""Unit tests for ecmwf_grib_index.py JSONL parsing and byte-range extraction."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from reformatters.common.config_models import (
    DataVarAttrs,
    Encoding,
)
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar, EcmwfInternalAttrs
from reformatters.ecmwf.ecmwf_grib_index import (
    _parse_index_file,
    get_message_byte_ranges_from_index,
)

# ---------------------------------------------------------------------------
# Minimal JSONL index fixture
# ---------------------------------------------------------------------------

# Represents an ECMWF GRIB index file with:
#   - control forecast (type=cf, no "number") for 2t at surface level
#   - perturbed forecast member 1 for 2t at surface level
#   - perturbed forecast member 2 for gh at 500 hPa pressure level
_EXAMPLE_INDEX_JSONL = """\
{"domain": "g", "date": "20240201", "time": "0000", "expver": "0001", "class": "od", "type": "cf", "stream": "enfo", "step": "3", "levtype": "sfc", "param": "2t", "_offset": 0, "_length": 665525}
{"domain": "g", "date": "20240201", "time": "0000", "expver": "0001", "class": "od", "type": "pf", "stream": "enfo", "step": "3", "levtype": "sfc", "number": "1", "param": "2t", "_offset": 1554442, "_length": 664922}
{"domain": "g", "date": "20240201", "time": "0000", "expver": "0001", "class": "od", "type": "pf", "stream": "enfo", "step": "3", "levtype": "pl", "levelist": "500", "number": "2", "param": "gh", "_offset": 674936844, "_length": 393429}
"""


@pytest.fixture
def index_file(tmp_path: Path) -> Path:
    path = tmp_path / "test.index"
    path.write_text(_EXAMPLE_INDEX_JSONL)
    return path


# ---------------------------------------------------------------------------
# _parse_index_file tests
# ---------------------------------------------------------------------------


def test_parse_index_file_returns_dataframe(index_file: Path) -> None:
    df = _parse_index_file(index_file)
    assert isinstance(df, pd.DataFrame)


def test_parse_index_file_fills_control_member_number_with_0(index_file: Path) -> None:
    df = _parse_index_file(index_file)
    # Reset index to inspect values easily
    df_reset = df.reset_index()
    # Control member (type=cf) has no 'number' in the JSON; it should be filled with 0
    cf_rows = df_reset[df_reset["type"] == "cf"]
    assert (cf_rows["number"] == 0).all()


def test_parse_index_file_has_multiindex(index_file: Path) -> None:
    df = _parse_index_file(index_file)
    assert df.index.names == ["number", "param", "levtype", "levelist"]


def test_parse_index_file_contains_offset_and_length_columns(index_file: Path) -> None:
    df = _parse_index_file(index_file)
    assert "_offset" in df.columns
    assert "_length" in df.columns


def test_parse_index_file_correct_control_member_offset(index_file: Path) -> None:
    df = _parse_index_file(index_file)
    row = df.loc[(0, "2t", "sfc", slice(None)), ["_offset", "_length"]]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    assert int(row["_offset"]) == 0
    assert int(row["_length"]) == 665525


def test_parse_index_file_correct_perturbed_member_offset(index_file: Path) -> None:
    df = _parse_index_file(index_file)
    row = df.loc[(1, "2t", "sfc", slice(None)), ["_offset", "_length"]]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    assert int(row["_offset"]) == 1554442
    assert int(row["_length"]) == 664922


# ---------------------------------------------------------------------------
# get_message_byte_ranges_from_index tests
# ---------------------------------------------------------------------------


def _make_ecmwf_var(
    name: str,
    grib_index_param: str,
    grib_index_level_type: Literal["sfc", "pl"] = "sfc",
    grib_index_level_value: float = float("nan"),
) -> EcmwfDataVar:
    return EcmwfDataVar(
        name=name,
        encoding=Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=(1,),
            shards=None,
        ),
        attrs=DataVarAttrs(
            units="K",
            long_name="Test variable",
            short_name=grib_index_param,
            step_type="instant",
        ),
        internal_attrs=EcmwfInternalAttrs(
            keep_mantissa_bits=10,
            grib_index_param=grib_index_param,
            grib_index_level_type=grib_index_level_type,
            grib_index_level_value=grib_index_level_value,
            grib_element=grib_index_param,
            grib_comment="Test",
            grib_description="Test level",
        ),
    )


def test_get_byte_ranges_control_member_surface_var(index_file: Path) -> None:
    var = _make_ecmwf_var("temperature_2m", "2t", "sfc")
    starts, ends = get_message_byte_ranges_from_index(
        index_file, [var], ensemble_member=0
    )
    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0] == 0
    assert ends[0] == 0 + 665525


def test_get_byte_ranges_perturbed_member(index_file: Path) -> None:
    var = _make_ecmwf_var("temperature_2m", "2t", "sfc")
    starts, ends = get_message_byte_ranges_from_index(
        index_file, [var], ensemble_member=1
    )
    assert starts[0] == 1554442
    assert ends[0] == 1554442 + 664922


def test_get_byte_ranges_pressure_level_var(index_file: Path) -> None:
    var = _make_ecmwf_var(
        "geopotential_500hpa", "gh", "pl", grib_index_level_value=500.0
    )
    starts, ends = get_message_byte_ranges_from_index(
        index_file, [var], ensemble_member=2
    )
    assert starts[0] == 674936844
    assert ends[0] == 674936844 + 393429


def test_get_byte_ranges_returns_ints(index_file: Path) -> None:
    var = _make_ecmwf_var("temperature_2m", "2t", "sfc")
    starts, ends = get_message_byte_ranges_from_index(
        index_file, [var], ensemble_member=0
    )
    assert all(isinstance(s, int) for s in starts)
    assert all(isinstance(e, int) for e in ends)


def test_get_byte_ranges_raises_for_missing_var(index_file: Path) -> None:
    missing_var = _make_ecmwf_var("missing", "nonexistent_param", "sfc")
    with pytest.raises((KeyError, AssertionError)):
        get_message_byte_ranges_from_index(index_file, [missing_var], ensemble_member=0)


def test_get_byte_ranges_multiple_vars(index_file: Path) -> None:
    var1 = _make_ecmwf_var("temperature_2m", "2t", "sfc")
    # Use member 2 which has "gh" at 500 pl
    var2 = _make_ecmwf_var(
        "geopotential_500hpa", "gh", "pl", grib_index_level_value=500.0
    )

    # Test each independently since they require different ensemble members
    starts1, ends1 = get_message_byte_ranges_from_index(
        index_file, [var1], ensemble_member=0
    )
    starts2, ends2 = get_message_byte_ranges_from_index(
        index_file, [var2], ensemble_member=2
    )

    assert len(starts1) == 1
    assert len(starts2) == 1
    assert starts1[0] < ends1[0]
    assert starts2[0] < ends2[0]
