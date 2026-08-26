import pandas as pd
import pytest

from scripts.remove_refs import _chunk_keys


class _Array:
    def __init__(self, shape: tuple[int, ...], chunks: tuple[int, ...]) -> None:
        self.shape = shape
        self.chunks = chunks


FORECAST = _Array((3, 49, 1059, 1799), (1, 1, 1059, 1799))
ANALYSIS = _Array((4, 1059, 1799), (1, 1059, 1799))
INITS = pd.to_datetime(pd.date_range("2018-07-13T12:00", periods=3, freq="6h"))
TIMES = pd.to_datetime(pd.date_range("2014-10-01", periods=4, freq="1h"))


def keys(*args: object) -> list[str]:
    return list(_chunk_keys(*args))  # ty: ignore[invalid-argument-type]


def test_lead_index_selects_that_lead_at_every_init() -> None:
    assert keys("precipitation_rate_surface", FORECAST, INITS, 0, None) == [
        "precipitation_rate_surface/c/0/0/0/0",
        "precipitation_rate_surface/c/1/0/0/0",
        "precipitation_rate_surface/c/2/0/0/0",
    ]


def test_before_selects_every_lead_of_the_earlier_positions() -> None:
    array = _Array((3, 2, 1059, 1799), (1, 1, 1059, 1799))
    assert keys("v", array, INITS, None, pd.Timestamp("2018-07-13T18:00")) == [
        "v/c/0/0/0/0",
        "v/c/0/1/0/0",
    ]


def test_an_array_without_a_lead_axis_keys_on_the_append_dim_alone() -> None:
    assert keys(
        "model_level/tke", ANALYSIS, TIMES, None, pd.Timestamp("2014-10-01T02:00")
    ) == [
        "model_level/tke/c/0/0/0",
        "model_level/tke/c/1/0/0",
    ]


def test_before_the_whole_record_selects_nothing() -> None:
    assert keys("v", ANALYSIS, TIMES, None, pd.Timestamp("2000-01-01")) == []


def test_lead_index_requires_a_lead_axis() -> None:
    with pytest.raises(AssertionError, match="no lead_time axis"):
        keys("v", ANALYSIS, TIMES, 0, None)


def test_lead_index_must_be_in_range() -> None:
    with pytest.raises(AssertionError):
        keys("v", FORECAST, INITS, 49, None)


def test_positions_must_match_the_array() -> None:
    with pytest.raises(AssertionError):
        keys("v", FORECAST, TIMES, 0, None)
