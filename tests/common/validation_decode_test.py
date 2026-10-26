from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation


@dataclass(frozen=True)
class _Var:
    path: str


@dataclass(frozen=True)
class _Coord:
    location: Mapping[str, Any]

    def out_loc(self) -> Mapping[str, Any]:
        return self.location


class _Job:
    append_dim = "init_time"

    def __init__(self, var_paths: Sequence[str], coords: Sequence[_Coord]) -> None:
        self.data_vars = tuple(_Var(path) for path in var_paths)
        self._coords = tuple(coords)

    def source_file_coords(self) -> Sequence[_Coord]:
        return self._coords

    def filter_already_present(
        self, _coords: Sequence[_Coord], _store: object
    ) -> list[_Coord]:
        return []


@dataclass(frozen=True)
class _LoadCall:
    name: str | None
    indexes: Mapping[str, list[Any]]
    coordinates: Mapping[str, Any]


def _context(
    ds: xr.Dataset, var_paths: Sequence[str], locations: Sequence[Mapping[str, Any]]
) -> validation.ValidationContext:
    store = Mock(spec=validation.IcechunkStore)
    assert isinstance(store, validation.IcechunkStore)
    job = _Job(var_paths, [_Coord(location) for location in locations])
    return validation.ValidationContext(
        store=cast("Any", store),
        ds=ds,
        append_dim=job.append_dim,
        data_vars=cast("Any", job.data_vars),
        region_job=cast("Any", job),
    )


def _spy_on_load(monkeypatch: pytest.MonkeyPatch) -> list[_LoadCall]:
    calls: list[_LoadCall] = []
    real_load = xr.DataArray.load

    def load(array: xr.DataArray) -> xr.DataArray:
        calls.append(
            _LoadCall(
                name=cast("str | None", array.name),
                indexes={str(dim): array.get_index(dim).tolist() for dim in array.dims},
                coordinates={
                    str(name): np.asarray(coord.values).tolist()
                    for name, coord in array.coords.items()
                },
            )
        )
        return real_load(array)

    def open_group(_store: object, *, mode: str) -> object:
        return object()

    monkeypatch.setattr(xr.DataArray, "load", load)
    monkeypatch.setattr(validation.zarr, "open_group", open_group)
    return calls


def test_default_presence_decodes_every_declared_variable_and_fails_fill_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = _spy_on_load(monkeypatch)
    init_time = pd.Timestamp("2025-01-01")
    ds = xr.Dataset(
        {
            "finite": (("init_time", "y", "x"), np.ones((1, 1, 1))),
            "missing_reference": (
                ("init_time", "y", "x"),
                np.full((1, 1, 1), np.nan),
            ),
        },
        coords={"init_time": [init_time], "y": [0], "x": [0]},
    )
    context = _context(ds, ["finite", "missing_reference"], [{"init_time": init_time}])

    result = validation.CheckVirtualDecodeHealth(max_workers=1).check(context)

    assert not result.passed
    assert result.checked_count == 2
    assert {call.name for call in loads} == {"finite", "missing_reference"}
    assert (
        "missing_reference: every sampled chunk decoded entirely NaN" in result.message
    )


def test_all_absent_levels_are_skipped_and_reported_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = _spy_on_load(monkeypatch)
    init_times = pd.date_range("2025-01-01", periods=2, freq="6h")
    levels = [1000, 850, 700, 500]
    shape = (len(init_times), len(levels), 1, 1)
    ds = xr.Dataset(
        {
            "pressure_level/finite": (
                ("init_time", "pressure_level", "y", "x"),
                np.ones(shape),
            ),
            "pressure_level/absent": (
                ("init_time", "pressure_level", "y", "x"),
                np.ones(shape),
            ),
        },
        coords={
            "init_time": init_times,
            "pressure_level": levels,
            "y": [0],
            "x": [0],
        },
    )
    context = _context(
        ds,
        ["pressure_level/finite", "pressure_level/absent"],
        [{"init_time": init_time} for init_time in init_times],
    )

    def reference_presence(
        var_path: str, out_loc: Mapping[str, Any]
    ) -> Mapping[Any, bool]:
        return {level: var_path.endswith("finite") for level in levels}

    result = validation.CheckVirtualDecodeHealth(
        positions="all",
        max_workers=1,
        reference_presence=reference_presence,
    ).check(context)

    assert result.passed, result.message
    assert result.checked_count == 2
    assert [call.name for call in loads] == [
        "pressure_level/finite",
        "pressure_level/finite",
    ]
    assert "1 variable(s) had no reference" in result.message


@pytest.mark.parametrize(
    ("sampled_levels", "expected"),
    [
        (3, [925, 700, 500]),
        (4, [925, 850, 700, 500]),
    ],
)
def test_only_present_levels_are_sampled_and_decoded(
    monkeypatch: pytest.MonkeyPatch,
    sampled_levels: int,
    expected: list[int],
) -> None:
    loads = _spy_on_load(monkeypatch)
    init_time = pd.Timestamp("2025-01-01")
    levels = [1000, 925, 850, 700, 500, 300]
    ds = xr.Dataset(
        {
            "pressure_level/temperature": (
                ("init_time", "pressure_level", "y", "x"),
                np.ones((1, len(levels), 1, 1)),
            )
        },
        coords={
            "init_time": [init_time],
            "pressure_level": levels,
            "y": [0],
            "x": [0],
        },
    )
    context = _context(ds, ["pressure_level/temperature"], [{"init_time": init_time}])
    presence = {level: level not in {1000, 300} for level in levels}

    result = validation.CheckVirtualDecodeHealth(
        sampled_levels=sampled_levels,
        max_workers=1,
        reference_presence=lambda var_path, out_loc: presence,
    ).check(context)

    assert result.passed, result.message
    assert result.checked_count == 1
    assert len(loads) == 1
    assert loads[0].indexes["pressure_level"] == expected


@pytest.mark.parametrize("present", [True, False])
def test_root_presence_uses_none_label(
    monkeypatch: pytest.MonkeyPatch, present: bool
) -> None:
    loads = _spy_on_load(monkeypatch)
    init_time = pd.Timestamp("2025-01-01")
    ds = xr.Dataset(
        {"temperature_2m": (("init_time", "y", "x"), np.ones((1, 1, 1)))},
        coords={"init_time": [init_time], "y": [0], "x": [0]},
    )
    context = _context(ds, ["temperature_2m"], [{"init_time": init_time}])

    result = validation.CheckVirtualDecodeHealth(
        max_workers=1,
        reference_presence=lambda var_path, out_loc: {None: present},
    ).check(context)

    assert result.passed is present, result.message
    assert result.checked_count == int(present)
    assert len(loads) == int(present)
    if not present:
        assert "No sampled variable had a reference" in result.message


@pytest.mark.parametrize("supplied", [False, True])
def test_missing_presence_labels_raise_descriptive_error(
    monkeypatch: pytest.MonkeyPatch, supplied: bool
) -> None:
    loads = _spy_on_load(monkeypatch)
    init_time = pd.Timestamp("2025-01-01")
    levels = [1000, 850, 700]
    ds = xr.Dataset(
        {
            "pressure_level/temperature": (
                ("init_time", "pressure_level", "y", "x"),
                np.ones((1, len(levels), 1, 1)),
            )
        },
        coords={
            "init_time": [init_time],
            "pressure_level": levels,
            "y": [0],
            "x": [0],
        },
    )
    context = _context(ds, ["pressure_level/temperature"], [{"init_time": init_time}])

    with pytest.raises(ValueError, match="reference presence labels") as exc_info:
        validation.CheckVirtualDecodeHealth(
            max_workers=1,
            reference_presence=lambda var_path, out_loc: {
                1000: supplied,
                850: supplied,
            },
        ).check(context)

    message = str(exc_info.value)
    assert "pressure_level/temperature" in message
    assert "missing=[700]" in message
    assert loads == []


def test_unexpected_true_presence_label_cannot_make_empty_selection_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = _spy_on_load(monkeypatch)
    init_time = pd.Timestamp("2025-01-01")
    levels = [1000, 850]
    ds = xr.Dataset(
        {
            "pressure_level/temperature": (
                ("init_time", "pressure_level", "y", "x"),
                np.ones((1, len(levels), 1, 1)),
            )
        },
        coords={
            "init_time": [init_time],
            "pressure_level": levels,
            "y": [0],
            "x": [0],
        },
    )
    context = _context(ds, ["pressure_level/temperature"], [{"init_time": init_time}])

    with pytest.raises(ValueError, match="reference presence labels") as exc_info:
        validation.CheckVirtualDecodeHealth(
            max_workers=1,
            reference_presence=lambda var_path, out_loc: {
                1000: False,
                850: False,
                700: True,
            },
        ).check(context)

    message = str(exc_info.value)
    assert "pressure_level/temperature" in message
    assert "unexpected=[700]" in message
    assert loads == []


def test_sample_levels_handles_mismatched_group_without_presence() -> None:
    da = xr.DataArray(
        np.ones((5, 1, 1)),
        dims=("height", "y", "x"),
        coords={"height": [0, 1, 2, 3, 4], "y": [0], "x": [0]},
    )

    sampled = validation.CheckVirtualDecodeHealth(sampled_levels=3)._sample_levels(
        da, "isobaric/temperature"
    )

    assert sampled.get_index("height").tolist() == [0, 2, 4]


def test_sample_levels_rejects_presence_for_mismatched_group_dimension() -> None:
    da = xr.DataArray(
        np.ones((3, 1, 1)),
        dims=("height", "y", "x"),
        coords={"height": [10, 20, 30], "y": [0], "x": [0]},
    )

    with pytest.raises(
        ValueError, match="reference presence requires group dimension"
    ) as exc_info:
        validation.CheckVirtualDecodeHealth()._sample_levels(
            da,
            "isobaric/temperature",
            {10: True, 20: True, 30: True},
        )

    message = str(exc_info.value)
    assert "isobaric/temperature" in message
    assert "isobaric" in message
    assert "height" in message


@pytest.mark.parametrize("present", [True, False])
def test_explicit_scalar_level_presence_is_validated_and_decoded(
    monkeypatch: pytest.MonkeyPatch,
    present: bool,
) -> None:
    loads = _spy_on_load(monkeypatch)
    init_time = pd.Timestamp("2025-01-01")
    levels = [1000, 850]
    ds = xr.Dataset(
        {
            "pressure_level/temperature": (
                ("init_time", "pressure_level", "y", "x"),
                np.ones((1, len(levels), 1, 1)),
            )
        },
        coords={
            "init_time": [init_time],
            "pressure_level": levels,
            "y": [0],
            "x": [0],
        },
    )
    context = _context(
        ds,
        ["pressure_level/temperature"],
        [{"init_time": init_time, "pressure_level": 850}],
    )

    result = validation.CheckVirtualDecodeHealth(
        max_workers=1,
        reference_presence=lambda var_path, out_loc: {850: present},
    ).check(context)

    assert result.passed is present, result.message
    assert result.checked_count == int(present)
    assert len(loads) == int(present)
    if present:
        assert loads[0].coordinates["pressure_level"] == 850
    else:
        assert "No sampled variable had a reference" in result.message


@pytest.mark.parametrize(
    ("var_path", "coords", "presence", "missing", "unexpected"),
    [
        ("temperature_2m", {}, {1000: False}, "[None]", "[1000]"),
        (
            "pressure_level/temperature",
            {"pressure_level": 850},
            {1000: True},
            "[850]",
            "[1000]",
        ),
        (
            "pressure_level/temperature",
            {"pressure_level": 850},
            {850: True, 1000: False},
            "[]",
            "[1000]",
        ),
    ],
)
def test_root_and_scalar_presence_labels_are_validated(
    var_path: str,
    coords: dict[str, int],
    presence: dict[int, bool],
    missing: str,
    unexpected: str,
) -> None:
    da = xr.DataArray(np.ones((1, 1)), dims=("y", "x"), coords=coords)
    with pytest.raises(ValueError, match="reference presence labels") as exc_info:
        validation.CheckVirtualDecodeHealth()._sample_levels(da, var_path, presence)
    message = str(exc_info.value)
    assert var_path in message
    assert f"missing={missing}" in message
    assert f"unexpected={unexpected}" in message
