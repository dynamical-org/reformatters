"""Multi-group (vertical dimension) template tests.

A synthetic dataset that mixes a root single-level variable with two vertical
groups that share a leaf variable name (`pressure_level/temperature` and
`model_level/temperature`) — the case a flat Dataset could not represent. Exercises
update_template, get_template, standalone group opening, and the structure validators.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import computed_field

from reformatters.common.config_models import (
    ROOT,
    BaseInternalAttrs,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVar,
    Encoding,
    Group,
)
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp

_LAT = np.array([10.0, 11.0, 12.0])
_LON = np.array([20.0, 21.0])
_PRESSURE_LEVELS = np.array([1000.0, 850.0, 500.0])
_MODEL_LEVELS = np.array([1, 2, 3, 4])


class _IA(BaseInternalAttrs):
    pass


class _DV(DataVar[_IA]):
    pass


def _coord(name: str, dtype: Literal["float64", "int64"] = "float64") -> Coordinate:
    # A single chunk larger than any test coordinate length.
    return Coordinate(
        name=name,
        encoding=Encoding(dtype=dtype, chunks=1000, shards=None, fill_value=0.0),
        attrs=CoordinateAttrs(units=None, statistics_approximate=None),
    )


def _data_var(name: str, group: Group, n_dims: int) -> _DV:
    return _DV(
        name=name,
        group=group,
        encoding=Encoding(
            dtype="float32",
            chunks=(1,) + (1,) * (n_dims - 1),
            shards=None,
            fill_value=np.nan,
            compressors=(),
        ),
        attrs=__import__(
            "reformatters.common.config_models", fromlist=["DataVarAttrs"]
        ).DataVarAttrs(long_name="X", short_name="x", units="K", step_type="instant"),
        internal_attrs=_IA(keep_mantissa_bits="no-rounding"),
    )


class MultiGroupConfig(TemplateConfig[_DV]):
    dims: dict[Group, tuple[Dim, ...]] = {
        ROOT: ("init_time", "latitude", "longitude"),
        "pressure_level": ("init_time", "latitude", "longitude", "pressure_level"),
        "model_level": ("init_time", "latitude", "longitude", "model_level"),
    }
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = pd.Timestamp("2020-01-01")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="multi-group-test",
            dataset_version="0.1.0",
            name="Multi group test",
            description="Synthetic multi-group dataset.",
            attribution="Test.",
            license="CC-BY-4.0",
            spatial_domain="Global",
            spatial_resolution="0.25 degrees (~20km)",
            time_domain="2020 to present",
            time_resolution="6 hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            "init_time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "latitude": _LAT,
            "longitude": _LON,
            "pressure_level": _PRESSURE_LEVELS,
            "model_level": _MODEL_LEVELS,
        }

    @computed_field  # type: ignore[prop-decorator]
    @property
    def coords(self) -> Sequence[Coordinate]:
        return [
            _coord("init_time", "int64"),
            _coord("latitude"),
            _coord("longitude"),
            _coord("pressure_level"),
            _coord("model_level", "int64"),
            Coordinate(
                name="spatial_ref",
                encoding=Encoding(dtype="int64", chunks=(), shards=None, fill_value=0),
                attrs=CoordinateAttrs(units=None, statistics_approximate=None),
            ),
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[_DV]:
        return [
            _data_var("temperature_2m", ROOT, 3),
            _data_var("temperature", "pressure_level", 4),
            _data_var("temperature", "model_level", 4),
        ]


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MultiGroupConfig:
    # Redirect the on-disk template to a temp dir so the test never writes the repo.
    template_path = tmp_path / "templates" / "latest.zarr"
    monkeypatch.setattr(MultiGroupConfig, "template_path", lambda self: template_path)
    return MultiGroupConfig()


def test_update_template_writes_three_node_tree(config: MultiGroupConfig) -> None:
    config.update_template()
    tree = xr.open_datatree(config.template_path(), consolidated=False)
    assert set(tree.groups) == {"/", "/pressure_level", "/model_level"}
    assert tree["pressure_level/temperature"].dims == (
        "init_time",
        "latitude",
        "longitude",
        "pressure_level",
    )
    assert tree["model_level/temperature"].dims == (
        "init_time",
        "latitude",
        "longitude",
        "model_level",
    )
    # The same leaf name lives in two groups without colliding.
    assert "temperature_2m" in tree["/"].data_vars


def test_groups_open_standalone_with_shared_coords(config: MultiGroupConfig) -> None:
    config.update_template()
    pressure = xr.open_zarr(
        config.template_path(), group="pressure_level", consolidated=False
    )
    # Shared coords are duplicated into the group; the other group's vertical coord is not.
    assert {"init_time", "latitude", "longitude", "pressure_level"} <= set(
        pressure.coords
    )
    assert "model_level" not in pressure.coords
    assert "temperature" in pressure.data_vars


def test_get_template_returns_reindexed_tree(config: MultiGroupConfig) -> None:
    config.update_template()
    tree = config.get_template(pd.Timestamp("2020-01-02"))
    assert isinstance(tree, xr.DataTree)
    # init_time reindexed from start to the requested end (every 6h over one day).
    assert tree["/"].sizes["init_time"] == 4
    assert tree["pressure_level"].sizes["init_time"] == 4
    assert tree["pressure_level/temperature"].encoding["fill_value"] is not None


class OnlyVerticalConfig(MultiGroupConfig):
    dims: dict[Group, tuple[Dim, ...]] = {
        ROOT: ("init_time", "latitude", "longitude"),
        "pressure_level": ("init_time", "latitude", "longitude", "pressure_level"),
    }

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            "init_time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "latitude": _LAT,
            "longitude": _LON,
            "pressure_level": _PRESSURE_LEVELS,
        }

    @computed_field  # type: ignore[prop-decorator]
    @property
    def coords(self) -> Sequence[Coordinate]:
        return [
            _coord("init_time", "int64"),
            _coord("latitude"),
            _coord("longitude"),
            _coord("pressure_level"),
            Coordinate(
                name="spatial_ref",
                encoding=Encoding(dtype="int64", chunks=(), shards=None, fill_value=0),
                attrs=CoordinateAttrs(units=None, statistics_approximate=None),
            ),
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[_DV]:
        # No root data var: every data var lives in the pressure_level group.
        return [_data_var("temperature", "pressure_level", 4)]


@pytest.fixture
def only_vertical_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> OnlyVerticalConfig:
    template_path = tmp_path / "templates" / "latest.zarr"
    monkeypatch.setattr(OnlyVerticalConfig, "template_path", lambda self: template_path)
    return OnlyVerticalConfig()


def test_only_vertical_vars_dataset_round_trips(
    only_vertical_config: OnlyVerticalConfig,
) -> None:
    only_vertical_config.update_template()
    tree = xr.open_datatree(only_vertical_config.template_path(), consolidated=False)

    # The root node carries the shared coords but has no data vars of its own.
    root = tree["/"]
    assert dict(root.data_vars) == {}
    assert {"init_time", "latitude", "longitude"} <= set(root.coords)

    # The pressure_level group still opens standalone with its var + shared coords.
    pressure = xr.open_zarr(
        only_vertical_config.template_path(),
        group="pressure_level",
        consolidated=False,
    )
    assert "temperature" in pressure.data_vars
    assert {"init_time", "latitude", "longitude", "pressure_level"} <= set(
        pressure.coords
    )


def test_validator_rejects_orphan_vertical_group() -> None:
    class OrphanGroup(MultiGroupConfig):
        @computed_field  # type: ignore[prop-decorator]
        @property
        def data_vars(self) -> Sequence[_DV]:
            # model_level is declared in dims but used by no var.
            return [
                _data_var("temperature_2m", ROOT, 3),
                _data_var("temperature", "pressure_level", 4),
            ]

    with pytest.raises(AssertionError, match="declared in dims but unused"):
        OrphanGroup()._assert_valid_structure()


def test_validator_group_must_add_its_own_dim() -> None:
    class BadAddedDim(MultiGroupConfig):
        dims: dict[Group, tuple[Dim, ...]] = {
            ROOT: ("init_time", "latitude", "longitude"),
            # pressure_level group adds "model_level" instead of "pressure_level"
            "pressure_level": ("init_time", "latitude", "longitude", "model_level"),
            "model_level": ("init_time", "latitude", "longitude", "model_level"),
        }

    with pytest.raises(AssertionError, match="must add exactly its own dim"):
        BadAddedDim()._assert_valid_structure()


def test_validator_root_var_name_cannot_equal_group_name() -> None:
    class RootNameCollision(MultiGroupConfig):
        dims: dict[Group, tuple[Dim, ...]] = {
            ROOT: ("init_time", "latitude", "longitude"),
            "pressure_level": ("init_time", "latitude", "longitude", "pressure_level"),
        }

        @computed_field  # type: ignore[prop-decorator]
        @property
        def data_vars(self) -> Sequence[_DV]:
            # A root var literally named "pressure_level" collides with the group node.
            return [
                _data_var("pressure_level", ROOT, 3),
                _data_var("temperature", "pressure_level", 4),
            ]

    with pytest.raises(AssertionError, match="collide with a group name"):
        RootNameCollision()._assert_valid_structure()


def test_validator_uniform_append_dim_chunk_size() -> None:
    class NonUniformAppendChunk(MultiGroupConfig):
        dims: dict[Group, tuple[Dim, ...]] = {
            ROOT: ("init_time", "latitude", "longitude"),
            "pressure_level": ("init_time", "latitude", "longitude", "pressure_level"),
        }

        @computed_field  # type: ignore[prop-decorator]
        @property
        def data_vars(self) -> Sequence[_DV]:
            root = _data_var("temperature_2m", ROOT, 3)
            # Pressure var with init_time chunk size 2 instead of 1.
            pressure = _DV(
                name="temperature",
                group="pressure_level",
                encoding=Encoding(
                    dtype="float32",
                    chunks=(2, 1, 1, 1),
                    shards=None,
                    fill_value=np.nan,
                    compressors=(),
                ),
                attrs=root.attrs,
                internal_attrs=_IA(keep_mantissa_bits="no-rounding"),
            )
            return [root, pressure]

    with pytest.raises(AssertionError, match="append-dim chunk size"):
        NonUniformAppendChunk()._assert_valid_structure()


def test_validator_encoding_chunk_length_must_match_group_dims() -> None:
    class WrongChunkLength(MultiGroupConfig):
        dims: dict[Group, tuple[Dim, ...]] = {
            ROOT: ("init_time", "latitude", "longitude"),
            "pressure_level": ("init_time", "latitude", "longitude", "pressure_level"),
        }

        @computed_field  # type: ignore[prop-decorator]
        @property
        def data_vars(self) -> Sequence[_DV]:
            # pressure_level group has 4 dims, but this var declares a 3-tuple chunks.
            short = _DV(
                name="temperature",
                group="pressure_level",
                encoding=Encoding(
                    dtype="float32",
                    chunks=(1, 1, 1),  # 3 entries, expected 4
                    shards=None,
                    fill_value=np.nan,
                    compressors=(),
                ),
                attrs=_data_var("temperature_2m", ROOT, 3).attrs,
                internal_attrs=_IA(keep_mantissa_bits="no-rounding"),
            )
            return [_data_var("temperature_2m", ROOT, 3), short]

    with pytest.raises(AssertionError, match="encoding chunks has 3 entries"):
        WrongChunkLength()._assert_valid_structure()


def test_validator_rejects_duplicate_group_name() -> None:
    class DuplicateVar(MultiGroupConfig):
        @computed_field  # type: ignore[prop-decorator]
        @property
        def data_vars(self) -> Sequence[_DV]:
            return [
                _data_var("temperature_2m", ROOT, 3),
                _data_var("temperature", "pressure_level", 4),
                _data_var("temperature", "model_level", 4),
                # Second pressure_level/temperature -> duplicate (group, name).
                _data_var("temperature", "pressure_level", 4),
            ]

    with pytest.raises(AssertionError, match=r"duplicate \(group, name\)"):
        DuplicateVar()._assert_valid_structure()


def test_validator_var_group_must_be_declared_in_dims() -> None:
    class UndeclaredGroup(MultiGroupConfig):
        dims: dict[Group, tuple[Dim, ...]] = {
            ROOT: ("init_time", "latitude", "longitude"),
            "pressure_level": ("init_time", "latitude", "longitude", "pressure_level"),
        }

        @computed_field  # type: ignore[prop-decorator]
        @property
        def data_vars(self) -> Sequence[_DV]:
            # model_level is never declared in dims above.
            return [
                _data_var("temperature_2m", ROOT, 3),
                _data_var("temperature", "pressure_level", 4),
                _data_var("temperature", "model_level", 4),
            ]

    with pytest.raises(AssertionError, match="not declared in dims"):
        UndeclaredGroup()._assert_valid_structure()
