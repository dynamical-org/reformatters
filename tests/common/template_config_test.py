from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Use a dummy DataVar that is a subtype of BaseInternalAttrs, as required by the type var
from reformatters.common.config_models import (
    BaseInternalAttrs,
    Coordinate,
    DatasetAttributes,
    DataVar,
)
from reformatters.common.template_config import (
    SPATIAL_REF_COORDS,
    AppendDim,
    Dim,
    TemplateConfig,
)
from reformatters.common.types import Timedelta, Timestamp


class DummyDataVar(DataVar[BaseInternalAttrs]):
    pass


class DummyCoordinate(Coordinate):
    pass


class DummyDatasetAttributes(DatasetAttributes):
    pass


class SimpleConfig(TemplateConfig[DummyDataVar]):
    """A minimal concrete implementation to test the happy‐path logic."""

    dims: tuple[Dim, ...] = ("time",)
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = pd.Timestamp("2000-01-01")
    append_dim_frequency: Timedelta = pd.Timedelta(days=1)

    @property
    def dataset_attributes(self) -> DummyDatasetAttributes:
        return SimpleNamespace(dataset_id="simple_dataset")  # type: ignore

    @property
    def coords(self) -> list[Coordinate]:
        # no extra coords beyond dims
        return []

    @property
    def data_vars(self) -> list[DummyDataVar]:
        return []

    def dimension_coordinates(self) -> dict[str, pd.DatetimeIndex]:
        # not used in these tests
        return {"time": self.append_dim_coordinates(self.append_dim_start)}

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        # exercise the base‐class fallback (which only adds spatial_ref)
        return super().derive_coordinates(ds)


class BadCoordsConfig(SimpleConfig):
    """Injects a coord whose name isn't in dims to trigger the NotImplementedError."""

    @property
    def coords(self) -> list[Coordinate]:
        # name "bad" is not in self.dims
        return [SimpleNamespace(name="bad")]  # type: ignore


@pytest.fixture
def simple_cfg() -> SimpleConfig:
    return SimpleConfig(
        dims=("time",),
        append_dim="time",
        append_dim_start=pd.Timestamp("2000-01-01"),
        append_dim_frequency=pd.Timedelta(days=1),
    )


def test_dataset_id_property(simple_cfg: SimpleConfig) -> None:
    assert simple_cfg.dataset_id == "simple_dataset"


def test_append_dim_coordinates_left_inclusive_right_exclusive(
    simple_cfg: SimpleConfig,
) -> None:
    # up to but not including 2000-01-05
    end = pd.Timestamp("2000-01-05")
    got = simple_cfg.append_dim_coordinates(end)
    expected = pd.date_range("2000-01-01", end, freq="1D", inclusive="left")
    pd.testing.assert_index_equal(got, expected)


@pytest.mark.parametrize(
    "start_year, expected_years",
    [
        (2000, max(2025 - 2000 + 15, 10)),
        (2024, max(2025 - 2024 + 15, 10)),
        (2030, max(2025 - 2030 + 15, 10)),
    ],
)
def test_append_dim_coordinate_chunk_size_varies_with_start(
    start_year: int, expected_years: int
) -> None:
    class C(SimpleConfig):
        append_dim_start: Timestamp = pd.Timestamp(f"{start_year}-01-01")

    inst = C(
        dims=("time",),
        append_dim="time",
        append_dim_start=pd.Timestamp(f"{start_year}-01-01"),
        append_dim_frequency=pd.Timedelta(days=1),
    )
    # total days = 365 * expected_years, freq = 1 day
    expected = int(pd.Timedelta(days=365 * expected_years) / inst.append_dim_frequency)
    assert inst.append_dim_coordinate_chunk_size() == expected


def test_default_derive_coordinates_returns_spatial_ref(
    simple_cfg: SimpleConfig,
) -> None:
    ds = xr.Dataset()
    coords = simple_cfg.derive_coordinates(ds)
    # only the spatial_ref key should be present
    assert set(coords) == {"spatial_ref"}
    assert coords["spatial_ref"] == SPATIAL_REF_COORDS


def test_derive_coordinates_raises_if_coords_not_returned() -> None:
    bad = BadCoordsConfig(
        dims=("time",),
        append_dim="time",
        append_dim_start=pd.Timestamp("2000-01-01"),
        append_dim_frequency=pd.Timedelta(days=1),
    )
    with pytest.raises(
        NotImplementedError,
        match=r"Coordinates \['bad'\] are defined.*derive_coordinates",
    ):
        bad.derive_coordinates(xr.Dataset())
