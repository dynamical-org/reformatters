import pytest
import pandas as pd
import xarray as xr
from types import SimpleNamespace

from reformatters.common.template_config import (
    TemplateConfig,
    SPATIAL_REF_COORDS,
)


class SimpleConfig(TemplateConfig[None]):
    """A minimal concrete implementation to test the happy‐path logic."""
    dims = ("time",)
    append_dim = "time"
    append_dim_start = pd.Timestamp("2000-01-01")
    append_dim_frequency = pd.Timedelta(days=1)

    @property
    def dataset_attributes(self):
        return SimpleNamespace(dataset_id="simple_dataset")

    @property
    def coords(self):
        # no extra coords beyond dims
        return []

    @property
    def data_vars(self):
        return []

    def dimension_coordinates(self) -> dict[str, pd.DatetimeIndex]:
        # not used in these tests
        return {"time": self.append_dim_coordinates(self.append_dim_start)}

    def derive_coordinates(self, ds: xr.Dataset):
        # exercise the base‐class fallback (which only adds spatial_ref)
        return super().derive_coordinates(ds)


class BadCoordsConfig(SimpleConfig):
    """Injects a coord whose name isn't in dims to trigger the NotImplementedError."""
    @property
    def coords(self):
        # name "bad" is not in self.dims
        return [SimpleNamespace(name="bad")]


@pytest.fixture
def simple_cfg():
    return SimpleConfig()


def test_dataset_id_property(simple_cfg):
    assert simple_cfg.dataset_id == "simple_dataset"


def test_append_dim_coordinates_left_inclusive_right_exclusive(simple_cfg):
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
def test_append_dim_coordinate_chunk_size_varies_with_start(start_year, expected_years):
    class C(SimpleConfig):
        append_dim_start = pd.Timestamp(f"{start_year}-01-01")

    inst = C()
    # total days = 365 * expected_years, freq = 1 day
    expected = int(pd.Timedelta(days=365 * expected_years) / inst.append_dim_frequency)
    assert inst.append_dim_coordinate_chunk_size() == expected


def test_default_derive_coordinates_returns_spatial_ref(simple_cfg):
    ds = xr.Dataset()
    coords = simple_cfg.derive_coordinates(ds)
    # only the spatial_ref key should be present
    assert set(coords) == {"spatial_ref"}
    assert coords["spatial_ref"] == SPATIAL_REF_COORDS


def test_derive_coordinates_raises_if_coords_not_returned():
    bad = BadCoordsConfig()
    with pytest.raises(
        NotImplementedError,
        match=r"Coordinates \['bad'\] are defined.*derive_coordinates",
    ):
        bad.derive_coordinates(xr.Dataset())
