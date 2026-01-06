import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Use a dummy DataVar that is a subtype of BaseInternalAttrs, as required by the type var
from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import template_utils
from reformatters.common.config_models import (
    BaseInternalAttrs,
    Coordinate,
    DatasetAttributes,
    DataVar,
)
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.template_config import (
    SPATIAL_REF_COORDS,
    TemplateConfig,
)
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp


class ExampleDataVar(DataVar[BaseInternalAttrs]):
    pass


class ExampleCoordinate(Coordinate):
    pass


class ExampleDatasetAttributes(DatasetAttributes):
    pass


class ExampleConfig(TemplateConfig[ExampleDataVar]):
    """A minimal concrete implementation to test the happy-path logic."""

    dims: tuple[Dim, ...] = ("time",)
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = pd.Timestamp("2000-01-01")
    append_dim_frequency: Timedelta = pd.Timedelta(days=1)

    @property
    def dataset_attributes(self) -> ExampleDatasetAttributes:
        return SimpleNamespace(dataset_id="simple_dataset")  # type: ignore[return-value]

    @property
    def coords(self) -> list[Coordinate]:
        # no extra coords beyond dims
        return []

    @property
    def data_vars(self) -> list[ExampleDataVar]:
        return []

    def dimension_coordinates(self) -> dict[str, pd.DatetimeIndex]:
        # not used in these tests
        return {"time": self.append_dim_coordinates(self.append_dim_start)}

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        # exercise the base-class fallback (which only adds spatial_ref)
        return super().derive_coordinates(ds)


class BadCoordsConfig(ExampleConfig):
    """Injects a coord whose name isn't in dims to trigger the NotImplementedError."""

    @property
    def coords(self) -> list[Coordinate]:
        # name "bad" is not in self.dims
        return [SimpleNamespace(name="bad")]  # type: ignore[list-item]


@pytest.fixture
def example_config() -> ExampleConfig:
    return ExampleConfig(
        dims=("time",),
        append_dim="time",
        append_dim_start=pd.Timestamp("2000-01-01"),
        append_dim_frequency=pd.Timedelta(days=1),
    )


def test_dataset_id_property(example_config: ExampleConfig) -> None:
    assert example_config.dataset_id == "simple_dataset"


def test_append_dim_coordinates_left_inclusive_right_exclusive(
    example_config: ExampleConfig,
) -> None:
    # up to but not including 2000-01-05
    end = pd.Timestamp("2000-01-05")
    got = example_config.append_dim_coordinates(end)
    expected = pd.date_range("2000-01-01", end, freq="1D", inclusive="left")
    pd.testing.assert_index_equal(got, expected)


@pytest.mark.parametrize(
    ("start_year", "expected_years"),
    [
        (2000, max(2025 - 2000 + 15, 10)),
        (2024, max(2025 - 2024 + 15, 10)),
        (2030, max(2025 - 2030 + 15, 10)),
    ],
)
def test_append_dim_coordinate_chunk_size_varies_with_start(
    start_year: int, expected_years: int
) -> None:
    class C(ExampleConfig):
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
    example_config: ExampleConfig,
) -> None:
    ds = xr.Dataset()
    coords = example_config.derive_coordinates(ds)
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
        match=r"Coordinates {'bad'} are defined.*derive_coordinates",
    ):
        bad.derive_coordinates(xr.Dataset())


@pytest.mark.parametrize(
    "dataset", DYNAMICAL_DATASETS, ids=[d.dataset_id for d in DYNAMICAL_DATASETS]
)
def test_update_template_matches_existing_template(
    dataset: DynamicalDataset[Any, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Ensure that `uv run main <dataset-id> update-template` has been run and
    all changes to the dataset's TemplateConfig are reflected in the on-disk Zarr template.
    """
    template_config = dataset.template_config

    # 1. Ensure that update_template() is a no-op

    with open(template_config.template_path() / "zarr.json") as f:
        existing_template = json.load(f)

    test_template_path = tmp_path / "latest.zarr"
    monkeypatch.setattr(
        type(template_config),
        "template_path",
        lambda _self: test_template_path,
    )

    template_config.update_template()

    with open(template_config.template_path() / "zarr.json") as f:
        updated_template = json.load(f)

    assert existing_template == updated_template


@pytest.mark.parametrize(
    "dataset", DYNAMICAL_DATASETS, ids=[d.dataset_id for d in DYNAMICAL_DATASETS]
)
def test_update_template_round_trips_correctly(
    dataset: DynamicalDataset[Any, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Ensure that the get_template() -> write_metadata() round trip produces exactly
    the same zarr.json as already exists on disk.
    """
    template_config = dataset.template_config

    with open(template_config.template_path() / "zarr.json") as f:
        existing_template = json.load(f)

    test_template_path = tmp_path / "latest.zarr"
    monkeypatch.setattr(
        type(template_config),
        "template_path",
        lambda _self: test_template_path,
    )

    template_config.update_template()

    # Compute an end_time to pass to get_template()
    dim_coords = template_config.dimension_coordinates()
    append_dim_coords = dim_coords[template_config.append_dim]
    end_time = append_dim_coords[-1] + pd.Timedelta(milliseconds=1)

    template_ds = template_config.get_template(end_time)

    test_write_metadata_path = tmp_path / "write_metadata_test.zarr"
    template_utils.write_metadata(
        template_ds,
        test_write_metadata_path,
    )
    with open(test_write_metadata_path / "zarr.json") as f:
        written_template = json.load(f)

    assert existing_template == written_template


@pytest.mark.parametrize(
    "dataset", DYNAMICAL_DATASETS, ids=[d.dataset_id for d in DYNAMICAL_DATASETS]
)
def test_update_template_fill_values_are_correct(
    dataset: DynamicalDataset[Any, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Ensure that the fill value we get in the case of a missing shard matches the template
    """
    template_config = dataset.template_config

    test_template_path = tmp_path / "latest.zarr"
    monkeypatch.setattr(
        type(template_config),
        "template_path",
        lambda _self: test_template_path,
    )

    template_config.update_template()

    # Compute an end_time to pass to get_template()
    dim_coords = template_config.dimension_coordinates()
    append_dim_coords = dim_coords[template_config.append_dim]
    end_time = append_dim_coords[-1] + pd.Timedelta(milliseconds=1)

    template_ds = template_config.get_template(end_time)

    test_write_metadata_path = tmp_path / "write_metadata_test.zarr"
    template_utils.write_metadata(
        template_ds,
        test_write_metadata_path,
    )

    # Ensure that the value we get when reading from an area that has not been written
    # to matches the fill value in template_config encodings
    # Coords are written by write_metadata() so we do expect them to be filled and don't test that here
    ds = xr.open_zarr(test_write_metadata_path, chunks=None)
    for var in template_config.data_vars:
        var_da = ds[var.name]
        np.testing.assert_array_equal(
            var_da.isel(dict.fromkeys(var_da.dims, 0)).values, var.encoding.fill_value
        )


@pytest.mark.parametrize(
    "dataset", DYNAMICAL_DATASETS, ids=[d.dataset_id for d in DYNAMICAL_DATASETS]
)
def test_coordinates_have_single_chunk(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    """
    Ensure that every coordinate array has only a single chunk.
    Coordinates should have only file '0' in their c/ directory, no file '1', '2', etc.
    """
    template_config = dataset.template_config
    template_path = template_config.template_path()

    # Open the template to get the coordinates
    template_ds = xr.open_zarr(template_path, decode_timedelta=True)

    for coord_name in template_ds.coords:
        coord_path = template_path / coord_name
        # Skip scalar coordinates (like spatial_ref) which have no chunks
        if not coord_path.is_dir():
            continue

        c_dir = coord_path / "c"
        # Some coordinates might not have a c/ directory (e.g., scalar arrays)
        if not c_dir.exists():
            continue

        # Check that only chunk file '0' exists
        chunk_files = list(c_dir.iterdir())
        chunk_file_names = [f.name for f in chunk_files]

        # Ensure exactly one chunk file exists and it's named '0'
        assert chunk_file_names == ["0"], (
            f"Coordinate '{coord_name}' should have only one chunk file '0', but found: {chunk_file_names}"
        )


@pytest.mark.parametrize(
    "dataset", DYNAMICAL_DATASETS, ids=[d.dataset_id for d in DYNAMICAL_DATASETS]
)
def test_coordinates_not_sharded(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    """
    Ensure that all coordinate arrays are encoded without shards.
    Coordinates should use standard zarr chunks, not sharding_indexed codec.
    """
    template_config = dataset.template_config
    template_path = template_config.template_path()

    # Open the template to get the coordinates
    template_ds = xr.open_zarr(template_path, decode_timedelta=True)

    for coord_name in template_ds.coords:
        coord_zarr_json_path = template_path / coord_name / "zarr.json"

        # Skip if zarr.json doesn't exist (shouldn't happen but be defensive)
        if not coord_zarr_json_path.exists():
            continue

        # Read the zarr.json for this coordinate
        with open(coord_zarr_json_path) as f:
            coord_metadata = json.load(f)

        # Check that the codecs list doesn't contain sharding_indexed
        codecs = coord_metadata.get("codecs", [])
        codec_names = [codec.get("name") for codec in codecs if isinstance(codec, dict)]

        assert "sharding_indexed" not in codec_names, (
            f"Coordinate '{coord_name}' should not use sharding, but found 'sharding_indexed' codec. "
            f"Codecs: {codec_names}"
        )
