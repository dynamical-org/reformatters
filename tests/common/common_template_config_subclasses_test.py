import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import template_utils
from reformatters.common.dynamical_dataset import DynamicalDataset


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
    template_ds = xr.open_zarr(template_path)

    for coord_name in template_ds.coords:
        coord_path = template_path / coord_name
        # Skip scalar coordinates (like spatial_ref) which have no chunks
        if not coord_path.is_dir():
            assert (
                coord_name == "spatial_ref"
            )  # spatial_ref doesn't write a chunk for older datasets where fill value = 0 and write empty chunks = false
            continue

        c_dir = coord_path / "c"

        if not c_dir.exists():
            # spatial_ref doesn't write a chunk for older datasets where fill value = 0 and write empty chunks = false
            assert coord_name == "spatial_ref"
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
    template_ds = xr.open_zarr(template_path)

    for coord_name in template_ds.coords:
        coord_zarr_json_path = template_path / coord_name / "zarr.json"

        assert coord_zarr_json_path.exists()

        with open(coord_zarr_json_path) as f:
            coord_metadata = json.load(f)

        codecs = coord_metadata["codecs"]
        codec_names = [codec["name"] for codec in codecs]

        assert "sharding_indexed" not in codec_names, (
            f"Coordinate '{coord_name}' should not use sharding, but found 'sharding_indexed' codec. "
            f"Codecs: {codec_names}"
        )
