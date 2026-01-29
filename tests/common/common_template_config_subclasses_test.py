import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import template_utils
from reformatters.common.dynamical_dataset import DynamicalDataset


@dataclass
class TemplateSetup:
    """Shared setup for template tests - computed once per dataset."""

    dataset: DynamicalDataset[Any, Any]
    existing_template: dict[str, Any]
    updated_template_path: Path
    roundtrip_template_path: Path


_template_setup_cache: dict[str, TemplateSetup] = {}


@pytest.fixture
def template_setup(
    dataset: DynamicalDataset[Any, Any], tmp_path_factory: pytest.TempPathFactory
) -> TemplateSetup:
    """
    Fixture that performs expensive template setup once per dataset.
    Caches the result so multiple tests can share it.
    """
    dataset_id = dataset.dataset_id
    if dataset_id in _template_setup_cache:
        return _template_setup_cache[dataset_id]

    template_config = dataset.template_config

    with open(template_config.template_path() / "zarr.json") as f:
        existing_template = json.load(f)

    tmp_path = tmp_path_factory.mktemp(dataset_id)
    test_template_path = tmp_path / "latest.zarr"

    # Monkeypatch the template_path method for this template_config's class
    original_template_path = type(template_config).template_path
    type(template_config).template_path = lambda _self: test_template_path  # type: ignore[assignment]

    try:
        template_config.update_template()

        dim_coords = template_config.dimension_coordinates()
        append_dim_coords = dim_coords[template_config.append_dim]
        end_time = append_dim_coords[-1] + pd.Timedelta(milliseconds=1)

        template_ds = template_config.get_template(end_time)

        test_write_metadata_path = tmp_path / "write_metadata_test.zarr"
        template_utils.write_metadata(template_ds, test_write_metadata_path)
    finally:
        type(template_config).template_path = original_template_path  # type: ignore[method-assign]

    setup = TemplateSetup(
        dataset=dataset,
        existing_template=existing_template,
        updated_template_path=test_template_path,
        roundtrip_template_path=test_write_metadata_path,
    )
    _template_setup_cache[dataset_id] = setup
    return setup


@pytest.fixture
def dataset(request: pytest.FixtureRequest) -> DynamicalDataset[Any, Any]:
    """Fixture that returns the dataset from the parametrize marker."""
    return request.param  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    "dataset", DYNAMICAL_DATASETS, ids=[d.dataset_id for d in DYNAMICAL_DATASETS]
)
def test_update_template_matches_existing_template(
    template_setup: TemplateSetup,
) -> None:
    """
    Ensure that `uv run main <dataset-id> update-template` has been run and
    all changes to the dataset's TemplateConfig are reflected in the on-disk Zarr template.
    """
    with open(template_setup.updated_template_path / "zarr.json") as f:
        updated_template = json.load(f)

    assert template_setup.existing_template == updated_template


@pytest.mark.parametrize(
    "dataset", DYNAMICAL_DATASETS, ids=[d.dataset_id for d in DYNAMICAL_DATASETS]
)
def test_update_template_round_trips_correctly(
    template_setup: TemplateSetup,
) -> None:
    """
    Ensure that the get_template() -> write_metadata() round trip produces exactly
    the same zarr.json as already exists on disk.
    """
    with open(template_setup.roundtrip_template_path / "zarr.json") as f:
        roundtrip_template = json.load(f)

    assert template_setup.existing_template == roundtrip_template


@pytest.mark.parametrize(
    "dataset", DYNAMICAL_DATASETS, ids=[d.dataset_id for d in DYNAMICAL_DATASETS]
)
def test_update_template_fill_values_are_correct(
    template_setup: TemplateSetup,
) -> None:
    """
    Ensure that the fill value we get in the case of a missing shard matches the template
    """
    template_config = template_setup.dataset.template_config

    ds = xr.open_zarr(template_setup.roundtrip_template_path, chunks=None)
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
