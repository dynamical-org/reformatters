import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import xarray as xr

from reformatters.common import iterating
from reformatters.common.dynamical_dataset import DynamicalDataset

_LEGACY_MATERIALIZED_FILL_VALUE_EXCEPTIONS = {
    ("u-arizona-swann-analysis", "snow_depth"),
    ("u-arizona-swann-analysis", "snow_water_equivalent"),
    ("noaa-ndvi-cdr-analysis", "ndvi_raw"),
    ("noaa-ndvi-cdr-analysis", "ndvi_usable"),
}


@pytest.fixture(scope="module")
def stored_template(dataset: DynamicalDataset[Any, Any]) -> Iterator[xr.Dataset]:
    with xr.open_zarr(
        dataset.template_config.template_path(), decode_timedelta=True
    ) as ds:
        yield ds


def test_template_fill_values_are_correct(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    """
    Missing chunks decode to the declared fill value in every group.
    """
    template_config = dataset.template_config

    # Flattened so vertical-group vars (keyed by path) are visible; xr.open_zarr
    # would expose only the root group.
    raw_ds = iterating.flatten_groups(
        xr.open_datatree(
            template_config.template_path(),
            engine="zarr",
            chunks=None,
            consolidated=False,
            decode_cf=False,
        )
    )
    ds = xr.decode_cf(raw_ds, decode_timedelta=True)
    for var in template_config.data_vars:
        var_da = ds[var.path]
        raw_var_da = raw_ds[var.path]
        is_float = np.issubdtype(np.dtype(var.encoding.dtype), np.floating)
        is_legacy_exception = (
            dataset.dataset_id,
            var.path,
        ) in _LEGACY_MATERIALIZED_FILL_VALUE_EXCEPTIONS
        expected = var.encoding.fill_value if is_legacy_exception else np.nan
        if not is_float:
            expected = var.encoding.fill_value
        np.testing.assert_array_equal(
            var_da.isel(dict.fromkeys(var_da.dims, 0)).values, expected
        )
        if is_float:
            expected_cf_fill = (
                np.nan if is_legacy_exception else var.encoding.fill_value
            )
            np.testing.assert_equal(raw_var_da.attrs["_FillValue"], expected_cf_fill)
            assert "missing_value" not in raw_var_da.attrs


def test_coordinates_load_without_error(
    dataset: DynamicalDataset[Any, Any],
    stored_template: xr.Dataset,
) -> None:
    """
    Regression test: loading every coordinate of the checked-in template must
    not raise.

    A timedelta coordinate stored as int64 encodes NaT as the int64 sentinel
    (np.iinfo(int64).min); decoding it to a finer resolution (seconds -> us/ns)
    multiplies the sentinel and overflows on some xarray versions
    (OutOfBoundsTimedelta / "Overflow in int64 * timedelta64 multiplication").

    Opening the stored zarr exercises stored NaT sentinels (e.g.
    ingested_forecast_length), which get_template would otherwise re-derive from
    numpy and never decode. get_template loads all coords too, so we cover both.
    """
    template_config = dataset.template_config

    stored_ds = stored_template
    for coord_name in stored_ds.coords:
        stored_ds[coord_name].load()

    end_time = template_config.append_dim_start + template_config.append_dim_frequency
    template_config.get_template(end_time)


def test_timedelta_coordinates_stored_as_float(
    dataset: DynamicalDataset[Any, Any],
    stored_template: xr.Dataset,
) -> None:
    """
    Cross-template consistency: every timedelta coordinate must be stored as a
    float dtype so NaT serializes to NaN rather than the int64 sentinel that
    overflows on decode. See test_coordinates_load_without_error.
    """
    template_config = dataset.template_config
    template_path = template_config.template_path()

    stored_ds = stored_template
    for coord_name in stored_ds.coords:
        if stored_ds[coord_name].dtype.kind != "m":  # timedelta64
            continue
        with open(template_path / str(coord_name) / "zarr.json") as f:
            data_type = json.load(f)["data_type"]
        assert data_type == "float64", (
            f"Timedelta coordinate '{coord_name}' is stored as '{data_type}'. "
            "Store it as float64 so NaT round-trips without int64 overflow."
        )


def test_coordinates_have_single_chunk(
    dataset: DynamicalDataset[Any, Any],
    stored_template: xr.Dataset,
) -> None:
    """
    Ensure that every coordinate array has only a single chunk.
    Coordinates should have only file '0' in their c/ directory, no file '1', '2', etc.
    """
    template_config = dataset.template_config
    template_path = template_config.template_path()

    # Open the template to get the coordinates
    template_ds = stored_template

    for coord_name in template_ds.coords:
        c_path = template_path / str(coord_name) / "c"

        # We write empty chunks, so every coordinate has a chunk on disk. A
        # scalar coordinate (e.g. spatial_ref) stores its single chunk as the
        # file `c`; an N-d coordinate stores chunks under the directory `c/`.
        if c_path.is_file():
            continue

        chunk_file_names = [f.name for f in c_path.iterdir()]
        assert chunk_file_names == ["0"], (
            f"Coordinate '{coord_name}' should have only one chunk file '0', but found: {chunk_file_names}"
        )


def test_coordinates_not_sharded(
    dataset: DynamicalDataset[Any, Any],
    stored_template: xr.Dataset,
) -> None:
    """
    Ensure that all coordinate arrays are encoded without shards.
    Coordinates should use standard zarr chunks, not sharding_indexed codec.
    """
    template_config = dataset.template_config
    template_path = template_config.template_path()

    # Open the template to get the coordinates
    template_ds = stored_template

    for coord_name in template_ds.coords:
        coord_zarr_json_path = template_path / str(coord_name) / "zarr.json"

        assert coord_zarr_json_path.exists()

        with open(coord_zarr_json_path) as f:
            coord_metadata = json.load(f)

        codecs = coord_metadata["codecs"]
        codec_names = [codec["name"] for codec in codecs]

        assert "sharding_indexed" not in codec_names, (
            f"Coordinate '{coord_name}' should not use sharding, but found 'sharding_indexed' codec. "
            f"Codecs: {codec_names}"
        )
