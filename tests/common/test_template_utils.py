import json
from pathlib import Path

import dask.array
import numpy as np
import pytest
import xarray as xr

from reformatters.common.config_models import (
    BaseInternalAttrs,
    DataVar,
    DataVarAttrs,
    Encoding,
)
from reformatters.common.template_utils import (
    _get_mode_from_path_store,
    assign_var_metadata,
    empty_copy_with_reindex,
    make_empty_variable,
    sort_consolidated_metadata,
)

# --- _get_mode_from_path_store tests ---


@pytest.mark.parametrize(
    ("path_str", "expected_mode"),
    [
        ("templates/latest.zarr", "w"),
        ("some/path/templates/latest.zarr", "w"),
        ("dev.zarr", "w"),
        ("something-tmp.zarr", "w"),
        ("prod-v1.zarr", "w-"),
        ("v1.0.zarr", "w-"),
        ("my-dataset/v1.5.zarr", "w-"),
    ],
)
def test_get_mode_from_path_store(path_str: str, expected_mode: str) -> None:
    assert _get_mode_from_path_store(Path(path_str)) == expected_mode


# --- sort_consolidated_metadata tests ---


def test_sort_consolidated_metadata_sorts_keys(tmp_path: Path) -> None:
    zarr_json = tmp_path / "zarr.json"
    content = {
        "consolidated_metadata": {"metadata": {"c_var": {}, "a_var": {}, "b_var": {}}}
    }
    zarr_json.write_text(json.dumps(content))

    sort_consolidated_metadata(zarr_json)

    result = json.loads(zarr_json.read_text())
    keys = list(result["consolidated_metadata"]["metadata"].keys())
    assert keys == sorted(keys)


def test_sort_consolidated_metadata_preserves_values(tmp_path: Path) -> None:
    zarr_json = tmp_path / "zarr.json"
    content = {
        "consolidated_metadata": {
            "metadata": {"z_var": {"dtype": "float32"}, "a_var": {"dtype": "int32"}}
        }
    }
    zarr_json.write_text(json.dumps(content))

    sort_consolidated_metadata(zarr_json)

    result = json.loads(zarr_json.read_text())
    assert result["consolidated_metadata"]["metadata"]["z_var"] == {"dtype": "float32"}
    assert result["consolidated_metadata"]["metadata"]["a_var"] == {"dtype": "int32"}


# --- make_empty_variable tests ---


def test_make_empty_variable_shape() -> None:
    coords = {"time": list(range(5)), "lat": list(range(3)), "lon": list(range(4))}
    var = make_empty_variable(("time", "lat", "lon"), coords, np.float32)
    assert var.shape == (5, 3, 4)


def test_make_empty_variable_dtype() -> None:
    coords = {"x": list(range(2)), "y": list(range(3))}
    var = make_empty_variable(("x", "y"), coords, np.float32)
    assert var.dtype == np.float32


def test_make_empty_variable_returns_xr_variable() -> None:
    coords = {"x": list(range(2))}
    var = make_empty_variable(("x",), coords, np.float32)
    assert isinstance(var, xr.Variable)


def test_make_empty_variable_is_dask_backed() -> None:
    coords = {"x": list(range(10))}
    var = make_empty_variable(("x",), coords, np.float32)
    assert isinstance(var.data, dask.array.Array)


# --- assign_var_metadata tests ---


class _TestDataVar(DataVar[BaseInternalAttrs]):
    encoding: Encoding = Encoding(
        dtype="float32",
        fill_value=np.nan,
        chunks=(1,),
        shards=None,
    )
    attrs: DataVarAttrs = DataVarAttrs(
        units="K",
        long_name="Temperature 2m",
        short_name="2t",
        step_type="instant",
    )
    internal_attrs: BaseInternalAttrs = BaseInternalAttrs(keep_mantissa_bits=10)


def test_assign_var_metadata_sets_encoding() -> None:
    var_config = _TestDataVar(name="temperature_2m")
    da = xr.DataArray([1.0, 2.0], name="temperature_2m")

    result = assign_var_metadata(da, var_config)

    assert result.encoding["dtype"] == "float32"
    assert np.isnan(result.encoding["fill_value"])


def test_assign_var_metadata_sets_attrs() -> None:
    var_config = _TestDataVar(name="temperature_2m")
    da = xr.DataArray([1.0, 2.0], name="temperature_2m")

    result = assign_var_metadata(da, var_config)

    assert result.attrs["long_name"] == "Temperature 2m"
    assert result.attrs["short_name"] == "2t"
    assert result.attrs["units"] == "K"


def test_assign_var_metadata_units_not_duplicated_when_in_encoding() -> None:
    # When encoding has units (for time vars), units should not be in attrs.
    # Valid TimedeltaUnits is "seconds since 1970-01-01 00:00:00"
    class TimeVar(DataVar[BaseInternalAttrs]):
        encoding: Encoding = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=(1,),
            shards=None,
            units="seconds since 1970-01-01 00:00:00",
        )
        attrs: DataVarAttrs = DataVarAttrs(
            units="seconds since 1970-01-01 00:00:00",
            long_name="Time",
            short_name="time",
            step_type="instant",
        )
        internal_attrs: BaseInternalAttrs = BaseInternalAttrs(keep_mantissa_bits=10)

    var_config = TimeVar(name="time")
    da = xr.DataArray([0, 1, 2], name="time")

    result = assign_var_metadata(da, var_config)

    # units in encoding but not in attrs (to avoid duplication)
    assert "units" not in result.attrs
    assert "units" in result.encoding


# --- empty_copy_with_reindex tests ---


def test_empty_copy_with_reindex_new_dim_size() -> None:
    original = xr.Dataset(
        {
            "var": xr.Variable(
                ("time", "lat"),
                np.zeros((3, 2), dtype=np.float32),
                encoding={"fill_value": np.nan},
            )
        },
        coords={
            "time": xr.Variable(("time",), [0, 1, 2], encoding={"fill_value": -1}),
            "lat": xr.Variable(("lat",), [10, 20], encoding={"fill_value": -1}),
        },
        attrs={"description": "test dataset"},
    )

    new_times = [0, 1, 2, 3, 4]
    result = empty_copy_with_reindex(original, "time", new_times)

    assert result.sizes["time"] == 5
    assert result.sizes["lat"] == 2


def test_empty_copy_with_reindex_preserves_attrs() -> None:
    original = xr.Dataset(
        {"var": xr.Variable(("time",), [1.0], encoding={"fill_value": np.nan})},
        coords={"time": xr.Variable(("time",), [0], encoding={"fill_value": -1})},
        attrs={"description": "test"},
    )
    result = empty_copy_with_reindex(original, "time", [0, 1])
    assert result.attrs["description"] == "test"


def test_empty_copy_with_reindex_preserves_var_encoding() -> None:
    original_encoding = {"fill_value": np.nan, "dtype": "float32"}
    original = xr.Dataset(
        {
            "var": xr.Variable(
                ("time",),
                [1.0],
                encoding=original_encoding,
            )
        },
        coords={"time": xr.Variable(("time",), [0], encoding={"fill_value": -1})},
    )
    result = empty_copy_with_reindex(original, "time", [0, 1])
    assert np.isnan(result["var"].encoding["fill_value"])


def test_empty_copy_with_reindex_with_derive_fn() -> None:
    original = xr.Dataset(
        {"var": xr.Variable(("time",), [1.0], encoding={"fill_value": np.nan})},
        coords={"time": xr.Variable(("time",), [0], encoding={"fill_value": -1})},
    )

    def derive(ds: xr.Dataset) -> dict:
        return {"derived": (("time",), np.array([99] * ds.sizes["time"]))}

    result = empty_copy_with_reindex(
        original, "time", [0, 1], derive_coordinates_fn=derive
    )
    assert "derived" in result.coords
    assert len(result.coords["derived"]) == 2
