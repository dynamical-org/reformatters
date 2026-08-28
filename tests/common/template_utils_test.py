import json
from pathlib import Path

import dask.array
import numpy as np
import pytest
import xarray as xr
from gribberish.zarr import GribberishCodec

from reformatters.common.config import Config, Env
from reformatters.common.config_models import (
    BaseInternalAttrs,
    DataVar,
    DataVarAttrs,
    Encoding,
)
from reformatters.common.template_utils import (
    _get_mode_from_path_store,
    assert_no_structural_drift_from_existing_store,
    assert_safe_overwrite,
    assign_var_metadata,
    empty_copy_with_reindex,
    make_empty_variable,
    sort_consolidated_metadata,
    store_written_coords,
)

# --- _get_mode_from_path_store tests ---


@pytest.mark.parametrize(
    ("path_str", "expected_mode"),
    [
        ("templates/latest.zarr", "w"),
        ("some/path/templates/latest.zarr", "w"),
        ("something-tmp.zarr", "w"),
        ("dev.zarr", "w-"),
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


def test_assign_var_metadata_includes_serializer_when_set() -> None:
    codec = GribberishCodec(var="TMP")

    class SerializerVar(DataVar[BaseInternalAttrs]):
        encoding: Encoding = Encoding(
            dtype="float64",
            fill_value=np.nan,
            chunks=(1,),
            shards=None,
            serializer=codec.to_dict(),
        )
        attrs: DataVarAttrs = DataVarAttrs(
            units="K",
            long_name="Temperature 2m",
            short_name="2t",
            step_type="instant",
        )
        internal_attrs: BaseInternalAttrs = BaseInternalAttrs(
            keep_mantissa_bits="no-rounding"
        )

    da = xr.DataArray([1.0, 2.0], name="temperature_2m")
    result = assign_var_metadata(da, SerializerVar(name="temperature_2m"))

    assert result.encoding["serializer"] == codec.to_dict()


def test_assign_var_metadata_omits_serializer_when_none() -> None:
    # Materialized vars don't declare a serializer; model_dump(exclude_none=True)
    # drops it so zarr falls back to its default BytesCodec.
    var_config = _TestDataVar(name="temperature_2m")
    da = xr.DataArray([1.0, 2.0], name="temperature_2m")

    result = assign_var_metadata(da, var_config)

    assert "serializer" not in result.encoding


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


# --- assert_no_structural_drift_from_existing_store tests ---


def _structured_var(
    *,
    dtype: object = "float32",
    chunks: tuple[int, ...] = (1, 3, 4),
    shards: tuple[int, ...] | None = (2, 3, 4),
    dims: tuple[str, ...] = ("time", "latitude", "longitude"),
    time_size: int = 4,
) -> xr.DataArray:
    sizes = {"time": time_size, "latitude": 3, "longitude": 4}
    da = xr.DataArray(
        np.zeros(tuple(sizes[d] for d in dims), dtype=np.float32), dims=dims
    )
    da.encoding = {"dtype": dtype, "chunks": chunks, "shards": shards}
    return da


def _structured_ds(**vars_: xr.DataArray) -> xr.DataTree:
    ds = xr.Dataset(dict(vars_) or {"var0": _structured_var()})
    return xr.DataTree.from_dict({"/": ds})


def test_structural_drift_passes_for_identical_structure() -> None:
    # Existing store reports a numpy dtype object; template reports a string.
    existing = _structured_ds(var0=_structured_var(dtype=np.dtype("float32")))
    template = _structured_ds(var0=_structured_var(dtype="float32"))
    assert_no_structural_drift_from_existing_store(template, existing, "time")


def test_structural_drift_allows_new_variables_in_template() -> None:
    existing = _structured_ds(var0=_structured_var())
    template = _structured_ds(var0=_structured_var(), var_new=_structured_var())
    assert_no_structural_drift_from_existing_store(template, existing, "time")


def test_structural_drift_allows_append_dim_chunk_and_shard_change() -> None:
    # Under the test env the append dim's chunk geometry is auto-shrunk to the (varying)
    # template length, so a differing append-axis chunk/shard must NOT trip the guard.
    # Non-append axes (here latitude/longitude) are unchanged.
    existing = _structured_ds(var0=_structured_var(chunks=(1, 3, 4), shards=(2, 3, 4)))
    template = _structured_ds(var0=_structured_var(chunks=(2, 3, 4), shards=(4, 3, 4)))
    assert_no_structural_drift_from_existing_store(template, existing, "time")


def test_structural_drift_checks_append_axis_outside_test_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # In dev/prod the append-dim chunk/shard IS part of the structural contract.
    monkeypatch.setattr(Config, "env", Env.prod)
    existing = _structured_ds(var0=_structured_var(chunks=(1, 3, 4)))
    template = _structured_ds(var0=_structured_var(chunks=(2, 3, 4)))  # append axis
    with pytest.raises(ValueError, match=r"var0\.chunks"):
        assert_no_structural_drift_from_existing_store(template, existing, "time")


def test_structural_drift_allows_new_variables_outside_test_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Adding a new variable must not trip the guard even when every axis is compared.
    monkeypatch.setattr(Config, "env", Env.prod)
    existing = _structured_ds(var0=_structured_var())
    template = _structured_ds(var0=_structured_var(), var_new=_structured_var())
    assert_no_structural_drift_from_existing_store(template, existing, "time")


def test_structural_drift_detects_removed_variable() -> None:
    existing = _structured_ds(var0=_structured_var(), var1=_structured_var())
    template = _structured_ds(var0=_structured_var())
    with pytest.raises(ValueError, match="var1: in existing store but missing"):
        assert_no_structural_drift_from_existing_store(template, existing, "time")


def test_structural_drift_detects_dtype_change() -> None:
    existing = _structured_ds(var0=_structured_var(dtype="float32"))
    template = _structured_ds(var0=_structured_var(dtype="float64"))
    with pytest.raises(ValueError, match=r"var0\.dtype"):
        assert_no_structural_drift_from_existing_store(template, existing, "time")


def test_structural_drift_detects_non_append_chunks_change() -> None:
    # Drift a spatial (non-append) axis: longitude chunk 4 -> 2.
    existing = _structured_ds(var0=_structured_var(chunks=(1, 3, 4)))
    template = _structured_ds(var0=_structured_var(chunks=(1, 3, 2)))
    with pytest.raises(ValueError, match=r"var0\.chunks"):
        assert_no_structural_drift_from_existing_store(template, existing, "time")


def test_structural_drift_detects_non_append_shards_change() -> None:
    # Drift a spatial (non-append) axis: longitude shard 4 -> 2.
    existing = _structured_ds(var0=_structured_var(shards=(2, 3, 4)))
    template = _structured_ds(var0=_structured_var(shards=(2, 3, 2)))
    with pytest.raises(ValueError, match=r"var0\.shards"):
        assert_no_structural_drift_from_existing_store(template, existing, "time")


def test_structural_drift_detects_dims_change() -> None:
    existing = _structured_ds(
        var0=_structured_var(dims=("time", "latitude", "longitude"))
    )
    template = _structured_ds(
        var0=_structured_var(dims=("time", "longitude", "latitude"))
    )
    with pytest.raises(ValueError, match=r"var0\.dims"):
        assert_no_structural_drift_from_existing_store(template, existing, "time")


# --- multi-group (root + vertical group) structural drift ---


def _pressure_var(
    *,
    chunks: tuple[int, ...] = (1, 3, 4, 2),
    pressure_levels: tuple[float, ...] = (1000.0, 850.0),
) -> xr.DataArray:
    # A pressure_level group var: (time, latitude, longitude, pressure_level).
    dims = ("time", "latitude", "longitude", "pressure_level")
    sizes = {"time": 4, "latitude": 3, "longitude": 4, "pressure_level": 2}
    da = xr.DataArray(
        np.zeros(tuple(sizes[d] for d in dims), dtype=np.float32),
        dims=dims,
        coords={"pressure_level": list(pressure_levels)},
    )
    da.encoding = {"dtype": "float32", "chunks": chunks, "shards": None}
    return da


def _two_node_ds(
    *,
    root: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
) -> xr.DataTree:
    root_ds = xr.Dataset({"var0": root if root is not None else _structured_var()})
    nodes: dict[str, xr.Dataset] = {"/": root_ds}
    if pressure is not None:
        nodes["/pressure_level"] = xr.Dataset({"temperature": pressure})
    return xr.DataTree.from_dict(nodes)


def test_structural_drift_detects_group_var_non_append_chunk_change() -> None:
    # Drift a non-append axis (pressure_level chunk 1 -> 2) on the GROUP var; the
    # mismatch must be reported against the group-qualified path.
    existing = _two_node_ds(pressure=_pressure_var(chunks=(1, 3, 4, 1)))
    template = _two_node_ds(pressure=_pressure_var(chunks=(1, 3, 4, 2)))
    with pytest.raises(ValueError, match=r"pressure_level/temperature\.chunks"):
        assert_no_structural_drift_from_existing_store(template, existing, "time")


def test_structural_drift_detects_group_var_missing_from_template() -> None:
    existing = _two_node_ds(pressure=_pressure_var())
    template = _two_node_ds()  # pressure_level group dropped
    with pytest.raises(
        ValueError, match=r"pressure_level/temperature: in existing store but missing"
    ):
        assert_no_structural_drift_from_existing_store(template, existing, "time")


def test_structural_drift_detects_group_coord_value_change() -> None:
    # Same vars/chunks, but the template reorders the pressure_level coord values.
    # A reordered vertical level would relabel existing chunks, so the guard fails.
    existing = _two_node_ds(pressure=_pressure_var(pressure_levels=(1000.0, 850.0)))
    template = _two_node_ds(pressure=_pressure_var(pressure_levels=(850.0, 1000.0)))
    with pytest.raises(ValueError, match=r"coord pressure_level"):
        assert_no_structural_drift_from_existing_store(template, existing, "time")


# --- assert_safe_overwrite tests ---


def test_assert_safe_overwrite_passes_for_identical_structure() -> None:
    existing = _structured_ds(var0=_structured_var())
    template = _structured_ds(var0=_structured_var())
    assert_safe_overwrite(
        template, existing, "time", allow_new_arrays=False, allow_expansion=False
    )


def test_assert_safe_overwrite_gates_new_arrays() -> None:
    existing = _structured_ds(var0=_structured_var())
    template = _structured_ds(var0=_structured_var(), var_new=_structured_var())
    with pytest.raises(ValueError, match="adds arrays not in the store"):
        assert_safe_overwrite(
            template, existing, "time", allow_new_arrays=False, allow_expansion=False
        )
    assert_safe_overwrite(
        template, existing, "time", allow_new_arrays=True, allow_expansion=False
    )


def test_assert_safe_overwrite_never_allows_trimming() -> None:
    existing = _structured_ds(var0=_structured_var(time_size=4))
    template = _structured_ds(var0=_structured_var(time_size=3))
    with pytest.raises(ValueError, match="trimming an existing store is never"):
        assert_safe_overwrite(
            template, existing, "time", allow_new_arrays=True, allow_expansion=True
        )


def test_assert_safe_overwrite_gates_expansion() -> None:
    existing = _structured_ds(var0=_structured_var(time_size=4))
    template = _structured_ds(var0=_structured_var(time_size=6))
    with pytest.raises(ValueError, match="extending requires"):
        assert_safe_overwrite(
            template, existing, "time", allow_new_arrays=True, allow_expansion=False
        )
    assert_safe_overwrite(
        template, existing, "time", allow_new_arrays=True, allow_expansion=True
    )


def test_assert_safe_overwrite_rejects_structural_drift() -> None:
    existing = _structured_ds(var0=_structured_var(dtype="float32"))
    template = _structured_ds(var0=_structured_var(dtype="float64"))
    with pytest.raises(ValueError, match=r"var0\.dtype"):
        assert_safe_overwrite(
            template, existing, "time", allow_new_arrays=True, allow_expansion=True
        )


# --- store_written_coords tests ---


def test_store_written_coords_finds_all_null_coords() -> None:
    times = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        coords={
            "time": ("time", times),
            "ingested_forecast_length": (
                "time",
                np.array(["NaT", "NaT"], dtype="datetime64[ns]"),
            ),
            "spatial_ref": ((), np.array(0)),
        }
    )
    tree = xr.DataTree.from_dict({"/": ds})
    assert store_written_coords(tree) == {"ingested_forecast_length"}
