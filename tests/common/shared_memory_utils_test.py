from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest
import xarray as xr

from reformatters.common.shared_memory_utils import (
    create_data_array_and_template,
    make_shared_buffer,
)


@pytest.fixture
def simple_ds() -> xr.Dataset:
    return xr.Dataset(
        {
            "temperature_2m": xr.Variable(
                ("time", "lat"),
                np.zeros((4, 6), dtype=np.float32),
                encoding={"fill_value": np.nan},
            ),
            "pressure_surface": xr.Variable(
                ("time", "lat"),
                np.zeros((4, 6), dtype=np.float32),
                encoding={"fill_value": np.nan},
            ),
        }
    )


def test_make_shared_buffer_creates_and_unlinks(simple_ds: xr.Dataset) -> None:
    with make_shared_buffer(simple_ds) as shm:
        assert isinstance(shm, SharedMemory)
        name = shm.name
        # Buffer is accessible during context
        assert shm.size > 0

    # After context exit the buffer should be unlinked (accessing it by name should fail)
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name, create=False)


def test_make_shared_buffer_size_at_least_largest_variable(
    simple_ds: xr.Dataset,
) -> None:
    max_nbytes = max(v.nbytes for v in simple_ds.data_vars.values())
    with make_shared_buffer(simple_ds) as shm:
        assert shm.size >= max_nbytes


def test_create_data_array_and_template_shape(simple_ds: xr.Dataset) -> None:
    with make_shared_buffer(simple_ds) as shm:
        da, template = create_data_array_and_template(
            simple_ds, "temperature_2m", shm, fill_value=np.nan
        )
        assert da.shape == simple_ds["temperature_2m"].shape
        assert template.shape == simple_ds["temperature_2m"].shape


def test_create_data_array_float_initialized_with_nan(simple_ds: xr.Dataset) -> None:
    with make_shared_buffer(simple_ds) as shm:
        da, _ = create_data_array_and_template(
            simple_ds, "temperature_2m", shm, fill_value=0.0
        )
        # Float arrays initialized with NaN regardless of fill_value
        assert np.all(np.isnan(da.values))


def test_create_data_array_non_float_initialized_with_fill_value() -> None:
    ds = xr.Dataset(
        {
            "flag": xr.Variable(
                ("time",),
                np.zeros((4,), dtype=bool),
                encoding={"fill_value": False},
            )
        }
    )
    with make_shared_buffer(ds) as shm:
        da, _ = create_data_array_and_template(ds, "flag", shm, fill_value=False)
        assert np.all(da.values == False)  # noqa: E712


def test_create_data_array_template_drops_non_dim_coords() -> None:
    ds = xr.Dataset(
        {
            "temperature_2m": xr.Variable(
                ("time", "lat"),
                np.zeros((2, 3), dtype=np.float32),
                encoding={"fill_value": np.nan},
            )
        },
        coords={
            "time": xr.Variable(("time",), [0, 1]),
            "lat": xr.Variable(("lat",), [10, 20, 30]),
            # non-dim coordinate
            "station_name": xr.Variable(("time",), ["a", "b"]),
        },
    )
    with make_shared_buffer(ds) as shm:
        _, template = create_data_array_and_template(
            ds, "temperature_2m", shm, fill_value=np.nan
        )
        # Non-dimension coordinates are dropped from template
        assert "station_name" not in template.coords


def test_create_data_array_is_backed_by_shared_memory(simple_ds: xr.Dataset) -> None:
    with make_shared_buffer(simple_ds) as shm:
        da, _ = create_data_array_and_template(
            simple_ds, "temperature_2m", shm, fill_value=np.nan
        )
        # Writing to da should modify shared memory buffer
        da.values[:] = 42.0
        raw = np.ndarray(da.shape, dtype=da.dtype, buffer=shm.buf)
        assert np.all(raw == 42.0)
