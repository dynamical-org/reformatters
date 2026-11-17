from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import eccodes
import eccodeslib
import numpy as np
import pytest

from reformatters.common.grib_eccodes import read_unstructured_grib
from reformatters.common.types import Array1D

GRID_UUID = "0123456789abcdef0123456789abcdef"
POINT_COUNT = 65_536


def test_loaded_library_is_bundled_with_pthread_support() -> None:
    library_path = Path(eccodes.codes_get_library_path()).resolve()
    bundle_path = Path(eccodeslib.__file__).resolve().parent

    assert library_path.is_relative_to(bundle_path)
    assert (
        "ECCODES_THREADS"
        in eccodes.codes_get_features(eccodes.CODES_FEATURES_ENABLED).split()
    )


def _write_grib(
    path: Path,
    values: Array1D[np.float64],
    *,
    grid_template: int = 101,
    missing_value: float | None = None,
) -> None:
    handle = eccodes.codes_grib_new_from_samples("GRIB2")
    try:
        eccodes.codes_set(handle, "gridDefinitionTemplateNumber", grid_template)
        if grid_template == 101:
            eccodes.codes_set(handle, "numberOfGridUsed", 1)
            eccodes.codes_set(handle, "numberOfGridInReference", 1)
            eccodes.codes_set(handle, "uuidOfHGrid", GRID_UUID)
        eccodes.codes_set(handle, "numberOfDataPoints", values.size)
        eccodes.codes_set(handle, "packingType", "grid_ccsds")
        eccodes.codes_set(handle, "bitsPerValue", 24)
        if missing_value is not None:
            eccodes.codes_set(handle, "missingValue", missing_value)
            eccodes.codes_set(handle, "bitmapPresent", 1)
        eccodes.codes_set_values(handle, values)
        with path.open("wb") as file:
            eccodes.codes_write(handle, file)
    finally:
        eccodes.codes_release(handle)


def test_ccsds_unstructured_values_survive_float32_decode(tmp_path: Path) -> None:
    values = np.linspace(-30, 45, POINT_COUNT)
    path = tmp_path / "field.grib2"
    _write_grib(path, values)

    result = read_unstructured_grib(path, GRID_UUID, POINT_COUNT)

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, values, rtol=0, atol=1e-5)


def test_rejects_changed_grid_uuid(tmp_path: Path) -> None:
    path = tmp_path / "field.grib2"
    _write_grib(path, np.arange(32, dtype=np.float64))

    with pytest.raises(AssertionError):
        read_unstructured_grib(path, "ffffffffffffffffffffffffffffffff", 32)


def test_rejects_changed_grid_point_count(tmp_path: Path) -> None:
    path = tmp_path / "field.grib2"
    _write_grib(path, np.arange(32, dtype=np.float64))

    with pytest.raises(AssertionError):
        read_unstructured_grib(path, GRID_UUID, 33)


def test_rejects_structured_grid(tmp_path: Path) -> None:
    path = tmp_path / "field.grib2"
    _write_grib(path, np.arange(32, dtype=np.float64), grid_template=0)

    with pytest.raises(AssertionError):
        read_unstructured_grib(path, GRID_UUID, 32)


def test_rejects_multiple_messages(tmp_path: Path) -> None:
    path = tmp_path / "field.grib2"
    _write_grib(path, np.arange(32, dtype=np.float64))
    path.write_bytes(path.read_bytes() * 2)

    with pytest.raises(AssertionError):
        read_unstructured_grib(path, GRID_UUID, 32)


def test_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.grib2"
    path.touch()

    with pytest.raises(AssertionError):
        read_unstructured_grib(path, GRID_UUID, 32)


def test_bitmap_missing_cells_are_nan_without_masking_valid_9999(
    tmp_path: Path,
) -> None:
    values = np.array([1, -99999, 9999, -3, -99999], dtype=np.float64)
    path = tmp_path / "bitmap.grib2"
    _write_grib(path, values, missing_value=-99999)

    result = read_unstructured_grib(path, GRID_UUID, values.size)

    np.testing.assert_array_equal(result, [1, np.nan, 9999, -3, np.nan])


def test_parallel_decodes_are_bit_exact_with_serial(tmp_path: Path) -> None:
    paths = [tmp_path / f"field-{index}.grib2" for index in range(8)]
    for index, path in enumerate(paths):
        values = np.random.default_rng(index).uniform(-100, 100, POINT_COUNT)
        _write_grib(path, values)
    serial = {
        path: read_unstructured_grib(path, GRID_UUID, POINT_COUNT).view(np.uint32)
        for path in paths
    }

    def decode_and_compare(path: Path) -> None:
        result = read_unstructured_grib(path, GRID_UUID, POINT_COUNT)
        np.testing.assert_array_equal(result.view(np.uint32), serial[path])

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(decode_and_compare, paths * 8))
