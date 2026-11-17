from pathlib import Path

import eccodes
import numpy as np

from reformatters.common.types import Array1D


def read_unstructured_grib(
    path: Path, expected_grid_uuid: str, expected_number_of_points: int
) -> Array1D[np.float32]:
    with path.open("rb") as file:
        message_count = eccodes.codes_count_in_file(file)
        assert message_count == 1, (
            f"Expected exactly one GRIB message, got {message_count} in {path}"
        )
        handle = eccodes.codes_grib_new_from_file(file)
        assert handle is not None
        try:
            template = eccodes.codes_get(handle, "gridDefinitionTemplateNumber")
            assert template == 101, (
                f"Expected an unstructured grid (template 3.101), got {template} in {path}"
            )
            grid_uuid = eccodes.codes_get(handle, "uuidOfHGrid")
            assert grid_uuid == expected_grid_uuid, (
                f"Expected grid UUID {expected_grid_uuid}, got {grid_uuid} in {path}"
            )
            number_of_points = eccodes.codes_get(handle, "numberOfDataPoints")
            assert number_of_points == expected_number_of_points, (
                f"Expected {expected_number_of_points} grid points, got {number_of_points} in {path}"
            )
            values = eccodes.codes_get_values(handle).astype(np.float32)
            if eccodes.codes_get(handle, "bitmapPresent"):
                values[eccodes.codes_get_array(handle, "bitmap") == 0] = np.nan
            return values
        finally:
            eccodes.codes_release(handle)
