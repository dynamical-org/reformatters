from pathlib import Path

import eccodes
import numpy as np

from reformatters.common.types import Array1D


def read_unstructured_grib(
    path: Path, expected_grid_uuid: str, expected_number_of_points: int
) -> Array1D[np.float32]:
    with path.open("rb") as file:
        assert eccodes.codes_count_in_file(file) == 1, (
            f"Expected exactly one GRIB message in {path}"
        )
        handle = eccodes.codes_grib_new_from_file(file)
        assert handle is not None
        try:
            assert eccodes.codes_get(handle, "gridDefinitionTemplateNumber") == 101, (
                f"Expected an unstructured grid (template 3.101) in {path}"
            )
            assert eccodes.codes_get(handle, "uuidOfHGrid") == expected_grid_uuid, (
                f"Unexpected grid UUID in {path}"
            )
            assert (
                eccodes.codes_get(handle, "numberOfDataPoints")
                == expected_number_of_points
            ), f"Unexpected number of grid points in {path}"
            values = eccodes.codes_get_values(handle).astype(np.float32)
            if eccodes.codes_get(handle, "bitmapPresent"):
                values[eccodes.codes_get_array(handle, "bitmap") == 0] = np.nan
            return values
        finally:
            eccodes.codes_release(handle)
