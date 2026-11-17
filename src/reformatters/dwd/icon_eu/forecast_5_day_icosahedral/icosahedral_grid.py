import io
import warnings
from functools import cache
from pathlib import Path
from typing import Final

import numcodecs
import numpy as np
import rasterio

from reformatters.common.binary_rounding import round_float32_inplace
from reformatters.common.types import Array1D

CELL_COUNT: Final = 768_608
GRID_UUID: Final = "832c1507fbc73eb384139890094d2c60"
SPHERE_RADIUS_METRES: Final = 6_371_229
CENTRES_PATH: Final = Path(__file__).with_name("cell_centres.npy.blosc")


def cell_indices() -> Array1D[np.int64]:
    return np.arange(CELL_COUNT, dtype=np.int64)


@cache
def cell_centres() -> tuple[Array1D[np.float32], Array1D[np.float32]]:
    centres = np.load(
        io.BytesIO(numcodecs.Blosc().decode(CENTRES_PATH.read_bytes())),
        allow_pickle=False,
    )
    assert centres.shape == (2, CELL_COUNT)
    assert centres.dtype == np.float32
    centres.flags.writeable = False
    return centres[0], centres[1]


def regenerate_cell_centres(grid_path: Path, output_path: Path = CENTRES_PATH) -> None:
    """Write rounded cell centres from the official ICON grid 62 NetCDF file."""
    coordinates: list[Array1D[np.float64]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        for name in ("clat", "clon"):
            with rasterio.open(f'NETCDF:"{grid_path}":{name}') as source:
                coordinates.append(np.rad2deg(source.read(1).ravel()))
    original = np.stack(coordinates)
    assert original.shape == (2, CELL_COUNT)
    for keep_bits in range(24):
        rounded = round_float32_inplace(original.astype(np.float32), keep_bits)
        error = np.deg2rad(rounded.astype(np.float64) - original) * SPHERE_RADIUS_METRES
        error[1] *= np.cos(np.deg2rad(original[0]))
        if np.hypot(error[0], error[1]).max() <= 2.5:
            break
    else:
        raise AssertionError("Float32 cell centres exceed 2.5 m horizontal error")
    buffer = io.BytesIO()
    np.save(buffer, rounded, allow_pickle=False)
    codec = numcodecs.Blosc(
        cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE, typesize=4
    )
    output_path.write_bytes(codec.encode(buffer.getvalue()))
