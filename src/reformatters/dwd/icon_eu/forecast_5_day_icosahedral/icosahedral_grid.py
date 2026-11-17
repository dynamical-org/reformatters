import io
import warnings
from functools import cache
from pathlib import Path
from typing import Any, Final

import numcodecs
import numpy as np
import rasterio
import xarray as xr
from zarr.codecs import BloscCodec

from reformatters.common.binary_rounding import round_float32_inplace
from reformatters.common.logging import get_logger
from reformatters.common.types import Array1D, Array2D

log = get_logger(__name__)

CELL_COUNT: Final = 768_608
NODE_COUNT: Final = 385_555
GRID_UUID: Final = "832c1507-fbc7-3eb3-8413-9890094d2c60"
GRID_NAME: Final = "ICON grid 62 (R03B08)"
SPHERE_RADIUS_METRES: Final = 6_371_229
GRID_PATH: Final = Path(__file__).with_name("grid.npz.blosc")


def cell_indices() -> Array1D[np.int32]:
    return np.arange(CELL_COUNT, dtype=np.int32)


@cache
def grid_arrays() -> dict[str, np.ndarray[Any, Any]]:
    with np.load(
        io.BytesIO(numcodecs.Blosc().decode(GRID_PATH.read_bytes())),
        allow_pickle=False,
    ) as archive:
        arrays = dict(archive)
    for array in arrays.values():
        array.flags.writeable = False
    return arrays


def cell_centres() -> tuple[Array1D[np.float32], Array1D[np.float32]]:
    arrays = grid_arrays()
    return arrays["latitude"], arrays["longitude"]


def mesh_dataset() -> xr.Dataset:
    arrays = grid_arrays()
    ds = xr.Dataset(
        {
            "mesh": (
                (),
                np.int32(0),
                {
                    "cf_role": "mesh_topology",
                    "topology_dimension": 2,
                    "node_coordinates": "node_longitude node_latitude",
                    "face_node_connectivity": "face_node_connectivity",
                    "face_dimension": "cell",
                    "node_dimension": "node",
                },
            ),
            "node_latitude": (
                "node",
                arrays["node_latitude"],
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "node_longitude": (
                "node",
                arrays["node_longitude"],
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "face_node_connectivity": (
                ("cell", "max_nodes_per_face"),
                arrays["face_node_connectivity"],
                {"cf_role": "face_node_connectivity", "start_index": 0},
            ),
            "cell_area": (
                "cell",
                arrays["cell_area"],
                {"standard_name": "cell_area", "units": "m2"},
            ),
        }
    )
    compressor = BloscCodec(
        typesize=4, cname="zstd", clevel=3, shuffle="bitshuffle"
    ).to_dict()
    for var in ds.data_vars.values():
        var.encoding.update(
            dtype=var.dtype,
            chunks=var.shape,
            compressors=[compressor],
            _FillValue=None,
            fill_value=np.nan if var.dtype == np.float32 else -1,
        )
    return ds


def _round_coordinates(original: Array2D[np.float64]) -> Array2D[np.float32]:
    latitude = np.deg2rad(original[0])
    for keep_bits in range(24):
        rounded = original.astype(np.float32)
        round_float32_inplace(rounded, keep_bits)
        delta_latitude, delta_longitude = np.deg2rad(
            rounded.astype(np.float64) - original
        )
        haversine = (
            np.sin(delta_latitude / 2) ** 2
            + np.cos(latitude)
            * np.cos(latitude + delta_latitude)
            * np.sin(delta_longitude / 2) ** 2
        )
        max_error = float(
            (2 * SPHERE_RADIUS_METRES * np.arcsin(np.sqrt(haversine))).max()
        )
        if max_error <= 2.5:
            log.info(
                f"Coordinates: {keep_bits} mantissa bits, {max_error:.6f} m maximum error"
            )
            return rounded
    raise AssertionError("Float32 coordinates exceed 2.5 m horizontal error")


def regenerate_grid(grid_path: Path, output_path: Path = GRID_PATH) -> None:
    """Write native geometry from the official ICON grid 62 NetCDF file."""
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    with (
        warnings.catch_warnings(),
        rasterio.Env(GDAL_NETCDF_BOTTOMUP="NO"),
    ):
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(grid_path) as source:
            assert source.tags()["NC_GLOBAL#uuidOfHGrid"] == GRID_UUID
            assert (
                float(source.tags()["NC_GLOBAL#semi_major_axis"])
                == SPHERE_RADIUS_METRES
            )
        for names in (("clat", "clon"), ("vlat", "vlon")):
            coordinates = []
            for name in names:
                with rasterio.open(f'NETCDF:"{grid_path}":{name}') as source:
                    coordinates.append(np.rad2deg(source.read(1).ravel()))
            latitude, longitude = _round_coordinates(np.stack(coordinates))
            prefix = "node_" if names[0] == "vlat" else ""
            arrays[f"{prefix}latitude"] = latitude
            arrays[f"{prefix}longitude"] = longitude
        with rasterio.open(f'NETCDF:"{grid_path}":vertex_of_cell') as source:
            arrays["face_node_connectivity"] = (source.read(1).T - 1).astype(np.int32)
        with rasterio.open(f'NETCDF:"{grid_path}":cell_area') as source:
            arrays["cell_area"] = source.read(1).ravel().astype(np.float32)
    assert arrays["latitude"].size == CELL_COUNT
    assert arrays["node_latitude"].size == NODE_COUNT
    assert arrays["face_node_connectivity"].shape == (CELL_COUNT, 3)
    assert arrays["face_node_connectivity"].min() == 0
    assert arrays["face_node_connectivity"].max() < NODE_COUNT
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=False, **arrays)
    codec = numcodecs.Blosc(
        cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE, typesize=4
    )
    output_path.write_bytes(codec.encode(buffer.getvalue()))
