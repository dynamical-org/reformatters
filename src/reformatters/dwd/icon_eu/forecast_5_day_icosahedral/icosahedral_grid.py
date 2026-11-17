import io
import warnings
from functools import cache
from pathlib import Path
from typing import Any, Final

import numcodecs
import numpy as np
import rasterio
import xarray as xr

from reformatters.common.binary_rounding import round_float32_inplace
from reformatters.common.config_models import Coordinate, CoordinateAttrs, Encoding
from reformatters.common.types import Array1D
from reformatters.common.zarr import BLOSC_4BYTE_ZSTD_LEVEL3_BITSHUFFLE

CELL_COUNT: Final = 768_608
NODE_COUNT: Final = 385_555
GRID_UUID: Final = "832c1507-fbc7-3eb3-8413-9890094d2c60"
GRID_NAME: Final = "ICON grid 62 (R03B08)"
SPHERE_RADIUS_METRES: Final = 6_371_229
# 21 mantissa bits maintain 2.16 m maximum horizontal error.
COORDINATE_KEEP_MANTISSA_BITS: Final = 21
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


def cell_centers() -> tuple[Array1D[np.float32], Array1D[np.float32]]:
    arrays = grid_arrays()
    return arrays["latitude"], arrays["longitude"]


@cache
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
                {"standard_name": "latitude", "units": "degree_north"},
            ),
            "node_longitude": (
                "node",
                arrays["node_longitude"],
                {"standard_name": "longitude", "units": "degree_east"},
            ),
            "face_node_connectivity": (
                ("cell", "max_nodes_per_face"),
                arrays["face_node_connectivity"],
                {"cf_role": "face_node_connectivity", "start_index": 0},
            ),
        }
    )
    for var in ds.data_vars.values():
        var.encoding = Encoding(
            dtype="float32" if var.dtype == np.float32 else "int32",
            chunks=var.shape,
            shards=None,
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_BITSHUFFLE],
            fill_value=np.nan if var.dtype == np.float32 else -1,
        ).model_dump(exclude_none=True)
    return ds


def spatial_reference_coordinate() -> Coordinate:
    wkt = (
        'GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",'
        f'SPHEROID["Sphere",{SPHERE_RADIUS_METRES},0]],PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
        'AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]'
    )
    return Coordinate(
        name="spatial_ref",
        encoding=Encoding(dtype="int64", fill_value=0, chunks=(), shards=None),
        attrs=CoordinateAttrs(
            units=None,
            statistics_approximate=None,
            crs_wkt=wkt,
            semi_major_axis=float(SPHERE_RADIUS_METRES),
            semi_minor_axis=float(SPHERE_RADIUS_METRES),
            inverse_flattening=0.0,
            reference_ellipsoid_name="Sphere",
            longitude_of_prime_meridian=0.0,
            prime_meridian_name="Greenwich",
            geographic_crs_name="Coordinate System imported from GRIB file",
            horizontal_datum_name="unnamed",
            grid_mapping_name="latitude_longitude",
            spatial_ref=wkt,
            comment="A perfect sphere geographic CRS with a radius of 6,371,229m, extracted from grib.",
        ),
    )


def regenerate_grid(grid_path: Path, output_path: Path = GRID_PATH) -> None:
    """Write native geometry from the official ICON grid 62 file, icon_grid_0062_R03B08_N02.nc from http://icon-downloads.mpimet.mpg.de/grids/public/edzw/."""
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
            rounded = np.stack(coordinates).astype(np.float32)
            round_float32_inplace(rounded, COORDINATE_KEEP_MANTISSA_BITS)
            latitude, longitude = rounded
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
