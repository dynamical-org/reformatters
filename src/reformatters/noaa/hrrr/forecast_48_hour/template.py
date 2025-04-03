from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyproj
import rioxarray  # noqa: F401  Adds .rio accessor to datasets
import xarray as xr

from reformatters.common.template_utils import assign_var_metadata, make_empty_variable
from reformatters.common.template_utils import (
    write_metadata as write_metadata,  # re-export
)
from reformatters.noaa.hrrr.read_data import (
    SourceFileCoords,
    download_file,
)

from .template_config import (
    COORDINATES,
    DATA_VARIABLES,
    DATASET_ATTRIBUTES,
    DIMS,
    EXPECTED_FORECAST_LENGTH_BY_INIT_HOUR,
    get_template_dimension_coordinates,
)
from .template_config import DATASET_ID as DATASET_ID

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "latest.zarr"


def update_template() -> None:
    coords = get_template_dimension_coordinates()

    data_vars = {
        var_config.name: make_empty_variable(DIMS, coords, var_config.encoding.dtype)
        for var_config in DATA_VARIABLES
    }

    ds = xr.Dataset(data_vars, coords, DATASET_ATTRIBUTES.model_dump(exclude_none=True))

    ds = ds.rio.write_crs(
        "+proj=lcc +a=6371229 +b=6371229 +lon_0=262.5 +lat_0=38.5 +lat_1=38.5 +lat_2=38.5"
    )

    ds = ds.assign_coords(derive_coordinates(ds))

    assert {d.name for d in DATA_VARIABLES} == set(ds.data_vars)
    for var_config in DATA_VARIABLES:
        assign_var_metadata(ds[var_config.name], var_config)

    assert {c.name for c in COORDINATES} == set(ds.coords)
    for coord_config in COORDINATES:
        # Don't overwrite -- retain the attributes that .rio.write_crs adds
        if coord_config.name == "spatial_ref":
            continue

        assign_var_metadata(ds.coords[coord_config.name], coord_config)

    write_metadata(ds, _TEMPLATE_PATH, mode="w")


def derive_coordinates(
    ds: xr.Dataset,
) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
    # derivation of HRRR latitude / longitude coordinates is simplest by downloading
    # a sample file and extracting the coordinates from the file itself.
    data_coords = SourceFileCoords(
        {
            "init_time": pd.Timestamp("2025-01-01T00:00:00"),
            "lead_time": pd.Timedelta("0h"),
            "domain": "conus",
        }
    )

    # "sfc" is the smallest available file type.
    # TODO (tony): ensure we only pull down 1 grib message for this to save download time/bdwth
    _, filepath = download_file(data_coords, "sfc")
    hrrrds = xr.open_dataset(str(filepath), engine="rasterio")

    hrrrds_bounds = hrrrds.rio.bounds(recalc=True)
    hrrrds_res = hrrrds.rio.resolution(recalc=True)
    dx, dy = (
        hrrrds_res[0],
        hrrrds_res[1] * -1,
    )  # y offset is positive, but reported as negative by `rio.resolution`
    # this may be a bug in how rioxarray is handling bounds that cross a central parallel
    # but also I'd expect that to be the case with lat/lon coordinates too, so I'm not sure.

    proj_xcorner, proj_ycorner = hrrrds_bounds[0], hrrrds_bounds[1]
    nx = ds.x.size
    ny = ds.y.size

    pj = pyproj.Proj(ds.rio.crs.to_proj4())
    # rio.bounds returns the lower left corner, but we want the center of the gridcell
    # so we offset by half the gridcell size.
    x = (proj_xcorner + (0.5 * dx)) + np.arange(nx) * dx
    y = (proj_ycorner + (0.5 * dy)) + np.arange(ny) * dy
    xs, ys = np.meshgrid(x, y)
    lons, lats = pj(xs, ys, inverse=True)

    return {
        "y": (("y",), y),
        "x": (("x",), x),
        "latitude": (("y", "x"), lats),
        "longitude": (("y", "x"), lons),
        "ingested_forecast_length": (
            ("init_time",),
            np.full((ds["init_time"].size), np.timedelta64("NaT", "ns")),
        ),
        "expected_forecast_length": (
            ("init_time",),
            EXPECTED_FORECAST_LENGTH_BY_INIT_HOUR.loc[ds["init_time"].dt.hour],
        ),
        "valid_time": ds["init_time"] + ds["lead_time"],
    }
