"""Spike 1: can one icechunk store hold multiple groups with different dims,
written and read back with xarray? Tests the proposed
root / single_level / pressure_level / model_level structure.
"""

import tempfile

import icechunk
import numpy as np
import xarray as xr

tmp = tempfile.mkdtemp()
repo = icechunk.Repository.create(icechunk.local_filesystem_storage(tmp))
session = repo.writable_session("main")
store = session.store

# Root: a back-compat single-level variable, dims (time, lat, lon)
root = xr.Dataset(
    {"temperature_2m": (("time", "lat", "lon"), np.zeros((3, 4, 5), "float32"))},
    coords={"time": np.arange(3), "lat": np.arange(4), "lon": np.arange(5)},
)
root.to_zarr(store, mode="w", consolidated=False)

# Group: pressure_level, with an EXTRA dim `level`, dims (time, level, lat, lon)
plevels = np.array([1000, 850, 500, 250], "int32")
pl = xr.Dataset(
    {
        "temperature": (
            ("time", "level", "lat", "lon"),
            np.ones((3, 4, 4, 5), "float32"),
        )
    },
    coords={
        "time": np.arange(3),
        "level": plevels,
        "lat": np.arange(4),
        "lon": np.arange(5),
    },
)
pl.to_zarr(store, group="pressure_level", mode="a", consolidated=False)

# Group: single_level, keeping suffix-named vars (sparse cross product)
sl = xr.Dataset(
    {
        "temperature_2m": (("time", "lat", "lon"), np.zeros((3, 4, 5), "float32")),
        "wind_u_100m": (("time", "lat", "lon"), np.zeros((3, 4, 5), "float32")),
    },
    coords={"time": np.arange(3), "lat": np.arange(4), "lon": np.arange(5)},
)
sl.to_zarr(store, group="single_level", mode="a", consolidated=False)

session.commit("multi-group write")

# ---- read back ----
ro = repo.readonly_session("main").store
print("=== read root group ===")
print(list(xr.open_zarr(ro, consolidated=False).data_vars))
print("\n=== read group='pressure_level' ===")
pds = xr.open_zarr(ro, group="pressure_level", consolidated=False)
print("dims:", dict(pds.sizes), "vars:", list(pds.data_vars))
print("\n=== read group='single_level' ===")
sds = xr.open_zarr(ro, group="single_level", consolidated=False)
print("vars:", list(sds.data_vars))

print("\n=== open_datatree (whole hierarchy at once) ===")
dt = xr.open_datatree(ro, engine="zarr", consolidated=False)
print(dt)
print(
    "\nnavigate: dt['pressure_level']['temperature'] level coord ->",
    dt["pressure_level"]["temperature"].level.values,
)
