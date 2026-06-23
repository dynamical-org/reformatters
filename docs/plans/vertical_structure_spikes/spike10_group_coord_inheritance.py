# Does a child group opened on its own carry the shared dim coords (time/lat/lon)
# that live only at the root, or only what's written into the group itself?
import tempfile
import icechunk, numpy as np, xarray as xr

repo = icechunk.Repository.create(icechunk.local_filesystem_storage(tempfile.mkdtemp()))
s = repo.writable_session("main")
store = s.store

# Root: single-level var + the shared coords (time/lat/lon)
root = xr.Dataset(
    {"temperature_2m": (("time", "lat", "lon"), np.zeros((3, 4, 5), "float32"))},
    coords={"time": np.arange(3), "lat": np.arange(4), "lon": np.arange(5)},
)
root.to_zarr(store, mode="w", consolidated=False)

# pressure_level group: temperature on (time, pressure_level, lat, lon).
# Write ONLY the pressure_level coord in the group; do NOT repeat time/lat/lon.
pl = xr.Dataset(
    {
        "temperature": (
            ("time", "pressure_level", "lat", "lon"),
            np.ones((3, 2, 4, 5), "float32"),
        )
    },
    coords={"pressure_level": np.array([1000, 500])},
)
pl.to_zarr(store, group="pressure_level", mode="a", consolidated=False)
s.commit("root coords only; group has its own z coord")

ro = repo.readonly_session("main").store
print("=== open_zarr(group='pressure_level') ALONE ===")
g = xr.open_zarr(ro, group="pressure_level", consolidated=False)
print("data_vars:", list(g.data_vars))
print("coords present:", list(g.coords))
print("has time coord values? ", "time" in g.coords)
print("has lat coord values?  ", "lat" in g.coords)
print("dims:", dict(g.sizes))

print("\n=== open_datatree (whole hierarchy) ===")
dt = xr.open_datatree(ro, engine="zarr", consolidated=False)
pl_node = dt["pressure_level"]
print("pressure_level node coords (incl inherited):", list(pl_node.coords))
print("inherited time values:", pl_node.coords.get("time", "MISSING"))
