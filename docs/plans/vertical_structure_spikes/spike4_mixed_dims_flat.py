# Decision-1 spike: can ONE flat root group hold a single-level var AND a
# pressure-level var (extra `level` dim) together? If yes, real vertical dims
# do NOT require zarr groups.
import tempfile
import icechunk, numpy as np, xarray as xr

repo = icechunk.Repository.create(icechunk.local_filesystem_storage(tempfile.mkdtemp()))
s = repo.writable_session("main")
store = s.store
ds = xr.Dataset(
    {
        "temperature_2m": (("time", "lat", "lon"), np.zeros((3, 4, 5), "float32")),
        "temperature": (
            ("time", "level", "lat", "lon"),
            np.ones((3, 6, 4, 5), "float32"),
        ),
    },
    coords={
        "time": np.arange(3),
        "level": np.array([1000, 850, 700, 500, 250, 100]),
        "lat": np.arange(4),
        "lon": np.arange(5),
    },
)
ds.to_zarr(store, mode="w", consolidated=False)
s.commit("mixed dims in one flat group")
ro = repo.readonly_session("main").store
back = xr.open_zarr(ro, consolidated=False)
print("vars + dims in ONE flat group:")
for n, v in back.data_vars.items():
    print(f"  {n:16s} {v.dims}")
print("single open_zarr, no group= needed. level coord:", back.level.values)
print("temperature.sel(level=500) shape:", back.temperature.sel(level=500).shape)
