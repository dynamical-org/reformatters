import bz2
import urllib.request

import xarray as xr

url = (
    "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/t/"
    "icon-eu_europe_regular-lat-lon_model-level_2026062100_000_10_T.grib2.bz2"
)
raw = bz2.decompress(urllib.request.urlopen(url, timeout=60).read())
open("/tmp/icon_t_ml10.grib2", "wb").write(raw)
print(f"downloaded+decompressed {len(raw)} bytes")

ds = xr.open_dataset("/tmp/icon_t_ml10.grib2", engine="cfgrib")
print("data_vars:", list(ds.data_vars))
v = next(iter(ds.data_vars.values()))
print("shape:", v.shape, "dims:", v.dims)
print("coords of interest:")
for c in ds.coords:
    val = ds.coords[c].values
    if val.ndim == 0:
        print(f"   {c} = {val}")
print(
    "attrs typeOfLevel / GRIB level:",
    {k: v.attrs.get(k) for k in v.attrs if "level" in k.lower() or "Level" in k},
)
