# Virtual Zarr + Icechunk Prototype Results

Prototype exploring virtual zarr datasets backed by Icechunk, using GFS-like weather data.

**Libraries**: virtualizarr 2.4.0, icechunk 1.1.15, xarray 2026.x, zarr 3.1.3, gribberish 0.29.0

**Scripts**:
- [`prototypes/virtual_zarr_icechunk.py`](virtual_zarr_icechunk.py) — NetCDF4 approach (full pipeline demo)
- [`prototypes/virtual_grib_gribberish.py`](virtual_grib_gribberish.py) — Direct GRIB approach using gribberish

## Overview

Virtual zarr datasets store *references* (byte ranges) to data in existing files (NetCDF4, HDF5, GRIB, etc.) rather than copying the data. Icechunk manages these references with version control (git-like commits, branches, time travel). Together they allow presenting archival data as a single zarr dataset without any data duplication.

## Key findings

### GRIB: kerchunk doesn't work, but gribberish does

**kerchunk** (`kerchunk.grib2.scan_grib`) scans GRIB files and produces reference dicts, but these use a `numcodecs.grib` codec not available in zarr v3. VirtualiZarr rejects these references.

**[gribberish](https://github.com/mpiannucci/gribberish)** solves this. It provides `gribberish.zarr.GribberishCodec`, a proper zarr v3 `ArrayBytesCodec` that decodes raw GRIB2 messages. The pipeline:

1. Use gribberish's low-level API to scan GRIB → byte offsets per message
2. Build VirtualiZarr `ManifestArray` objects with `GribberishCodec` as the codec
3. Write to Icechunk with `vds.virtualize.to_icechunk(store)`

This enables **direct GRIB virtualization without any format conversion**.

Note: gribberish also has a `gribberish.kerchunk` module with `scan_gribberish()`, but it uses a numcodecs v2 codec class that is incompatible with zarr v3. The manual ManifestArray approach using `gribberish.zarr.GribberishCodec` is needed until `scan_gribberish` is updated for zarr v3, or a native VirtualiZarr GRIB parser is added ([VirtualiZarr #312](https://github.com/zarr-developers/VirtualiZarr/issues/312)).

### What works well

1. **VirtualiZarr + HDF5/NetCDF4**: `open_virtual_dataset()` with `HDFParser()` cleanly creates virtual xarray datasets from NetCDF4 files. Each file's data arrays become `ManifestArray` objects that store byte-range references.

2. **xarray operations on virtual datasets**: rename, concat, assign_coords, and attribute updates all work on virtual datasets without loading data.

3. **Icechunk storage**: `vds.virtualize.to_icechunk(store)` writes virtual references. Reading back with `xr.open_zarr(store)` returns a normal lazy xarray dataset. Data reads go through icechunk to the original files.

4. **Incremental append**: `to_icechunk(store, append_dim="lead_time")` appends new data along a dimension, growing the dataset. Each append is a separate commit.

5. **Version history + time travel**: Icechunk tracks all commits. You can read any previous snapshot by ID, seeing the dataset as it was before an append.

## Step-by-step results

### Step 0: Source data

Generated 6 NetCDF4 files mimicking GFS 0.25-degree output (2 init times x 3 lead times, 7 surface variables, 181x360 lat/lon grid = ~1.8 MB each).

In production, these would be real GFS/HRRR NetCDF4 or HDF5 files on S3.

### Step 1: VirtualiZarr

```python
import virtualizarr as vz
from virtualizarr.parsers import HDFParser
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

registry = ObjectStoreRegistry()
registry.register("file:///", LocalStore(prefix="/"))

vds = vz.open_virtual_dataset(
    "file:///path/to/gfs_data.nc",
    registry=registry,
    parser=HDFParser(),
    loadable_variables=["time", "step", "latitude", "longitude", "valid_time"],
)
# Concat multiple files along a dimension
combined = xr.concat([vds1, vds2, vds3], dim="step")
```

**Result** (per init time, 3 lead times combined):
```
<xarray.Dataset> Size: 5MB
Dimensions:     (step: 3, latitude: 181, longitude: 360)
Coordinates:
  * step        (step) timedelta64[ns] 24B 00:00:00 03:00:00 06:00:00
    valid_time  (step) datetime64[ns] 24B 2026-03-13 ... 2026-03-13T06:00:00
  * latitude    (latitude) float32 724B 90.0 89.0 88.0 ... -88.0 -89.0 -90.0
  * longitude   (longitude) float32 1kB 0.0 1.0 2.0 3.0 ... 357.0 358.0 359.0
    time        datetime64[ns] 8B 2026-03-13
Data variables:
    t2m         (step, latitude, longitude) float32 782kB ManifestArray<shape=(3, 181, 360)>
    d2m         (step, latitude, longitude) float32 782kB ManifestArray<shape=(3, 181, 360)>
    u10         (step, latitude, longitude) float32 782kB ManifestArray<shape=(3, 181, 360)>
    v10         (step, latitude, longitude) float32 782kB ManifestArray<shape=(3, 181, 360)>
    sp          (step, latitude, longitude) float32 782kB ManifestArray<shape=(3, 181, 360)>
    prmsl       (step, latitude, longitude) float32 782kB ManifestArray<shape=(3, 181, 360)>
    gust        (step, latitude, longitude) float32 782kB ManifestArray<shape=(3, 181, 360)>
```

Note the `ManifestArray` data type - no actual data is loaded, only byte-range references.

### Step 1b: Direct GRIB virtualization with gribberish

For GRIB files, use gribberish to scan byte offsets and build ManifestArrays with the zarr v3 GribberishCodec:

```python
import fsspec
import numpy as np
from gribberish import parse_grib_dataset
from gribberish.kerchunk.mapper import _split_file
from gribberish.zarr.codec import GribberishCodec
from virtualizarr.manifests import ManifestArray, ChunkManifest
from virtualizarr.manifests.utils import create_v3_array_metadata

url = "s3://noaa-gfs-bdp-pds/gfs.20260313/00/atmos/gfs.t00z.pgrb2.0p25.f000"

# Scan GRIB file for byte offsets
var_refs = {}
with fsspec.open(url, "rb", anon=True) as f:
    for offset, size, data in _split_file(f):
        dataset = parse_grib_dataset(data, encode_coords=True)
        var_name = next(iter(dataset["data_vars"]))
        var_data = dataset["data_vars"][var_name]
        var_refs[var_name] = {
            "offset": offset, "size": size,
            "shape": tuple(var_data["values"]["shape"]),
            "dims": var_data["dims"], "attrs": var_data["attrs"],
        }

# Build ManifestArray with GribberishCodec for each variable
for var_name, info in var_refs.items():
    manifest = ChunkManifest.from_arrays(
        paths=np.array([url], dtype=np.dtypes.StringDType()),
        offsets=np.array([info["offset"]], dtype=np.uint64),
        lengths=np.array([info["size"]], dtype=np.uint64),
    )
    metadata = create_v3_array_metadata(
        shape=info["shape"], chunk_shape=info["shape"],
        data_type=np.dtype("float64"),
        codecs=[GribberishCodec(var=var_name).to_dict()],
        fill_value=np.nan, attributes=info["attrs"],
        dimension_names=info["dims"],
    )
    marr = ManifestArray(metadata=metadata, chunkmanifest=manifest)
    # ... add to xr.Dataset as xr.Variable(info["dims"], marr)
```

**Result** (6 surface variables from real GFS GRIB on S3):
```
<xarray.Dataset> Size: 50MB
Dimensions:  (time: 1, latitude: 721, longitude: 1440)
Data variables:
    prmsl    (time, latitude, longitude) float64 8MB ManifestArray<shape=(1, 721, 1440)>
    ugrd     (time, latitude, longitude) float64 8MB ManifestArray<shape=(1, 721, 1440)>
    vgrd     (time, latitude, longitude) float64 8MB ManifestArray<shape=(1, 721, 1440)>
    gust     (time, latitude, longitude) float64 8MB ManifestArray<shape=(1, 721, 1440)>
    tmp      (time, latitude, longitude) float64 8MB ManifestArray<shape=(1, 721, 1440)>
    dpt      (time, latitude, longitude) float64 8MB ManifestArray<shape=(1, 721, 1440)>

Codecs for 'tmp': (GribberishCodec(var='tmp'),)
```

Each ManifestArray points to a byte range in the original GRIB file on S3. The GribberishCodec (Rust-based) decodes the GRIB2 message at read time.

### Step 2: Store in Icechunk

```python
import icechunk

storage = icechunk.local_filesystem_storage("/path/to/repo")
# For S3 sources, use s3_anonymous_credentials() or s3_credentials()
repo = icechunk.Repository.create(
    storage=storage,
    config=icechunk.RepositoryConfig(
        virtual_chunk_containers={
            "s3://noaa-gfs-bdp-pds/": icechunk.VirtualChunkContainer(
                url_prefix="s3://noaa-gfs-bdp-pds/",
                store=icechunk.ObjectStoreConfig.S3Compatible(
                    region="us-east-1",
                    endpoint_url=None,
                    anonymous=True,
                    allow_http=False,
                ),
            ),
        },
    ),
    authorize_virtual_chunk_access=icechunk.containers_credentials({
        "s3://noaa-gfs-bdp-pds/": icechunk.s3_anonymous_credentials(),
    }),
)

session = repo.writable_session("main")
vds.virtualize.to_icechunk(session.store)
session.commit("Initial GFS virtual dataset")

# Read back as normal xarray
ds = xr.open_zarr(repo.readonly_session(branch="main").store, consolidated=False)
```

**Result** (read back from Icechunk):
```
<xarray.Dataset> Size: 5MB
Dimensions:                  (lead_time: 3, latitude: 181, longitude: 360)
Coordinates:
  * lead_time                (lead_time) timedelta64[ns] 24B 00:00:00 ... 06:00:00
    valid_time               (lead_time) datetime64[ns] 24B dask.array<chunksize=(3,)>
  * latitude                 (latitude) float32 724B 90.0 89.0 ... -89.0 -90.0
  * longitude                (longitude) float32 1kB 0.0 1.0 2.0 ... 358.0 359.0
    init_time                datetime64[ns] 8B ...
    time                     datetime64[ns] 8B ...
Data variables:
    temperature_2m           (lead_time, latitude, longitude) float32 782kB dask.array<chunksize=(1, 181, 360)>
    dewpoint_temperature_2m  (lead_time, latitude, longitude) float32 782kB dask.array<chunksize=(1, 181, 360)>
    wind_u_10m               (lead_time, latitude, longitude) float32 782kB dask.array<chunksize=(1, 181, 360)>
    wind_v_10m               (lead_time, latitude, longitude) float32 782kB dask.array<chunksize=(1, 181, 360)>
    surface_pressure         (lead_time, latitude, longitude) float32 782kB dask.array<chunksize=(1, 181, 360)>
    mean_sea_level_pressure  (lead_time, latitude, longitude) float32 782kB dask.array<chunksize=(1, 181, 360)>
    wind_gust                (lead_time, latitude, longitude) float32 782kB dask.array<chunksize=(1, 181, 360)>
```

Note the data is now `dask.array` - reads are lazy and go through Icechunk to the source files.

### Step 3: Modify variable names and attributes

Variable renaming and attribute updates work on virtual datasets before writing to Icechunk:

```python
vds = vds.rename({"t2m": "temperature_2m", "u10": "wind_u_10m", ...})
vds["temperature_2m"].attrs.update({
    "standard_name": "air_temperature",
    "units": "K",
    "long_name": "2 metre temperature",
})
```

These modifications are purely metadata - no data copying occurs.

### Step 4: Add additional coordinates

New coordinates (e.g., `init_time`, `valid_time`) can be computed and added:

```python
init_time = pd.Timestamp("2026-03-13T00:00:00")
vds = vds.assign_coords(init_time=init_time)

# Compute valid_time = init_time + lead_time
valid_times = [init_time + pd.Timedelta(hours=h) for h in [0, 3, 6]]
vds = vds.assign_coords(valid_time=("step", valid_times))

# Rename dimensions
vds = vds.rename({"step": "lead_time"})
```

The added coordinates are small arrays stored directly in Icechunk (not virtual references).

### Step 5: Incremental append

```python
# Process new data (e.g., new init time)
vds_new = virtualize_and_transform(new_files)

# Append to existing dataset
session = repo.writable_session("main")
vds_new.virtualize.to_icechunk(session.store, append_dim="lead_time")
session.commit("Append GFS 06z data")
```

**Result** (after append):
```
Dimensions: (lead_time: 6, latitude: 181, longitude: 360)   # was 3, now 6
```

**Version history**:
```
5KJ42JCS...: "Append GFS 06z data"                (2026-03-14T00:56:10Z)
BK1NFTN7...: "Initial virtual dataset: GFS 00z"   (2026-03-14T00:56:10Z)
1CECHNKR...: "Repository initialized"             (2026-03-14T00:56:10Z)
```

**Time travel** - read a previous version:
```python
ds_old = xr.open_zarr(
    repo.readonly_session(snapshot_id="BK1NFTN7...").store,
    consolidated=False,
)
# ds_old has lead_time: 3 (pre-append state)
```

## Integration considerations for this repo

### How this could fit alongside rechunked datasets

Virtual datasets could serve as a "fast path" companion to the fully rechunked zarr archives:

1. **Operational updates**: When new GFS/HRRR data arrives, immediately create virtual references (seconds) while the full rechunk pipeline runs in the background (minutes/hours). Users get access to latest data immediately.

2. **Historical archives**: For archival data that doesn't need rechunking (e.g. accessing specific variables from NODD), virtual references avoid duplicating terabytes of data.

3. **Development/testing**: Virtual datasets are fast to create and don't require storage for the data itself, useful for testing pipeline changes.

### Limitations to consider

- **GRIB virtualization requires gribberish**: kerchunk's GRIB scanner uses `numcodecs.grib` (zarr v2 only). Direct GRIB virtualization works via [gribberish](https://github.com/mpiannucci/gribberish)'s zarr v3 `GribberishCodec`, but requires manually building ManifestArrays (no high-level `open_virtual_dataset` parser yet — tracked in [VirtualiZarr #312](https://github.com/zarr-developers/VirtualiZarr/issues/312)).
- **Read performance**: Virtual references read data from the original files, which may not be optimally chunked for the access patterns users want. This is the core reason rechunking exists.
- **Source file durability**: Virtual datasets break if source files are moved or deleted. NODD data rotates off after ~10 days for real-time products.
- **No compression control**: Data is read in whatever compression the source file uses - can't optimize for zarr-specific codecs like zstd or blosc.
- **Append dimension choice**: The append dimension must exist at write time and match between initial write and subsequent appends. Both datasets must have the same structure for non-append dimensions.

### Relevant icechunk configuration for S3

For virtualizing real NOAA NODD files on S3:

```python
repo = icechunk.Repository.create(
    storage=icechunk.s3_storage(
        bucket="your-icechunk-repo-bucket",
        prefix="virtual-gfs/",
        region="us-east-1",
    ),
    config=icechunk.RepositoryConfig(
        virtual_chunk_containers={
            "s3://noaa-gfs-bdp-pds/": icechunk.VirtualChunkContainer(
                url_prefix="s3://noaa-gfs-bdp-pds/",
                store=icechunk.ObjectStoreConfig.S3Compatible(
                    region="us-east-1",
                    endpoint_url=None,
                    anonymous=True,
                    allow_http=False,
                ),
            ),
        },
    ),
    authorize_virtual_chunk_access=icechunk.containers_credentials({
        "s3://noaa-gfs-bdp-pds/": icechunk.s3_anonymous_credentials(),
    }),
)
```
