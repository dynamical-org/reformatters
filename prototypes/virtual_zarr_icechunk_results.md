# Virtual Zarr + Icechunk Prototype Results

Prototype exploring virtual zarr datasets backed by Icechunk, using GFS-like weather data.

**Libraries**: icechunk 1.1.15, zarr 3.1.3, gribberish 0.29.0 (read-side codec only), xarray 2026.x

**Scripts**:
- [`prototypes/virtual_zarr_icechunk.py`](virtual_zarr_icechunk.py) — NetCDF4 approach (full pipeline demo)
- [`prototypes/virtual_grib_gribberish.py`](virtual_grib_gribberish.py) — Direct GRIB approach using gribberish

## Proposed pipeline

Virtual GRIB datasets can be built using just icechunk + zarr + GRIB index files. No VirtualiZarr or gribberish needed at write time — gribberish is only the read-side codec that zarr uses to decode GRIB bytes when someone reads the data.

### Write-time dependencies

- **icechunk**: store + version control
- **zarr**: array metadata (shape, dtype, codecs, attributes)
- **GRIB `.idx` files**: small text files published by NOAA alongside every GRIB, containing variable names + byte offsets + lengths

### Read-time dependencies

- **icechunk + zarr**: serves chunks
- **gribberish**: `GribberishCodec` (zarr v3 `ArrayBytesCodec`) decodes GRIB bytes on read

### One-time setup

Create the zarr structure in icechunk with the right metadata. No data is written — just array definitions, fixed coordinates, and attributes.

This can reuse our existing `TemplateConfig` subclasses almost entirely. The template config already defines all dimensions, variable names, CF attributes, coordinate arrays, and dataset-level metadata. Each `DataVar`'s `internal_attrs.grib_element` provides the GRIB variable name mapping. The only differences for virtual datasets are the codecs (GribberishCodec instead of sharding+blosc+zstd), chunk shape (one GRIB message per chunk), and dtype (float64 from GribberishCodec).

```python
import zarr
import icechunk
from gribberish.zarr.codec import GribberishCodec

# Reuse existing template config — it already has all dims, coords, attrs, var definitions
from reformatters.noaa.gfs.forecast.template_config import GFSForecastTemplateConfig

template_config = GFSForecastTemplateConfig()

# Create icechunk repo with virtual chunk container for NOAA S3
storage = icechunk.s3_storage(bucket="our-icechunk-bucket", prefix="virtual-gfs/", region="us-east-1")
config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer("s3://noaa-gfs-bdp-pds/", icechunk.s3_store(region="us-east-1"))
)
repo = icechunk.Repository.create(storage, config)

session = repo.writable_session("main")
root = zarr.open_group(session.store, mode="w")

# Data variable arrays — reuse names, attrs from template config.
# Swap encoding: GribberishCodec instead of sharding+blosc+zstd,
# one chunk per GRIB message, float64 output.
spatial_shape = (template_config.dims["latitude"], template_config.dims["longitude"])
n_lead_times = template_config.dims["lead_time"]

for data_var in template_config.data_vars:
    grib_var = data_var.internal_attrs.grib_element  # e.g. "TMP", "UGRD"
    root.create_array(
        data_var.name,
        shape=(0, n_lead_times, *spatial_shape),
        chunk_shape=(1, 1, *spatial_shape),
        dtype="float64",
        codecs=[GribberishCodec(var=grib_var).to_dict()],
        fill_value=float("nan"),
        dimension_names=list(template_config.dims.keys()),
        attributes=data_var.attrs.model_dump(exclude_none=True),  # CF attrs from template
    )

# Coordinates — reuse from template config's dimension_coordinates() and coord definitions.
# Fixed coordinates (written once)
dim_coords = template_config.dimension_coordinates()
for coord in template_config.coords:
    if coord.name in ("latitude", "longitude", "lead_time"):
        root.create_array(
            coord.name,
            data=dim_coords[coord.name],
            dtype=coord.encoding.dtype,
            dimension_names=[coord.name],
            attributes=coord.attrs.model_dump(exclude_none=True),
        )

# Growable coordinates (start empty, grow with init_time)
for coord in template_config.coords:
    if coord.name == "init_time":
        root.create_array(
            "init_time", shape=(0,),
            chunk_shape=(template_config.append_dim_coordinate_chunk_size(),),
            dtype=coord.encoding.dtype,
            dimension_names=["init_time"],
            attributes=coord.attrs.model_dump(exclude_none=True),
        )
    elif coord.name == "valid_time":
        root.create_array(
            "valid_time", shape=(0, n_lead_times),
            chunk_shape=(template_config.append_dim_coordinate_chunk_size(), n_lead_times),
            dtype=coord.encoding.dtype,
            dimension_names=["init_time", "lead_time"],
            attributes=coord.attrs.model_dump(exclude_none=True),
        )

# Dataset-level attributes — also from template config
root.attrs.update(template_config.dataset_attributes.model_dump(exclude_none=True))

session.commit("Initialize virtual GFS dataset")
```

This shares all variable names, CF attributes (`standard_name`, `long_name`, `units`, `step_type`), coordinate definitions, dimension structure, and dataset metadata with the rechunked dataset. The only virtual-specific code is the codec and chunk shape.

### Update (runs on each new data arrival)

Parse GRIB `.idx` files for byte offsets and place virtual references. Resize only if there's a new init_time.

```python
session = repo.writable_session("main")
root = zarr.open_group(session.store, mode="r+")

# --- Resize if new init_time ---
if is_new_init_time:
    init_idx = root["init_time"].shape[0]  # index of the new init_time
    new_n = init_idx + 1

    # Resize every array that has init_time as a dimension
    for name in root:
        arr = root[name]
        if "init_time" in (arr.metadata.dimension_names or []):
            new_shape = list(arr.shape)
            new_shape[arr.metadata.dimension_names.index("init_time")] = new_n
            arr.resize(tuple(new_shape))

    # Update coordinate arrays
    root["init_time"][init_idx] = int(init_ts.timestamp())
    root["valid_time"][init_idx, :] = [
        int((init_ts + lead_td).timestamp()) for lead_td in lead_timedeltas
    ]
else:
    init_idx = ...  # existing init_time index

# --- Place virtual refs from .idx files ---
# .idx files are small text files (~100KB) with lines like:
#   1:0:d=2026031300:TMP:2 m above ground:anl:
#   2:385212:d=2026031300:RH:2 m above ground:anl:
# Each line gives: message_number:byte_offset:metadata
# Byte length = next message's offset - this message's offset

for lt_idx, lead_hour in enumerate(lead_hours_to_update):
    grib_url = f"s3://noaa-gfs-bdp-pds/gfs.{date}/{init_hour}/atmos/gfs.t{init_hour}z.pgrb2.0p25.f{lead_hour:03d}"
    idx_entries = parse_idx_file(grib_url + ".idx")  # → {grib_var: (offset, length)}

    for var_name, grib_var in var_mapping.items():
        offset, length = idx_entries[grib_var]
        session.store.set_virtual_ref(
            f"{var_name}/c/{init_idx}/{lt_idx}/0/0",
            grib_url,
            offset=offset,
            length=length,
        )

session.commit(f"Add GFS {init_ts}")
```

### What this gives us

- **Write path**: parse `.idx` text files + `set_virtual_ref` calls. No GRIB data access, no scanning. Fast.
- **Read path**: `xr.open_zarr(store)` returns a lazy dataset. Reads go through icechunk → fetch GRIB byte range from S3 → GribberishCodec decodes → numpy array.
- **Partial fills**: If only 12 of 48 lead times are available, set refs for those 12. The rest return `fill_value` (NaN). When more arrive, set refs for the new ones. No resize needed.
- **Version control**: Every commit is a snapshot. Time travel to any previous state.

---

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

---

## Deep dive: GribberishCodec vs repo template structure

### How GribberishCodec works internally

`GribberishCodec` is a zarr v3 `ArrayBytesCodec` registered under the name `"gribberish"`. Key details:

- **Read-only**: `encode()` raises `NotImplementedError`. This is fine for virtual datasets (we never write GRIB data, only references to it).
- **`var` parameter is not used at read time** (except for the special `latitude`/`longitude` cases which extract spatial coords via `.latlng()`). For all data variables, the codec calls `parse_grib_array(chunk_bytes, 0)` which simply decodes whatever GRIB message is in the byte range — it doesn't match on the variable name. This means the codec decodes whatever bytes `set_virtual_ref` points to, and GRIB element renaming by the data provider (e.g. NOAA renaming `prmsl` → `msla`) is handled entirely at write time in our `.idx` parsing logic, not in the codec.
- **Output dtype**: Always `float64`. Each GRIB message is decoded to a float64 numpy array.
- **Whole-message fetch**: The entire GRIB message (byte range) must be fetched to decode any variable from it. No partial reads within a message.
- **One variable per message**: Each GRIB message contains one field. The codec decodes whatever data is in it.

### Comparison to existing rechunked dataset structure

Our rechunked GFS forecast dataset uses:

| Aspect | Rechunked dataset | Virtual GRIB dataset |
|---|---|---|
| **Dimensions** | `(init_time, lead_time, latitude, longitude)` | Achievable — we control dims when building ManifestArrays |
| **Append dim** | `init_time` | Achievable — `to_icechunk(append_dim="init_time")` |
| **Data dtype** | `float32` | `float64` (GribberishCodec always decodes to float64) |
| **Data codecs** | `sharding_indexed` → `bytes` + `blosc(zstd, level 3)` | `GribberishCodec` only (no sharding, no compression control) |
| **Chunk shape** | `(1, 105, 121, 121)` inside `(1, 210, 726, 726)` shards | One chunk = one GRIB message = `(1, 721, 1440)` for GFS 0.25° |
| **Coord dtype** | `int64` (seconds since epoch) | Computed & written directly as numpy arrays |
| **Coord codecs** | `bytes` + `blosc(zstd, level 3)` | Standard zarr codecs (not virtual) |
| **fill_value** | `0.0` | `NaN` (configurable) |
| **CF attributes** | Full: standard_name, long_name, units, step_type, etc. | We set these ourselves when building the dataset |
| **Variable names** | CF-style (e.g. `temperature_2m`) | We rename from GRIB names (e.g. `tmp` → `temperature_2m`) |
| **Derived coords** | `valid_time` (2D: init_time × lead_time), `ingested_forecast_length`, `expected_forecast_length`, `spatial_ref` | `valid_time` achievable; other derived coords written as normal arrays |

### What we can match

1. **Dimensions and dim names** — Fully achievable. We construct the ManifestArrays with whatever `dimension_names` we want.

2. **Variable names and CF attributes** — Fully achievable. We rename variables and set `standard_name`, `long_name`, `units`, `step_type` etc. before writing to Icechunk.

3. **Coordinate arrays** — Fully achievable. Coordinates like `init_time`, `lead_time`, `latitude`, `longitude`, `valid_time`, `spatial_ref` are all written as normal (non-virtual) arrays into Icechunk. We compute `valid_time = init_time + lead_time` ourselves.

4. **Dataset-level attributes** — Fully achievable. We can set `dataset_id`, `dataset_version`, `description`, etc.

5. **Append along init_time** — Achievable with `to_icechunk(append_dim="init_time")`.

### Where there are gaps

1. **Data dtype mismatch** — GribberishCodec always returns `float64`. Our rechunked datasets use `float32`. This doubles memory usage on read. Options:
   - Accept the overhead (virtual datasets prioritize availability over performance anyway)
   - Contribute a `dtype` option to gribberish upstream
   - Add a post-read cast codec in the pipeline (not currently possible in zarr v3 codec chain without a custom codec)

2. **No sharding** — GRIB messages are one field per message, so each chunk = one full spatial grid `(721, 1440)` for GFS. Our rechunked datasets use `(121, 121)` inner chunks inside `(726, 726)` shards. This means:
   - **Spatial subsetting is expensive**: Reading a small region fetches the entire global grid (~8 MB for float64), vs ~60 KB for a single chunk in the rechunked dataset.
   - This is the fundamental tradeoff of virtual datasets — you get the source file's chunking.

3. **No compression control** — Data is stored in GRIB2 compression (JPEG2000, simple packing, etc.). Can't apply zstd/blosc. Read performance depends on the GRIB compression used by the data provider.

4. **No bit-rounding/mantissa truncation** — Our rechunked datasets use `keep_mantissa_bits` to reduce data size. Virtual datasets serve the original precision.

5. **No deaccumulation** — Our rechunked pipeline converts accumulation fields (e.g. total precipitation) to rates via `deaccumulate_to_rate`. GribberishCodec serves raw values. Deaccumulation would need to happen at read time (a user-side transform) or in a wrapping codec.

6. **`ingested_forecast_length` / `expected_forecast_length`** — These are repo-specific derived coordinates tracking how much of a forecast has been ingested. They'd need to be computed and written as normal arrays, identical to how `valid_time` is handled.

### Summary: virtual datasets as a complement, not a replacement

Virtual GRIB datasets can match the *metadata structure* (dims, coords, attributes, variable names) of our rechunked datasets almost exactly. The gaps are all in *data encoding and access patterns*:
- No spatial sharding (whole-grid reads)
- float64 instead of float32
- No compression control
- No derived transformations (deaccumulation, etc.)

This confirms the "fast path" role: virtual datasets provide immediate access to new data with the right metadata shape, while rechunked datasets provide optimized read performance.

## Mixed storage: zstd-compressed coords + GRIB data vars

### Can we mix storage types in one Icechunk dataset?

**Yes.** Icechunk stores each array independently. Within a single dataset:
- **Coordinate arrays** (init_time, lead_time, latitude, longitude, valid_time, spatial_ref) are written as normal zarr arrays with standard codecs (bytes + blosc/zstd). These are small and stored directly in Icechunk.
- **Data variables** (temperature_2m, wind_u_10m, etc.) are virtual references to GRIB byte ranges, decoded by GribberishCodec at read time.

This happens naturally in the prototype — when we `assign_coords()` with numpy arrays and then call `to_icechunk()`, the coordinates are stored as native zarr chunks while the ManifestArray-backed data variables are stored as virtual references.

```python
# Coordinates: normal numpy arrays → stored as native zarr with default codecs
vds = vds.assign_coords(
    init_time=np.array([init_ts], dtype="datetime64[ns]"),
    lead_time=lead_time_values,   # timedelta64
    valid_time=(("init_time", "lead_time"), valid_time_2d),  # 2D derived coord
    latitude=lat_array,
    longitude=lon_array,
)

# Data variables: ManifestArray → stored as virtual references with GribberishCodec
# (already constructed with ManifestArray + GribberishCodec)

# Write to Icechunk — both types coexist in the same store
session = repo.writable_session("main")
vds.virtualize.to_icechunk(session.store)
session.commit("Mixed storage: native coords + virtual GRIB data")
```

When reading back:
```python
ds = xr.open_zarr(store, consolidated=False)
# ds.coords["latitude"]  → loaded from native zarr (zstd-compressed)
# ds["temperature_2m"]   → lazy dask array, reads GRIB via GribberishCodec
```

### Encoding control for coordinates

To control coordinate encoding (e.g., force int64 seconds-since-epoch for timestamps, blosc/zstd compression), set encoding before writing:

```python
# Before calling to_icechunk, encode time coordinates as int64
init_time_seconds = (init_ts - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
vds = vds.assign_coords(
    init_time=xr.Variable(
        "init_time",
        np.array([init_time_seconds], dtype="int64"),
        attrs={
            "units": "seconds since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
            "standard_name": "forecast_reference_time",
        },
    ),
)
```

This approach matches the encoding used by our rechunked datasets (int64 seconds since epoch with blosc/zstd).

## Updating non-GRIB coordinates (e.g. `init_time`, `valid_time`)

### The problem

When appending a new init_time to the dataset, we need to:
1. Add virtual references for the GRIB data (ManifestArrays)
2. Update coordinate arrays that aren't in the GRIB files: `init_time`, `valid_time`, `ingested_forecast_length`, etc.

### The solution

Coordinates are normal zarr arrays in Icechunk. When building the virtual dataset for a new init_time, we include both:

```python
def build_dataset_for_init_time(init_ts, grib_urls, lead_hours):
    """Build a complete virtual dataset for one init_time, ready for append."""

    # 1. Build virtual ManifestArrays for each data variable from GRIB files
    data_vars = {}
    for url, lead_h in zip(grib_urls, lead_hours):
        var_refs = scan_grib_variables(url, target_vars)
        for var_name, info in var_refs.items():
            # Build ManifestArray with shape (1, 1, nlat, nlon) for this lead time
            # ... (as shown in Step 1b above)
            pass

    # Concat along lead_time to get shape (1, n_lead, nlat, nlon)
    # ...

    # 2. Compute and assign coordinate arrays (non-virtual, stored directly)
    lead_time_ns = np.array([pd.Timedelta(hours=h) for h in lead_hours])
    valid_times = np.array([init_ts + pd.Timedelta(hours=h) for h in lead_hours])

    vds = vds.assign_coords(
        init_time=("init_time", [init_ts]),
        lead_time=("lead_time", lead_time_ns),
        valid_time=(("init_time", "lead_time"), valid_times.reshape(1, -1)),
    )

    return vds

# Append to existing dataset
session = repo.writable_session("main")
vds_new.virtualize.to_icechunk(session.store, append_dim="init_time")
session.commit(f"Add init_time={init_ts}")
```

The key insight: `to_icechunk(append_dim="init_time")` handles *both* virtual arrays (data vars) and native arrays (coordinates) in the same append operation. The init_time coordinate array grows by one element, the valid_time 2D array grows by one row, and the data variable ManifestArrays grow along the init_time dimension.

## Partial writes: filling in missing lead times

### Scenario

Dataset has dims `(init_time, lead_time, y, x)`. All init_times have complete lead_time coverage except the latest one, which only has the first 12 hours of a 48-hour forecast. Now 18 hours are available and we want to write the 6 new hours.

### Zarr's hypercube model makes this straightforward

Zarr datasets are hypercubes — all arrays share the same dimension extents. When we append a new init_time (even with only 12 of 48 lead times available), the data variable arrays are extended to `(n_init+1, 48, lat, lon)`. Chunks for lead times 12-47 simply don't exist yet and return `fill_value` on read. There's no explicit "pre-allocation" — the hypercube shape implies the slots exist, and missing chunks are fill values by definition.

So the question is just: **how do we write virtual references into existing chunk positions** when the remaining lead times become available?

### Approach 1: `set_virtual_ref` on individual chunks

Icechunk's store exposes a `set_virtual_ref` API to place virtual chunk references at specific chunk keys:

```python
session = repo.writable_session("main")
store = session.store

# For each new lead time (hours 12-17), write the virtual reference
# into the correct chunk position
for var_name in data_var_names:
    for lt_idx in range(12, 18):
        # Chunk key follows zarr v3 convention: array_name/c/dim0/dim1/.../dimN
        chunk_key = f"{var_name}/c/{init_idx}/{lt_idx}/0/0"
        store.set_virtual_ref(
            chunk_key,
            location=grib_url_for_lead_hour[lt_idx],
            offset=byte_offset,
            length=byte_length,
        )

session.commit("Fill lead times 12-17 for latest init_time")
```

This is the most direct approach — we're placing GRIB byte-range references into the exact chunk slots that were previously empty (returning fill_value).

### Approach 2: `to_icechunk(region=...)` via VirtualiZarr

VirtualiZarr has experimental `region` support for `to_icechunk`, providing a higher-level API:

```python
# Build virtual dataset for just the new lead times (hours 12-17)
vds_new = build_virtual_dataset_for_lead_times(
    init_ts=latest_init_time,
    lead_hours=range(12, 18),
)
# vds_new has shape: init_time=1, lead_time=6, lat=721, lon=1440

# Write into a specific region of the existing dataset
session = repo.writable_session("main")
vds_new.virtualize.to_icechunk(
    session.store,
    region={
        "init_time": slice(-1, None),      # last init_time
        "lead_time": slice(12, 18),         # lead times 12-17
    },
)
session.commit("Fill lead times 12-17 for latest init_time")
```

### The full workflow

```python
# 1. New init_time arrives with 12h of forecast data available
vds_12h = build_virtual_dataset(init_ts, lead_hours=range(0, 12))
# Append along init_time — zarr hypercube extends all arrays to include
# the new init_time. Lead time slots 12-47 are implicitly empty (fill_value).
session = repo.writable_session("main")
vds_12h.virtualize.to_icechunk(session.store, append_dim="init_time")
session.commit(f"Add init_time={init_ts} (12h available)")

# 2. Later, hours 12-17 become available — fill in the empty slots
vds_new_hours = build_virtual_dataset(init_ts, lead_hours=range(12, 18))
session = repo.writable_session("main")
# Use region write or set_virtual_ref to place refs in existing positions
vds_new_hours.virtualize.to_icechunk(
    session.store,
    region={"init_time": slice(-1, None), "lead_time": slice(12, 18)},
)
session.commit(f"Fill lead times 12-17 for init_time={init_ts}")
```

This matches our rechunked dataset pattern: `init_time` grows via append, and `ingested_forecast_length` tracks how much of each forecast has been filled in.

### Summary of partial write options

| Approach | Pros | Cons |
|---|---|---|
| `set_virtual_ref` on individual chunks | Fine-grained control, no overhead | Low-level API, must manage chunk keys manually |
| `to_icechunk(region=...)` | High-level, works with xarray | Experimental, may not work with virtual references yet |

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
- **float64 output**: GribberishCodec always returns float64, doubling memory vs our float32 rechunked datasets.
- **No data transformations**: Deaccumulation, unit conversion, bit-rounding all happen at read time with virtual datasets (if at all). The rechunked pipeline handles these at write time.

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
