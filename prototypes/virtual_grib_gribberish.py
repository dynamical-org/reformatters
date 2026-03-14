"""
Prototype: Direct GRIB virtualization using gribberish + VirtualiZarr + Icechunk.

Unlike the NetCDF4-based prototype, this virtualizes real NOAA GFS GRIB2 files
*directly* using gribberish's zarr v3 codec (GribberishCodec). No format
conversion needed.

Pipeline:
  1. gribberish scans GRIB file → byte offsets per message
  2. VirtualiZarr ManifestArray built with GribberishCodec
  3. Written to Icechunk as virtual references to S3 GRIB byte ranges

Run: uv run python prototypes/virtual_grib_gribberish.py
Requires: pip install gribberish virtualizarr icechunk
"""

import tempfile
from pathlib import Path

import fsspec
import numpy as np
import xarray as xr

from gribberish import parse_grib_dataset
from gribberish.kerchunk.mapper import _split_file
from gribberish.zarr.codec import GribberishCodec

from virtualizarr.manifests import ManifestArray, ChunkManifest
from virtualizarr.manifests.utils import create_v3_array_metadata

import icechunk

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GFS_URL = "s3://noaa-gfs-bdp-pds/gfs.20260313/00/atmos/gfs.t00z.pgrb2.0p25.f000"
TARGET_VARS = {"tmp", "dpt", "ugrd", "vgrd", "gust", "prmsl"}


# ============================================================================
# Step 1: Scan GRIB with gribberish → byte offsets + metadata
# ============================================================================
def scan_grib_variables(
    url: str,
    target_vars: set[str],
) -> dict[str, dict]:
    """Scan a GRIB2 file with gribberish and return byte ranges for target variables."""
    var_refs: dict[str, dict] = {}

    with fsspec.open(url, "rb", anon=True) as f:
        for offset, size, data in _split_file(f):
            try:
                dataset = parse_grib_dataset(data, encode_coords=True)
                var_name = next(iter(dataset["data_vars"]))
                if var_name in target_vars and var_name not in var_refs:
                    var_data = dataset["data_vars"][var_name]
                    var_refs[var_name] = {
                        "offset": offset,
                        "size": size,
                        "shape": tuple(var_data["values"]["shape"]),
                        "dims": var_data["dims"],
                        "attrs": var_data["attrs"],
                    }
            except Exception:
                continue

    return var_refs


# ============================================================================
# Step 2: Build VirtualiZarr ManifestArrays with GribberishCodec
# ============================================================================
def build_virtual_dataset(
    url: str,
    var_refs: dict[str, dict],
) -> xr.Dataset:
    """Create a virtual xarray Dataset from GRIB byte ranges using GribberishCodec."""
    data_vars: dict[str, xr.Variable] = {}

    for var_name, info in var_refs.items():
        manifest = ChunkManifest.from_arrays(
            paths=np.array([url], dtype=np.dtypes.StringDType()),
            offsets=np.array([info["offset"]], dtype=np.uint64),
            lengths=np.array([info["size"]], dtype=np.uint64),
        )
        metadata = create_v3_array_metadata(
            shape=info["shape"],
            chunk_shape=info["shape"],
            data_type=np.dtype("float64"),
            codecs=[GribberishCodec(var=var_name).to_dict()],
            fill_value=np.nan,
            attributes=info["attrs"],
            dimension_names=info["dims"],
        )
        marr = ManifestArray(metadata=metadata, chunkmanifest=manifest)
        data_vars[var_name] = xr.Variable(info["dims"], marr, attrs=info["attrs"])

    return xr.Dataset(data_vars)


# ============================================================================
# Step 3: Store in Icechunk
# ============================================================================
def store_in_icechunk(
    vds: xr.Dataset,
    repo_path: Path,
    s3_bucket: str,
) -> icechunk.Repository:
    """Write virtual GRIB references to a local Icechunk repository."""
    storage = icechunk.local_filesystem_storage(str(repo_path))

    s3_prefix = f"s3://{s3_bucket}/"
    s3_opts = icechunk.S3Options(region="us-east-1", anonymous=True)

    repo = icechunk.Repository.create(
        storage=storage,
        config=icechunk.RepositoryConfig(
            virtual_chunk_containers={
                s3_prefix: icechunk.VirtualChunkContainer(
                    url_prefix=s3_prefix,
                    store=icechunk.ObjectStoreConfig.S3Compatible(s3_opts),
                ),
            },
        ),
        authorize_virtual_chunk_access=icechunk.containers_credentials({
            s3_prefix: icechunk.s3_anonymous_credentials(),
        }),
    )

    session = repo.writable_session("main")
    vds.virtualize.to_icechunk(session.store)
    snap = session.commit("GFS GRIB virtual dataset via GribberishCodec")
    print(f"  Committed: {snap}")

    return repo


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    print("=" * 80)
    print("Direct GRIB Virtualization: gribberish + VirtualiZarr + Icechunk")
    print(f"Source: {GFS_URL}")
    print("=" * 80)

    # Step 1: Scan
    print("\n[Step 1] Scanning GRIB with gribberish...")
    var_refs = scan_grib_variables(GFS_URL, TARGET_VARS)
    for name, info in var_refs.items():
        print(f"  {name}: shape={info['shape']}, offset={info['offset']}, size={info['size']}")

    # Step 2: Build virtual dataset
    print(f"\n[Step 2] Building virtual dataset with GribberishCodec...")
    vds = build_virtual_dataset(GFS_URL, var_refs)
    print(f"  {vds}")
    print(f"  Codecs: {vds[next(iter(var_refs))].data.metadata.codecs}")

    # Step 3: Store in Icechunk
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "grib_icechunk_repo"
        print(f"\n[Step 3] Writing to Icechunk at {repo_path}...")
        repo = store_in_icechunk(vds, repo_path, "noaa-gfs-bdp-pds")

        # Read back
        print(f"\n[Read back] Opening from Icechunk...")
        ds = xr.open_zarr(repo.readonly_session(branch="main").store, consolidated=False)
        print(f"  {ds}")

        # Try reading actual data
        print(f"\n[Data read] Fetching tmp[:, :3, :3] from S3 via Icechunk...")
        try:
            vals = ds["tmp"].isel(time=0, latitude=slice(0, 3), longitude=slice(0, 3)).values
            print(f"  Temperature values (K): {vals}")
        except Exception as e:
            print(f"  Read failed (likely SSL issue in this env): {e}")

    print(f"\n{'=' * 80}")
    print("Done!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
