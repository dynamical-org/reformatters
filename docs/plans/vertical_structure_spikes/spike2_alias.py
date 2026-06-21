"""Spike 2: the 'virtual alias' idea. Can the SAME source-file byte range be
referenced from TWO array paths in one icechunk store (a back-compat alias at
root + a canonical location in a group)? And what does it cost in the manifest?
"""

import tempfile
from pathlib import Path

import icechunk
import numpy as np
import xarray as xr
import zarr
from icechunk import VirtualChunkSpec

tmp = tempfile.mkdtemp()
src_dir = Path(tempfile.mkdtemp())

# Fake "source file": one chunk's worth of float32 bytes (like one decoded GRIB msg).
payload = np.arange(4 * 5, dtype="float32").reshape(4, 5)
src = src_dir / "source.bin"
src.write_bytes(payload.tobytes())
location = f"file://{src}"

# Register the local dir as a virtual chunk container.
config = icechunk.RepositoryConfig.default()
container = icechunk.VirtualChunkContainer(
    url_prefix=f"file://{src_dir}/",
    store=icechunk.local_filesystem_store(str(src_dir)),
)
config.set_virtual_chunk_container(container)
creds = icechunk.containers_credentials({f"file://{src_dir}/": None})

repo = icechunk.Repository.create(
    icechunk.local_filesystem_storage(tmp),
    config=config,
    authorize_virtual_chunk_access=creds,
)
session = repo.writable_session("main")
store = session.store

# Build empty arrays (no real chunks) at root and in a group, same shape/dtype.
# No compressor/serializer: virtual refs are raw bytes, decoded as-is (real
# datasets swap in a GribberishCodec serializer; uncompressed is enough here).
enc = {"temperature_2m": {"compressors": None, "chunks": (4, 5)}}
root = xr.Dataset(
    {"temperature_2m": (("lat", "lon"), np.zeros((4, 5), "float32"))},
    coords={"lat": np.arange(4), "lon": np.arange(5)},
)
root.to_zarr(store, mode="w", compute=False, consolidated=False, encoding=enc)
canon = xr.Dataset(
    {"temperature_2m": (("lat", "lon"), np.zeros((4, 5), "float32"))},
    coords={"lat": np.arange(4), "lon": np.arange(5)},
)
canon.to_zarr(
    store,
    group="single_level",
    mode="a",
    compute=False,
    consolidated=False,
    encoding=enc,
)

spec = VirtualChunkSpec(
    index=[0, 0], location=location, offset=0, length=payload.nbytes
)
# Same spec -> two different array paths.
assert (
    store.set_virtual_refs("temperature_2m", [spec], validate_containers=True) is None
)
assert (
    store.set_virtual_refs(
        "single_level/temperature_2m", [spec], validate_containers=True
    )
    is None
)
snap = session.commit("alias same ref at root and in group")
print("committed snapshot:", snap)

ro = repo.readonly_session("main").store
a = xr.open_zarr(ro, consolidated=False)["temperature_2m"].values
b = xr.open_zarr(ro, group="single_level", consolidated=False)["temperature_2m"].values
print("root values == source:", np.array_equal(a, payload))
print("group values == source:", np.array_equal(b, payload))
print("root == group (same bytes, one source):", np.array_equal(a, b))

# Manifest cost: each alias is a separate manifest entry even though bytes are shared.
g = zarr.open_group(ro, mode="r")
print(
    "\nNote: 2 array paths -> 2 chunk-ref manifest entries pointing at the same "
    "(location, offset, length). No byte duplication in source, but the ref count "
    "(and manifest size) doubles for aliased variables."
)
