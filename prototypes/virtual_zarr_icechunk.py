"""
Prototype: Virtual Zarr + Icechunk with GFS-like weather data.

Demonstrates:
1. Creating virtual zarr references from NetCDF4 files using VirtualiZarr
2. Storing virtual references in Icechunk
3. Modifying data variable attributes and names
4. Adding additional coordinate arrays (valid_time)
5. Incremental append of new data with version history + time travel

Creates realistic GFS 0.25° NetCDF4 files as source data (structure matches
real NOAA GFS output), then virtualizes them. In production these would be
real source files on S3/GCS/local disk.

Run: uv run python prototypes/virtual_zarr_icechunk.py
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import virtualizarr as vz
from virtualizarr.parsers import HDFParser
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

import icechunk

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATE = "20260313"
INIT_HOURS = ["00", "06"]
LEAD_HOURS = [0, 3, 6]

# GFS 0.25° grid (subsample to 1° for speed)
LATS = np.arange(90, -90.1, -1.0, dtype=np.float32)
LONS = np.arange(0, 360, 1.0, dtype=np.float32)

DATA_VARS = {
    "t2m": {"long_name": "2 metre temperature", "units": "K", "fill": 288.0, "noise": 15.0},
    "d2m": {"long_name": "2 metre dewpoint temperature", "units": "K", "fill": 278.0, "noise": 10.0},
    "u10": {"long_name": "10 metre U wind component", "units": "m s-1", "fill": 0.0, "noise": 8.0},
    "v10": {"long_name": "10 metre V wind component", "units": "m s-1", "fill": 0.0, "noise": 8.0},
    "sp": {"long_name": "Surface pressure", "units": "Pa", "fill": 101325.0, "noise": 3000.0},
    "prmsl": {"long_name": "Pressure reduced to MSL", "units": "Pa", "fill": 101325.0, "noise": 2000.0},
    "gust": {"long_name": "Wind speed (gust)", "units": "m s-1", "fill": 5.0, "noise": 10.0},
}

RENAME_MAP = {
    "t2m": "temperature_2m",
    "d2m": "dewpoint_temperature_2m",
    "u10": "wind_u_10m",
    "v10": "wind_v_10m",
    "sp": "surface_pressure",
    "prmsl": "mean_sea_level_pressure",
    "gust": "wind_gust",
}

CF_ATTRS: dict[str, dict[str, str]] = {
    "temperature_2m": {"standard_name": "air_temperature"},
    "dewpoint_temperature_2m": {"standard_name": "dew_point_temperature"},
    "wind_u_10m": {"standard_name": "eastward_wind"},
    "wind_v_10m": {"standard_name": "northward_wind"},
    "surface_pressure": {"standard_name": "surface_air_pressure"},
    "mean_sea_level_pressure": {"standard_name": "air_pressure_at_mean_sea_level"},
    "wind_gust": {"standard_name": "wind_speed_of_gust"},
}


# ============================================================================
# STEP 0: Generate realistic GFS-like NetCDF4 source files
# ============================================================================
def step0_generate_source_files(tmpdir: Path) -> dict[str, list[Path]]:
    """Create GFS-like NetCDF4 files that mimic real NOAA GFS GRIB output."""
    print("=" * 80)
    print("STEP 0: Generate GFS-like NetCDF4 source files")
    print("=" * 80)
    print("  (In production these would be real GFS GRIB or NetCDF4 files on S3)")

    rng = np.random.default_rng(42)
    src_dir = tmpdir / "source_files"
    src_dir.mkdir()

    nc_files: dict[str, list[Path]] = {}

    for init_hour in INIT_HOURS:
        init_time = pd.Timestamp(f"{DATE}T{init_hour}:00:00")
        paths: list[Path] = []

        for lead_h in LEAD_HOURS:
            step = pd.Timedelta(hours=lead_h)
            valid_time = init_time + step

            data_vars_xr: dict[str, xr.Variable] = {}
            for var_name, var_cfg in DATA_VARS.items():
                data = (var_cfg["fill"] + var_cfg["noise"] * rng.standard_normal((len(LATS), len(LONS)))).astype(np.float32)
                data_vars_xr[var_name] = xr.Variable(
                    ["latitude", "longitude"],
                    data,
                    attrs={"long_name": var_cfg["long_name"], "units": var_cfg["units"]},
                )

            ds = xr.Dataset(
                data_vars_xr,
                coords={
                    "latitude": ("latitude", LATS, {"units": "degrees_north", "standard_name": "latitude"}),
                    "longitude": ("longitude", LONS, {"units": "degrees_east", "standard_name": "longitude"}),
                    "time": init_time,
                    "step": step,
                    "valid_time": valid_time,
                },
                attrs={
                    "Conventions": "CF-1.7",
                    "institution": "US National Weather Service - NCEP",
                    "source": "GFS 0.25 degree (prototype synthetic data)",
                    "history": f"Synthetic GFS data for virtualization prototype, init={init_time}",
                },
            )

            nc_path = src_dir / f"gfs_{DATE}_{init_hour}z_f{lead_h:03d}.nc"
            ds.to_netcdf(nc_path)
            paths.append(nc_path)
            print(f"  Created: {nc_path.name} ({nc_path.stat().st_size / 1024:.0f} KB)")

        nc_files[init_hour] = paths

    print(f"\n  Total: {sum(len(v) for v in nc_files.values())} files in {src_dir}")
    return nc_files


# ============================================================================
# STEP 1: VirtualiZarr — create virtual references
# ============================================================================
def step1_virtualize(nc_files: dict[str, list[Path]]) -> dict[str, xr.Dataset]:
    """Create virtual xarray Datasets backed by references to NetCDF4 chunks."""
    print("\n" + "=" * 80)
    print("STEP 1: Create virtual zarr references with VirtualiZarr")
    print("=" * 80)

    registry = ObjectStoreRegistry()
    registry.register("file:///", LocalStore(prefix="/"))

    virtual_datasets: dict[str, xr.Dataset] = {}

    for init_hour, paths in nc_files.items():
        print(f"\n  --- init_hour={init_hour}z ---")

        per_lead: list[xr.Dataset] = []
        for path in paths:
            vds = vz.open_virtual_dataset(
                f"file://{path}",
                registry=registry,
                parser=HDFParser(),
                loadable_variables=["time", "step", "latitude", "longitude", "valid_time"],
            )
            per_lead.append(vds)
            print(f"    {path.name}: vars={list(vds.data_vars)}, dims={dict(vds.dims)}")

        # Concatenate along step (lead time) dimension
        combined = xr.concat(per_lead, dim="step")
        print(f"\n    Combined dataset:")
        print(f"    {combined}")
        virtual_datasets[init_hour] = combined

    return virtual_datasets


# ============================================================================
# STEP 2: Store virtual dataset in Icechunk
# ============================================================================
def step2_icechunk_store(
    vds: xr.Dataset,
    repo: icechunk.Repository,
) -> str:
    """Write virtual references to an Icechunk repository."""
    print("\n" + "=" * 80)
    print("STEP 2: Store virtual references in Icechunk")
    print("=" * 80)

    session = repo.writable_session("main")
    vds.virtualize.to_icechunk(session.store)
    snap_id = session.commit("Initial virtual dataset: GFS 00z")
    print(f"  Committed snapshot: {snap_id}")

    # Read back as a regular xarray dataset
    ds = xr.open_zarr(repo.readonly_session(branch="main").store, consolidated=False)
    print(f"\n  Dataset read from Icechunk:")
    print(f"  {ds}")
    print(f"\n  Time: {ds.coords['time'].values}")
    print(f"  Lead time: {ds.coords['lead_time'].values}")

    # Read actual data to prove virtual references work
    print(f"\n  Reading a data slice (temperature_2m[0, :3, :3])...")
    vals = ds["temperature_2m"].isel(lead_time=0, latitude=slice(0, 3), longitude=slice(0, 3)).values
    print(f"  Values:\n  {vals}")

    return snap_id


# ============================================================================
# STEP 3: Modify variable attributes and names
# ============================================================================
def step3_modify_attrs(vds: xr.Dataset) -> xr.Dataset:
    """Rename variables and update CF-compliant attributes."""
    print("\n" + "=" * 80)
    print("STEP 3: Modify variable names and attributes")
    print("=" * 80)

    # Rename
    rename_existing = {old: new for old, new in RENAME_MAP.items() if old in vds.data_vars}
    vds = vds.rename(rename_existing)
    print(f"  Renamed: {rename_existing}")

    # Add CF standard_name attributes
    for var_name, attrs in CF_ATTRS.items():
        if var_name in vds.data_vars:
            vds[var_name].attrs.update(attrs)

    print(f"  Variables after rename: {list(vds.data_vars)}")
    for var in list(vds.data_vars)[:3]:
        print(f"    {var}: {dict(vds[var].attrs)}")

    return vds


# ============================================================================
# STEP 4: Add additional coordinate arrays
# ============================================================================
def step4_add_coordinates(vds: xr.Dataset, init_hour: str) -> xr.Dataset:
    """Add init_time and valid_time coordinates, rename step → lead_time."""
    print("\n" + "=" * 80)
    print("STEP 4: Add additional coordinate arrays (init_time, valid_time)")
    print("=" * 80)

    init_time = pd.Timestamp(f"{DATE}T{init_hour}:00:00")
    vds = vds.assign_coords(init_time=init_time)
    print(f"  Added init_time = {init_time}")

    # Compute valid_time = init_time + step
    if "step" in vds.coords:
        step_values = vds.coords["step"].values
        if np.ndim(step_values) == 0:
            step_values = np.array([step_values.item()])
        valid_times = np.array([init_time + pd.Timedelta(s) for s in step_values])
        vds = vds.assign_coords(valid_time=("step", valid_times))
        print(f"  Added valid_time = {valid_times}")

    # Rename step → lead_time
    if "step" in vds.dims:
        vds = vds.rename({"step": "lead_time"})
        print("  Renamed 'step' → 'lead_time'")

    print(f"\n  Coordinates: {list(vds.coords)}")
    print(f"  Dataset:\n  {vds}")

    return vds


# ============================================================================
# STEP 5: Incremental append + version history
# ============================================================================
def step5_append(
    vds_append: xr.Dataset,
    repo: icechunk.Repository,
    snap_before: str,
) -> None:
    """Append new data, show version history, and demonstrate time travel."""
    print("\n" + "=" * 80)
    print("STEP 5: Incremental append + version history")
    print("=" * 80)

    # Determine append dimension
    append_dim = "lead_time" if "lead_time" in vds_append.dims else "step"
    print(f"  Appending along: '{append_dim}'")
    print(f"  New data shape: {dict(vds_append.dims)}")

    session = repo.writable_session("main")
    vds_append.virtualize.to_icechunk(session.store, append_dim=append_dim)
    snap_after = session.commit("Append GFS 06z data")
    print(f"  Committed: {snap_after}")

    # Read appended dataset
    ds = xr.open_zarr(repo.readonly_session(branch="main").store, consolidated=False)
    print(f"\n  Dataset after append:")
    print(f"  {ds}")
    print(f"\n  Lead time values: {ds.coords['lead_time'].values}")

    # Version history
    print(f"\n  === Version history ===")
    for snap in repo.ancestry(branch="main"):
        print(f"    {snap.id}: \"{snap.message}\" ({snap.written_at})")

    # Time travel
    print(f"\n  === Time travel: reading pre-append version ===")
    ds_old = xr.open_zarr(
        repo.readonly_session(snapshot_id=snap_before).store,
        consolidated=False,
    )
    print(f"  Pre-append shape: {dict(ds_old.dims)}")
    print(f"  Post-append shape: {dict(ds.dims)}")


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    print("=" * 80)
    print("Virtual Zarr + Icechunk Prototype")
    print(f"GFS-like data | Init hours: {INIT_HOURS} | Lead times: {LEAD_HOURS}h")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        # Step 0: Generate source files
        nc_files = step0_generate_source_files(tmpdir)

        # Step 1: Virtualize
        virtual_datasets = step1_virtualize(nc_files)

        # Steps 3-4 on first init time
        vds0 = virtual_datasets[INIT_HOURS[0]]
        vds0 = step3_modify_attrs(vds0)
        vds0 = step4_add_coordinates(vds0, INIT_HOURS[0])

        # Create Icechunk repo
        repo_path = tmpdir / "icechunk_repo"
        storage = icechunk.local_filesystem_storage(str(repo_path))
        vcc_config = {
            "file:///tmp/": icechunk.VirtualChunkContainer(
                url_prefix="file:///tmp/",
                store=icechunk.ObjectStoreConfig.LocalFileSystem("/tmp/"),
            ),
        }
        vcc_credentials = icechunk.containers_credentials({"file:///tmp/": None})
        repo = icechunk.Repository.create(
            storage=storage,
            config=icechunk.RepositoryConfig(virtual_chunk_containers=vcc_config),
            authorize_virtual_chunk_access=vcc_credentials,
        )

        # Step 2: Store in Icechunk
        snap_id = step2_icechunk_store(vds0, repo)

        # Steps 3-4 on second init time
        vds1 = virtual_datasets[INIT_HOURS[1]]
        vds1 = step3_modify_attrs(vds1)
        vds1 = step4_add_coordinates(vds1, INIT_HOURS[1])

        # Step 5: Append
        step5_append(vds1, repo, snap_id)

    print(f"\n{'=' * 80}")
    print("Prototype complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
