import warnings

import xarray as xr


def open_grib_with_multiple_coordinates(file_path: str, data_type: str) -> xr.Dataset:
    """
    Open a GRIB file that has variables with conflicting coordinate values.

    This function handles the common issue where cfgrib fails to load variables
    due to coordinate conflicts (e.g., different step times or heightAboveGround values).

    The solution is to:
    1. Load subsets of the GRIB file using specific filter_by_keys combinations
    2. Rename conflicting coordinates to be more descriptive
    3. Merge all datasets together

    Args:
        file_path: Path to the GRIB file
        data_type: GRIB data type filter (e.g., "cf", "pf")

    Returns:
        Merged xarray Dataset containing all variables with resolved coordinate conflicts
    """
    datasets = []

    # Define coordinate filters to handle different variable groups
    coordinate_filters = [
        # Base dataset - variables without coordinate conflicts
        {"dataType": data_type},
        # Variables with step=3 (like mx2t3, mn2t3)
        {"dataType": data_type, "step": 3},
        # Variables at 10m height (like u10, v10, fg10)
        {"dataType": data_type, "typeOfLevel": "heightAboveGround", "level": 10},
        # Variables at 2m height with step=0 (like t2m, d2m)
        {
            "dataType": data_type,
            "typeOfLevel": "heightAboveGround",
            "level": 2,
            "step": 0,
        },
        # Variables at 2m height with step=3 (some 2m variables might have step=3)
        {
            "dataType": data_type,
            "typeOfLevel": "heightAboveGround",
            "level": 2,
            "step": 3,
        },
        # Variables at 100m height (like u100, v100)
        {"dataType": data_type, "typeOfLevel": "heightAboveGround", "level": 100},
    ]

    for i, filter_keys in enumerate(coordinate_filters):  # noqa: B007
        try:
            ds = xr.open_dataset(file_path, engine="cfgrib", filter_by_keys=filter_keys)

            # Rename coordinates to avoid conflicts when merging
            if "step" in ds.coords and ds.coords["step"].values == 3.0:  # noqa: PLR2004
                ds = ds.rename({"step": "step_3h"})

            if "heightAboveGround" in ds.coords:
                height_val = ds.coords["heightAboveGround"].values
                if height_val == 2.0:  # noqa: PLR2004
                    ds = ds.rename({"heightAboveGround": "height_2m"})
                elif height_val == 10.0:  # noqa: PLR2004
                    ds = ds.rename({"heightAboveGround": "height_10m"})
                elif height_val == 100.0:  # noqa: PLR2004
                    ds = ds.rename({"heightAboveGround": "height_100m"})

            if ds.data_vars:  # Only add if it has variables
                datasets.append(ds)

        except Exception as e:
            # Skip failed filters
            print(e)  # noqa: T201
            continue

    if not datasets:
        raise ValueError("No datasets could be loaded")

    # Merge all datasets
    merged_ds = xr.merge(datasets, compat="override")

    return merged_ds


# Usage:
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="skipping variable")

    dsc = open_grib_with_multiple_coordinates(
        "data/20250301000000-0h-enfo-ef.grib2", "cf"
    )
    dsp = open_grib_with_multiple_coordinates(
        "data/20250301000000-0h-enfo-ef.grib2", "pf"
    )
    vars_to_long_names = {dv: dsp[dv].attrs["long_name"] for dv in dsp.data_vars}

"""
{'ro': 'Runoff',
 'str': 'Surface net long-wave (thermal) radiation',
 'q': 'Specific humidity',
 'u100': '100 metre U wind component',
 'vsw': 'Volumetric soil moisture',
 'tp': 'Total precipitation',
 'tprate': 'Total precipitation rate',
 'sot': 'Soil temperature',
 'ttr': 'Top net long-wave (thermal) radiation',
 'v100': '100 metre V wind component',
 'lsm': 'Land-sea mask',
 'asn': 'Snow albedo',
 'ewss': 'Time-integrated eastward turbulent surface stress',
 'msl': 'Mean sea level pressure',
 'vo': 'Vorticity (relative)',
 'sp': 'Surface pressure',
 'gh': 'Geopotential height',
 'nsss': 'Time-integrated northward turbulent surface stress',
 'skt': 'Skin temperature',
 'sithick': 'Sea ice thickness',
 'sve': 'Eastward surface sea water velocity',
 'v': 'V component of wind',
 'u': 'U component of wind',
 'w': 'Vertical velocity',
 'tcw': 'Total column water',
 'mucape': 'Most-unstable CAPE',
 'svn': 'Northward surface sea water velocity',
 'd': 'Divergence',
 'strd': 'Surface long-wave (thermal) radiation downwards',
 't': 'Temperature',
 'ssr': 'Surface net short-wave (solar) radiation',
 'tcwv': 'Total column vertically-integrated water vapour',
 'ssrd': 'Surface short-wave (solar) radiation downwards',
 'ptype': 'Precipitation type',
 'r': 'Relative humidity',
 'zos': 'Sea surface height'}
"""
