import json
import xml.etree.ElementTree as ET
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import cf_xarray  # noqa: F401 - needed for ds.cf accessor
import pytest
import xarray as xr

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common.dynamical_dataset import DynamicalDataset

# Downloaded from https://codes.ecmwf.int/parameter-database/api/v1/param/?format=json
ECMWF_PARAMS_PATH = Path(__file__).parent / "ecmwf_params.json"
ECMWF_PARAM_DB_URL = "https://codes.ecmwf.int/grib/param-db/"

# Downloaded from https://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml
CF_STANDARD_NAME_TABLE_PATH = Path(__file__).parent / "cf-standard-name-table.xml"


@pytest.fixture(scope="session")
def ecmwf_params() -> list[dict[str, Any]]:
    """Load the ECMWF parameter database from the local JSON file."""
    with open(ECMWF_PARAMS_PATH) as f:
        params: list[dict[str, Any]] = json.load(f)
        return params


@pytest.fixture(scope="session")
def ecmwf_shortnames(ecmwf_params: list[dict[str, Any]]) -> set[str]:
    """Get all ECMWF shortnames."""
    return {p["shortname"] for p in ecmwf_params}


@pytest.fixture(scope="session")
def ecmwf_names(ecmwf_params: list[dict[str, Any]]) -> set[str]:
    """Get all ECMWF parameter names (long names)."""
    return {p["name"] for p in ecmwf_params}


@pytest.fixture(scope="session")
def ecmwf_shortname_to_id(ecmwf_params: list[dict[str, Any]]) -> dict[str, int]:
    """Map ECMWF shortnames to their parameter IDs for linking."""
    return {p["shortname"]: p["id"] for p in ecmwf_params}


@pytest.fixture(scope="session")
def ecmwf_name_to_id(ecmwf_params: list[dict[str, Any]]) -> dict[str, int]:
    """Map ECMWF names to their parameter IDs for linking."""
    return {p["name"]: p["id"] for p in ecmwf_params}


@pytest.fixture(scope="session")
def ecmwf_shortname_to_names(ecmwf_params: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Map ECMWF shortnames to all long names that share that shortname."""
    result: dict[str, set[str]] = {}
    for p in ecmwf_params:
        result.setdefault(p["shortname"], set()).add(p["name"])
    return result


@pytest.fixture(scope="session")
def cf_standard_name_to_canonical_units() -> dict[str, str]:
    """
    Load the CF Standard Name Table from the local XML file.
    Returns a dict mapping standard_name -> canonical_units.
    """
    xml = ET.parse(CF_STANDARD_NAME_TABLE_PATH).getroot()  # noqa: S314 trusted local file
    result: dict[str, str] = {}
    for entry in xml.findall(".//entry"):
        standard_name = entry.get("id")
        canonical_units_elem = entry.find("canonical_units")
        if (
            standard_name is not None
            and canonical_units_elem is not None
            and canonical_units_elem.text is not None
        ):
            result[standard_name] = canonical_units_elem.text
    return result


@pytest.mark.parametrize(
    "dataset", DYNAMICAL_DATASETS, ids=[d.dataset_id for d in DYNAMICAL_DATASETS]
)
def test_cf_latitude_longitude_recognized(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    """
    Ensure latitude and longitude coordinates are recognized as CF coordinates.
    CF requires these to have standard_name and units attributes.
    For non-projected datasets, they should also have axis attributes.
    """
    template_config = dataset.template_config
    template_path = template_config.template_path()

    ds = xr.open_zarr(template_path)

    # Check if this is a projected coordinate system (has x, y as dimension coords)
    if "x" in ds.dims and "y" in ds.dims:
        assert "x" in ds.coords
        assert "y" in ds.coords
        is_projected = True
    elif "latitude" in ds.dims and "longitude" in ds.dims:
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords
        is_projected = False
    else:
        raise ValueError(
            f"Unknown spatial coordinate dimensions for dataset {dataset.dataset_id}. "
            f"Expected latitude/longitude or x/y as dimension coordinates."
        )

    # Check latitude is recognized
    if "latitude" in ds.coords:
        assert "latitude" in ds.cf.coordinates, (
            f"latitude coordinate not recognized by cf_xarray. "
            f"Ensure it has standard_name='latitude', units='degree_north'. "
            f"Current attrs: {dict(ds['latitude'].attrs)}"
        )
        # Verify latitude attrs are CF compliant
        lat_attrs = ds["latitude"].attrs
        assert lat_attrs.get("standard_name") == "latitude", (
            f"latitude missing standard_name='latitude', got: {lat_attrs.get('standard_name')}"
        )
        assert lat_attrs.get("units") == "degree_north", (
            f"latitude missing units='degree_north', got: {lat_attrs.get('units')}"
        )
        # Only check for axis if latitude is a dimension coordinate (not projected)
        if not is_projected:
            assert lat_attrs.get("axis") == "Y", (
                f"latitude missing axis='Y', got: {lat_attrs.get('axis')}"
            )

    # Check longitude is recognized
    if "longitude" in ds.coords:
        assert "longitude" in ds.cf.coordinates, (
            f"longitude coordinate not recognized by cf_xarray. "
            f"Ensure it has standard_name='longitude', units='degree_east'. "
            f"Current attrs: {dict(ds['longitude'].attrs)}"
        )
        # Verify longitude attrs are CF compliant
        lon_attrs = ds["longitude"].attrs
        assert lon_attrs.get("standard_name") == "longitude", (
            f"longitude missing standard_name='longitude', got: {lon_attrs.get('standard_name')}"
        )
        assert lon_attrs.get("units") == "degree_east", (
            f"longitude missing units='degree_east', got: {lon_attrs.get('units')}"
        )
        # Only check for axis if longitude is a dimension coordinate (not projected)
        if not is_projected:
            assert lon_attrs.get("axis") == "X", (
                f"longitude missing axis='X', got: {lon_attrs.get('axis')}"
            )

    # For projected datasets, check x and y have correct CF attributes
    if is_projected:
        if "x" in ds.coords:
            x_attrs = ds["x"].attrs
            assert x_attrs.get("standard_name") == "projection_x_coordinate", (
                f"x missing standard_name='projection_x_coordinate', got: {x_attrs.get('standard_name')}"
            )
            assert x_attrs.get("axis") == "X", (
                f"x missing axis='X', got: {x_attrs.get('axis')}"
            )
        if "y" in ds.coords:
            y_attrs = ds["y"].attrs
            assert y_attrs.get("standard_name") == "projection_y_coordinate", (
                f"y missing standard_name='projection_y_coordinate', got: {y_attrs.get('standard_name')}"
            )
            assert y_attrs.get("axis") == "Y", (
                f"y missing axis='Y', got: {y_attrs.get('axis')}"
            )


@pytest.mark.parametrize(
    "dataset", DYNAMICAL_DATASETS, ids=[d.dataset_id for d in DYNAMICAL_DATASETS]
)
def test_cf_time_coordinates_recognized(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    """
    Ensure time-related coordinates are recognized by cf_xarray.
    CF convention requires time coordinates to have standard_name and appropriate units.
    """
    template_config = dataset.template_config
    template_path = template_config.template_path()

    ds = xr.open_zarr(template_path)

    # Analysis dataset: just time coordinate
    if "time" in ds.coords:
        assert "init_time" not in ds.coords
        assert "lead_time" not in ds.coords

    # Forecast dataset: init_time and lead_time coordinates
    if "init_time" in ds.coords:
        assert "lead_time" in ds.coords
        assert "time" not in ds.coords

    # Check for time coordinate (used in analysis datasets)
    if "time" in ds.coords and "time" in ds.dims:
        time_attrs = ds["time"].attrs
        assert time_attrs.get("standard_name") == "time", (
            f"time coordinate missing standard_name='time', got: {time_attrs.get('standard_name')}"
        )
        assert time_attrs.get("axis") == "T", (
            f"time coordinate missing axis='T', got: {time_attrs.get('axis')}"
        )

    # Check for init_time coordinate (used in forecast datasets)
    if "init_time" in ds.coords and "init_time" in ds.dims:
        init_time_attrs = ds["init_time"].attrs
        assert init_time_attrs.get("standard_name") == "forecast_reference_time", (
            f"init_time coordinate missing standard_name='forecast_reference_time', got: {init_time_attrs.get('standard_name')}"
        )

    # Check for lead_time coordinate (used in forecast datasets)
    if "lead_time" in ds.coords and "lead_time" in ds.dims:
        lead_time_attrs = ds["lead_time"].attrs
        assert lead_time_attrs.get("standard_name") == "forecast_period", (
            f"lead_time coordinate missing standard_name='forecast_period', got: {lead_time_attrs.get('standard_name')}"
        )

    # Check for valid_time coordinate
    if "valid_time" in ds.coords:
        valid_time_attrs = ds["valid_time"].attrs
        assert valid_time_attrs.get("standard_name") == "time", (
            f"valid_time coordinate missing standard_name='time', got: {valid_time_attrs.get('standard_name')}"
        )


@pytest.mark.parametrize(
    "dataset", DYNAMICAL_DATASETS, ids=[d.dataset_id for d in DYNAMICAL_DATASETS]
)
def test_cf_ensemble_member_recognized(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    """
    Ensure ensemble_member coordinate is CF compliant where present.
    """
    template_config = dataset.template_config
    template_path = template_config.template_path()

    ds = xr.open_zarr(template_path)

    if "ensemble_member" in ds.coords:
        ens_attrs = ds["ensemble_member"].attrs
        assert ens_attrs.get("standard_name") == "realization", (
            f"ensemble_member missing standard_name='realization', got: {ens_attrs.get('standard_name')}"
        )


@pytest.mark.parametrize(
    "dataset", DYNAMICAL_DATASETS, ids=[d.dataset_id for d in DYNAMICAL_DATASETS]
)
def test_cf_coordinates_have_long_name(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    """
    Ensure all coordinates have a long_name attribute as recommended by CF.
    """
    template_config = dataset.template_config
    template_path = template_config.template_path()

    ds = xr.open_zarr(template_path)

    for coord_name in ds.coords:
        # spatial_ref is a special CRS coordinate and doesn't need long_name
        if coord_name == "spatial_ref":
            continue
        coord_attrs = ds[coord_name].attrs
        assert "long_name" in coord_attrs, (
            f"Coordinate '{coord_name}' missing 'long_name' attribute. "
            f"CF conventions recommend all variables have a long_name. "
            f"Current attrs: {dict(coord_attrs)}"
        )


@pytest.mark.parametrize(
    "dataset", DYNAMICAL_DATASETS, ids=[d.dataset_id for d in DYNAMICAL_DATASETS]
)
def test_cf_data_variables_have_long_name(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    """
    Ensure all data variables have a long_name attribute as required by CF.
    """
    template_config = dataset.template_config
    template_path = template_config.template_path()

    ds = xr.open_zarr(template_path)

    for var_name in ds.data_vars:
        var_attrs = ds[var_name].attrs
        assert "long_name" in var_attrs, (
            f"Data variable '{var_name}' missing 'long_name' attribute. "
            f"CF conventions require all data variables have a long_name. "
            f"Current attrs: {dict(var_attrs)}"
        )
        assert isinstance(var_attrs["long_name"], str), (
            f"Data variable '{var_name}' has 'long_name' that is not a string. "
            f"Current value: {var_attrs['long_name']!r}"
        )
        assert var_attrs["long_name"] != "", (
            f"Data variable '{var_name}' has an empty 'long_name' attribute."
        )


# --- CF standard_name and units validation ---

# Variable names for which CF Conventions does NOT define a standard name.
ALLOWED_MISSING_STANDARD_NAME: set[str] = {
    "percent_frozen_precipitation_surface",
    "categorical_snow_surface",
    "categorical_ice_pellets_surface",
    "categorical_freezing_rain_surface",
    "categorical_rain_surface",
    "categorical_precipitation_type_surface",
    "composite_reflectivity",
    "soil_water_runoff",
    "qa",
    # snowfall_surface is a snow depth rate (m s-1); CF has no standard name for this quantity
    "snowfall_surface",
}

# (standard_name, units) pairs that are intentionally non-canonical but allowed for all datasets.
CF_UNITS_VARIANCES_ALLOWLIST: set[tuple[str, str]] = {
    ("air_temperature", "degree_Celsius"),
    ("dew_point_temperature", "degree_Celsius"),
    ("cloud_area_fraction", "percent"),
    ("cloud_area_fraction_in_atmosphere_layer", "percent"),
    ("relative_humidity", "percent"),
}

# (standard_name, units, dataset_id) for dataset-specific unit variances.
CF_UNITS_VARIANCES_DATASET_ALLOWLIST: set[tuple[str, str, str]] = {
    # U Arizona SWANN uses mm for snow variables to match source data conventions
    ("surface_snow_thickness", "mm", "u-arizona-swann-analysis"),
    ("lwe_thickness_of_surface_snow_amount", "mm", "u-arizona-swann-analysis"),
}


@pytest.mark.parametrize(
    "dataset", DYNAMICAL_DATASETS, ids=[d.dataset_id for d in DYNAMICAL_DATASETS]
)
def test_cf_standard_name_and_units(
    dataset: DynamicalDataset[Any, Any],
    cf_standard_name_to_canonical_units: dict[str, str],
) -> None:
    """
    For each data variable:
    1. standard_name must be set (or the variable must be in ALLOWED_MISSING_STANDARD_NAME)
    2. If standard_name is set, it must exist in the CF Standard Name Table
    3. If standard_name is set, units must match CF canonical units (with allowlist)
    4. standard_name must be properly written to and recognized in the zarr template
    """
    template_config = dataset.template_config
    template_path = template_config.template_path()

    ds = xr.open_zarr(template_path)
    recognized_standard_names = ds.cf.standard_names

    errors: list[str] = []

    for var_config in template_config.data_vars:
        standard_name = var_config.attrs.standard_name
        units = var_config.attrs.units

        if standard_name is None:
            if var_config.name not in ALLOWED_MISSING_STANDARD_NAME:
                errors.append(
                    f"Variable '{var_config.name}' does not have a standard_name defined, "
                    f"but is not in ALLOWED_MISSING_STANDARD_NAME. "
                    f"If CF Conventions defines a standard name for this variable, add it to the DataVarAttrs. "
                    f"If CF Conventions does NOT define a standard name, "
                    f"add '{var_config.name}' to ALLOWED_MISSING_STANDARD_NAME."
                )
            continue

        # standard_name is set — validate it
        if standard_name not in cf_standard_name_to_canonical_units:
            errors.append(
                f"Variable '{var_config.name}' has standard_name='{standard_name}', "
                f"which is not in the official CF Standard Name Table. "
                f"See https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html"
            )
            continue

        # Check units match CF canonical units
        canonical_units = cf_standard_name_to_canonical_units[standard_name]
        if (
            units != canonical_units
            and (standard_name, units) not in CF_UNITS_VARIANCES_ALLOWLIST
            and (standard_name, units, dataset.dataset_id)
            not in CF_UNITS_VARIANCES_DATASET_ALLOWLIST
        ):
            errors.append(
                f"Variable '{var_config.name}' has standard_name='{standard_name}' with units='{units}', "
                f"but expected canonical units '{canonical_units}'. "
                f"If this is an intentional exception, add to CF_UNITS_VARIANCES_ALLOWLIST or CF_UNITS_VARIANCES_DATASET_ALLOWLIST."
            )

        # Check standard_name is properly recognized in the zarr template
        if standard_name not in recognized_standard_names:
            errors.append(
                f"Variable '{var_config.name}' has standard_name='{standard_name}' configured "
                f"but cf_xarray did not recognize it in the zarr template. "
                f"Run 'uv run main {dataset.dataset_id} update-template' to regenerate the template."
            )
        elif var_config.name not in recognized_standard_names[standard_name]:
            errors.append(
                f"Variable '{var_config.name}' has standard_name='{standard_name}' configured "
                f"but is not recognized by cf_xarray in the zarr template. "
                f"Found: {recognized_standard_names[standard_name]}. "
                f"Run 'uv run main {dataset.dataset_id} update-template' to regenerate the template."
            )

    # Also check coordinate standard_names and units are valid CF
    for var_name in ds.coords:
        standard_name = ds[var_name].attrs.get("standard_name")
        units = ds[var_name].attrs.get("units")
        if standard_name is None or units is None:
            continue

        if standard_name not in cf_standard_name_to_canonical_units:
            errors.append(
                f"Coordinate '{var_name}' has standard_name='{standard_name}', "
                f"which is not in the official CF Standard Name Table."
            )

    assert not errors, (
        f"CF standard_name/units errors in dataset '{dataset.dataset_id}':\n\n"
        + "\n\n".join(errors)
    )


# --- ECMWF parameter validation ---

# Variables that don't have ECMWF parameter database entries and are exempt from ECMWF validation.
ECMWF_SHORTNAME_EXEMPT: set[str] = {
    # NOAA NDVI CDR variables
    "ndvi_raw",
    "ndvi_usable",
    "qa",
    # NASA SMAP soil moisture variables
    "soil_moisture_am",
    "soil_moisture_pm",
    # DWD ICON-specific variables
    "aswdifd_s",
    "aswdir_s",
    # HRRR 80m wind (no ECMWF equivalent)
    "80u",
    "80v",
}

ECMWF_LONGNAME_EXEMPT: set[str] = {
    # Non-meteorological or dataset-specific variables
    "Normalized Difference Vegetation Index (usable)",
    "quality_assurance",
    "Soil Moisture (AM)",
    "Soil Moisture (PM)",
    # DWD ICON-specific variables
    "Downward diffusive short wave radiation flux at surface",
    "Downward direct short wave radiation flux at surface",
    # HRRR 80m wind (no ECMWF equivalent)
    "80 metre U wind component",
    "80 metre V wind component",
}


def _format_ecmwf_suggestions(
    value: str,
    all_values: set[str],
    value_to_id: dict[str, int],
    n: int = 5,
) -> str:
    """Format suggestions for ECMWF parameter values with links."""
    matches = get_close_matches(value, list(all_values), n=n, cutoff=0.4)
    if not matches:
        return f"No close matches found. See {ECMWF_PARAM_DB_URL}"

    suggestions = []
    for match in matches:
        param_id = value_to_id.get(match)
        if param_id:
            suggestions.append(f"  - '{match}' ({ECMWF_PARAM_DB_URL}?id={param_id})")
        else:
            suggestions.append(f"  - '{match}'")
    return "Possible matches:\n" + "\n".join(suggestions)


@pytest.mark.parametrize(
    "dataset", DYNAMICAL_DATASETS, ids=[d.dataset_id for d in DYNAMICAL_DATASETS]
)
def test_ecmwf_parameter_compliance(
    dataset: DynamicalDataset[Any, Any],
    ecmwf_shortnames: set[str],
    ecmwf_names: set[str],
    ecmwf_shortname_to_id: dict[str, int],
    ecmwf_name_to_id: dict[str, int],
    ecmwf_shortname_to_names: dict[str, set[str]],
) -> None:
    """
    For each data variable, validate ECMWF parameter compliance:
    1. short_name must be in the ECMWF parameter database (or exempt)
    2. long_name must be in the ECMWF parameter database (or exempt)
    3. If both are in the ECMWF database, they must refer to the same parameter
    """
    template_config = dataset.template_config
    errors: list[str] = []

    for var_config in template_config.data_vars:
        short_name = var_config.attrs.short_name
        long_name = var_config.attrs.long_name
        short_exempt = short_name in ECMWF_SHORTNAME_EXEMPT
        long_exempt = long_name in ECMWF_LONGNAME_EXEMPT

        # Check short_name
        if not short_exempt and short_name not in ecmwf_shortnames:
            suggestions = _format_ecmwf_suggestions(
                short_name, ecmwf_shortnames, ecmwf_shortname_to_id
            )
            errors.append(
                f"Variable '{var_config.name}' has short_name='{short_name}' "
                f"which is not in the ECMWF parameter database.\n{suggestions}"
            )

        # Check long_name
        if not long_exempt and long_name not in ecmwf_names:
            suggestions = _format_ecmwf_suggestions(
                long_name, ecmwf_names, ecmwf_name_to_id
            )
            errors.append(
                f"Variable '{var_config.name}' has long_name='{long_name}' "
                f"which is not in the ECMWF parameter database.\n{suggestions}"
            )

        # Check short_name and long_name refer to the same ECMWF parameter
        if (
            not short_exempt
            and not long_exempt
            and short_name in ecmwf_shortnames
            and long_name in ecmwf_names
        ):
            names_for_shortname = ecmwf_shortname_to_names[short_name]
            if long_name not in names_for_shortname:
                param_id = ecmwf_shortname_to_id[short_name]
                errors.append(
                    f"Variable '{var_config.name}' has short_name='{short_name}' "
                    f"(ECMWF long_names for this shortname: {sorted(names_for_shortname)}) "
                    f"but long_name='{long_name}'. They must refer to the same ECMWF parameter. "
                    f"See {ECMWF_PARAM_DB_URL}?id={param_id}"
                )

    assert not errors, (
        f"ECMWF parameter errors in dataset '{dataset.dataset_id}':\n\n"
        + "\n\n".join(errors)
    )


# --- Cross-dataset metadata consistency ---

# Dataset-specific exceptions for cross-dataset consistency checks.
# Format: (variable_or_coord_name, attribute_name, dataset_id)
# These are intentional exceptions where source data conventions differ.
CROSS_DATASET_CONSISTENCY_EXCEPTIONS: set[tuple[str, str, str]] = {
    # U Arizona SWANN uses mm for snow variables to match source data conventions,
    # while other datasets use CF-compliant meters.
    # Excepted by var_name, short_name, and long_name since all three groupings detect the conflict.
    ("snow_depth", "units", "u-arizona-swann-analysis"),
    ("snow_water_equivalent", "units", "u-arizona-swann-analysis"),
    ("sd", "units", "u-arizona-swann-analysis"),
    ("sde", "units", "u-arizona-swann-analysis"),
    ("Snow depth", "units", "u-arizona-swann-analysis"),
    ("Snow depth water equivalent", "units", "u-arizona-swann-analysis"),
}


def _format_conflict(
    var_name: str, attr_name: str, values_to_datasets: dict[str | None, list[str]]
) -> str:
    """Format a conflict message showing which datasets use which values."""
    values_list = list(values_to_datasets.keys())
    values_str = " vs ".join(f"'{v}'" for v in values_list)
    header = f"{var_name} {attr_name} conflict: {values_str}"

    lines = [header]
    for value, dataset_ids in values_to_datasets.items():
        lines.append(f"  '{value}': {', '.join(dataset_ids)}")

    return "\n".join(lines)


def _check_consistency(
    metadata_by_name: dict[str, dict[str, dict[str, str | None]]],
    attrs_to_check: list[str],
) -> list[str]:
    """
    Check that for each name, the given attributes are consistent across datasets.
    metadata_by_name: {name: {dataset_id: {attr: value}}}
    Returns list of conflict messages.
    """
    conflicts: list[str] = []
    for name, datasets_metadata in metadata_by_name.items():
        if len(datasets_metadata) < 2:
            continue

        for attr_name in attrs_to_check:
            values_to_datasets: dict[str | None, list[str]] = {}
            for dataset_id, metadata in datasets_metadata.items():
                value = metadata.get(attr_name)
                values_to_datasets.setdefault(value, []).append(dataset_id)

            if len(values_to_datasets) <= 1:
                continue

            # Filter out allowed exceptions
            filtered_values: dict[str | None, list[str]] = {}
            for value, dataset_ids in values_to_datasets.items():
                remaining = [
                    ds_id
                    for ds_id in dataset_ids
                    if (name, attr_name, ds_id)
                    not in CROSS_DATASET_CONSISTENCY_EXCEPTIONS
                ]
                if remaining:
                    filtered_values[value] = remaining

            if len(filtered_values) > 1:
                conflicts.append(_format_conflict(name, attr_name, filtered_values))

    return conflicts


def test_metadata_consistency_across_datasets() -> None:
    """
    Ensure metadata is consistent across all datasets. Checks:

    1. Same var_name → same (short_name, long_name, standard_name, units)
    2. Same short_name → same (long_name, standard_name, units)
    3. Same long_name → same (short_name, standard_name, units)
    4. Same coord_name → same (long_name, standard_name, units)
    """
    # Collect data variable metadata grouped by var_name, short_name, and long_name
    by_var_name: dict[str, dict[str, dict[str, str | None]]] = {}
    by_short_name: dict[str, dict[str, dict[str, str | None]]] = {}
    by_long_name: dict[str, dict[str, dict[str, str | None]]] = {}

    for dataset in DYNAMICAL_DATASETS:
        template_config = dataset.template_config
        for var_config in template_config.data_vars:
            attrs = {
                "short_name": var_config.attrs.short_name,
                "units": var_config.attrs.units,
                "standard_name": var_config.attrs.standard_name,
                "long_name": var_config.attrs.long_name,
            }

            by_var_name.setdefault(var_config.name, {})[dataset.dataset_id] = attrs
            by_short_name.setdefault(var_config.attrs.short_name, {})[
                dataset.dataset_id
            ] = attrs
            by_long_name.setdefault(var_config.attrs.long_name, {})[
                dataset.dataset_id
            ] = attrs

    conflicts: list[str] = []

    # Same var_name → consistent metadata
    conflicts.extend(
        _check_consistency(
            by_var_name, ["short_name", "long_name", "standard_name", "units"]
        )
    )

    # Same short_name → consistent long_name, standard_name, units
    conflicts.extend(
        _check_consistency(by_short_name, ["long_name", "standard_name", "units"])
    )

    # Same long_name → consistent short_name, standard_name, units
    conflicts.extend(
        _check_consistency(by_long_name, ["short_name", "standard_name", "units"])
    )

    # Collect coordinate metadata and check consistency
    by_coord_name: dict[str, dict[str, dict[str, str | None]]] = {}

    for dataset in DYNAMICAL_DATASETS:
        template_config = dataset.template_config
        for coord_config in template_config.coords:
            by_coord_name.setdefault(coord_config.name, {})[dataset.dataset_id] = {
                "long_name": coord_config.attrs.long_name,
                "standard_name": coord_config.attrs.standard_name,
                "units": coord_config.attrs.units,
            }

    conflicts.extend(
        _check_consistency(by_coord_name, ["long_name", "standard_name", "units"])
    )

    assert not conflicts, (
        "Metadata inconsistencies found across datasets:\n\n" + "\n\n".join(conflicts)
    )
