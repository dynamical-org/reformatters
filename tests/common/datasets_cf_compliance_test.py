import json
import xml.etree.ElementTree as ET
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import cf_xarray  # noqa: F401 - needed for ds.cf accessor
import pytest
import requests
import xarray as xr

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common.dynamical_dataset import DynamicalDataset

# Downloaded from https://codes.ecmwf.int/parameter-database/api/v1/param/?format=json
ECMWF_PARAMS_PATH = Path(__file__).parent / "ecmwf_params.json"
ECMWF_PARAM_DB_URL = "https://codes.ecmwf.int/grib/param-db/"


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


@pytest.fixture(scope="session")  # session scope downloads once per test run
def cf_standard_name_to_canonical_units() -> dict[str, str]:
    """
    Download the latest CF Standard Name Table.
    Returns a dict mapping standard_name -> canonical_units.
    """
    url = "https://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    xml = ET.fromstring(response.content)  # noqa: S314 cfconventions.org is fairly safe and this is test not prod
    return {
        standard_name: entry.find("canonical_units").text  # type: ignore[union-attr,misc]
        for entry in xml.findall(".//entry")
        if (standard_name := entry.get("id")) is not None
    }


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
def test_cf_data_variables_have_standard_names_where_applicable(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    """
    Verify that:
    1. All data variables have standard_name where CF Conventions defines one
    2. Variables with standard_name in config are properly recognized in the zarr template

    Variables in the allowlist are exempt because CF Conventions does not define
    standard names for them.
    """
    # Variables for which CF Conventions does NOT define a standard name
    allowed_missing_standard_name = {
        "percent_frozen_precipitation_surface",
        "categorical_snow_surface",
        "categorical_ice_pellets_surface",
        "categorical_freezing_rain_surface",
        "categorical_rain_surface",
        "categorical_precipitation_type_surface",
        "composite_reflectivity",
        "soil_water_runoff",
        "qa",
    }

    template_config = dataset.template_config
    template_path = template_config.template_path()

    ds = xr.open_zarr(template_path)

    # Get all standard names from cf_xarray
    recognized_standard_names = ds.cf.standard_names

    # Check each data variable
    for var_config in template_config.data_vars:
        if var_config.attrs.standard_name is not None:
            # Variable has a standard_name - verify it's properly written to zarr
            expected_std_name = var_config.attrs.standard_name
            assert expected_std_name in recognized_standard_names, (
                f"Variable '{var_config.name}' has standard_name='{expected_std_name}' configured "
                f"but cf_xarray did not recognize it in the zarr template. "
                f"Run 'uv run main {dataset.dataset_id} update-template' to regenerate the template."
            )
            assert var_config.name in recognized_standard_names[expected_std_name], (
                f"Variable '{var_config.name}' has standard_name='{expected_std_name}' configured "
                f"but is not recognized by cf_xarray in the zarr template. Found: {recognized_standard_names[expected_std_name]}. "
                f"Run 'uv run main {dataset.dataset_id} update-template' to regenerate the template."
            )
        else:
            # Variable does not have a standard_name - it must be in the allowlist
            assert var_config.name in allowed_missing_standard_name, (
                f"Variable '{var_config.name}' does not have a standard_name defined, "
                f"but is not in the allowed_missing_standard_name list. "
                f"If CF Conventions defines a standard name for this variable, add it to the DataVarAttrs. "
                f"If CF Conventions does NOT define a standard name for this variable, "
                f"add '{var_config.name}' to allowed_missing_standard_name in this test."
            )


CF_UNITS_VARIANCES_ALLOWLIST: set[tuple[str, str]] = {
    ("air_temperature", "degree_Celsius"),
    ("dew_point_temperature", "degree_Celsius"),
    ("cloud_area_fraction", "percent"),
    ("cloud_area_fraction_in_atmosphere_layer", "percent"),
    ("relative_humidity", "percent"),
    ("surface_snow_thickness", "mm"),
    ("lwe_thickness_of_surface_snow_amount", "mm"),
}


@pytest.mark.parametrize(
    "dataset", DYNAMICAL_DATASETS, ids=[d.dataset_id for d in DYNAMICAL_DATASETS]
)
def test_cf_standard_names_and_units(
    dataset: DynamicalDataset[Any, Any],
    cf_standard_name_to_canonical_units: dict[str, str],
) -> None:
    """
    Ensure standard_name values are valid and units match canonical units from the CF Standard Name Table.
    """
    template_path = dataset.template_config.template_path()
    ds = xr.open_zarr(template_path)

    for var_name in ds.variables:  # coordinates and data variables
        standard_name = ds[var_name].attrs.get("standard_name")
        units = ds[var_name].attrs.get("units")
        if standard_name is None:
            # test_cf_data_variables_have_standard_names_where_applicable validates
            continue

        if units is None and var_name in ds.coords:  # not all coords need units
            continue

        assert standard_name in cf_standard_name_to_canonical_units, (
            f"Variable '{var_name}' has standard_name='{standard_name}', which is not in the official CF Standard Name Table. "
            f"See https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html"
        )

        canonical_units = cf_standard_name_to_canonical_units[standard_name]

        if (standard_name, units) in CF_UNITS_VARIANCES_ALLOWLIST:
            continue

        assert units == canonical_units, (
            f"{ds.attrs['dataset_id']}: Variable '{var_name}' has standard_name='{standard_name}' with units='{units}', "
            f"but expected canonical units '{canonical_units}'. "
            f"If this is an intentional exception, add ('{standard_name}', '{units}') to CF_UNITS_VARIANCES_ALLOWLIST."
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


# Datasets that are allowed to have different metadata for specific variables.
# Format: (variable_name, attribute_name, dataset_id)
# These are intentional exceptions where source data conventions differ.
CROSS_DATASET_CONSISTENCY_EXCEPTIONS: set[tuple[str, str, str]] = {
    # U Arizona SWANN uses mm for snow variables to match source data conventions,
    # while other datasets use CF-compliant meters.
    ("snow_depth", "units", "u-arizona-swann-analysis"),
    ("snow_water_equivalent", "units", "u-arizona-swann-analysis"),
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


def test_cf_data_variable_metadata_consistency_across_datasets() -> None:
    """
    Ensure that data variables with the same name have consistent metadata
    (short_name, units, standard_name, long_name) across all datasets.

    This test helps ensure that users can trust that variables with the same name
    represent the same physical quantity with consistent metadata across datasets.
    """
    # Collect metadata for all data variables across all datasets
    # Structure: {var_name: {dataset_id: {short_name, units, standard_name, long_name}}}
    var_metadata: dict[str, dict[str, dict[str, str | None]]] = {}

    for dataset in DYNAMICAL_DATASETS:
        template_config = dataset.template_config
        for var_config in template_config.data_vars:
            var_name = var_config.name
            if var_name not in var_metadata:
                var_metadata[var_name] = {}

            var_metadata[var_name][dataset.dataset_id] = {
                "short_name": var_config.attrs.short_name,
                "units": var_config.attrs.units,
                "standard_name": var_config.attrs.standard_name,
                "long_name": var_config.attrs.long_name,
            }

    # Check for inconsistencies
    conflicts: list[str] = []

    for var_name, datasets_metadata in var_metadata.items():
        # Skip variables that only appear in one dataset
        if len(datasets_metadata) < 2:
            continue

        for attr_name in ["short_name", "units", "standard_name", "long_name"]:
            # Group datasets by their value for this attribute
            values_to_datasets: dict[str | None, list[str]] = {}
            for dataset_id, metadata in datasets_metadata.items():
                value = metadata[attr_name]
                values_to_datasets.setdefault(value, []).append(dataset_id)

            # Check for conflicts (more than one unique value)
            if len(values_to_datasets) > 1:
                # Filter out allowed exceptions
                filtered_values: dict[str | None, list[str]] = {}
                for value, dataset_ids in values_to_datasets.items():
                    remaining_datasets = [
                        ds_id
                        for ds_id in dataset_ids
                        if (var_name, attr_name, ds_id)
                        not in CROSS_DATASET_CONSISTENCY_EXCEPTIONS
                    ]
                    if remaining_datasets:
                        filtered_values[value] = remaining_datasets

                # If after filtering we still have multiple values, it's a conflict
                if len(filtered_values) > 1:
                    conflicts.append(
                        _format_conflict(var_name, attr_name, filtered_values)
                    )

    assert not conflicts, (
        "Data variable metadata inconsistencies found across datasets:\n\n"
        + "\n\n".join(conflicts)
    )


def test_cf_coordinate_metadata_consistency_across_datasets() -> None:
    """
    Ensure that coordinates with the same name have consistent metadata
    (long_name, standard_name, units) across all datasets.

    This test helps ensure that users can trust that coordinates with the same name
    represent the same physical quantity with consistent metadata across datasets.

    Note: Coordinates do not have short_name attributes.
    """
    # Collect metadata for all coordinates across all datasets
    # Structure: {coord_name: {dataset_id: {long_name, standard_name, units}}}
    coord_metadata: dict[str, dict[str, dict[str, str | None]]] = {}

    for dataset in DYNAMICAL_DATASETS:
        template_config = dataset.template_config
        for coord_config in template_config.coords:
            coord_name = coord_config.name
            if coord_name not in coord_metadata:
                coord_metadata[coord_name] = {}

            coord_metadata[coord_name][dataset.dataset_id] = {
                "long_name": coord_config.attrs.long_name,
                "standard_name": coord_config.attrs.standard_name,
                "units": coord_config.attrs.units,
            }

    # Check for inconsistencies
    conflicts: list[str] = []

    for coord_name, datasets_metadata in coord_metadata.items():
        # Skip coordinates that only appear in one dataset
        if len(datasets_metadata) < 2:
            continue

        for attr_name in ["long_name", "standard_name", "units"]:
            # Group datasets by their value for this attribute
            values_to_datasets: dict[str | None, list[str]] = {}
            for dataset_id, metadata in datasets_metadata.items():
                value = metadata[attr_name]
                values_to_datasets.setdefault(value, []).append(dataset_id)

            # Check for conflicts (more than one unique value)
            if len(values_to_datasets) > 1:
                # Filter out allowed exceptions (using same exception mechanism)
                filtered_values: dict[str | None, list[str]] = {}
                for value, dataset_ids in values_to_datasets.items():
                    remaining_datasets = [
                        ds_id
                        for ds_id in dataset_ids
                        if (coord_name, attr_name, ds_id)
                        not in CROSS_DATASET_CONSISTENCY_EXCEPTIONS
                    ]
                    if remaining_datasets:
                        filtered_values[value] = remaining_datasets

                # If after filtering we still have multiple values, it's a conflict
                if len(filtered_values) > 1:
                    conflicts.append(
                        _format_conflict(coord_name, attr_name, filtered_values)
                    )

    assert not conflicts, (
        "Coordinate metadata inconsistencies found across datasets:\n\n"
        + "\n\n".join(conflicts)
    )


# Variables that don't have ECMWF parameter database entries and are exempt from validation.
# These are either non-meteorological variables or dataset-specific variables.
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
    "normalized_difference_vegetation_index",
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
def test_ecmwf_data_variable_shortnames(
    dataset: DynamicalDataset[Any, Any],
    ecmwf_shortnames: set[str],
    ecmwf_shortname_to_id: dict[str, int],
) -> None:
    """
    Ensure data variable short_name values match ECMWF parameter conventions.
    ECMWF short_name is the 'shortname' field in the ECMWF parameter database.
    """
    template_config = dataset.template_config

    invalid_shortnames: list[str] = []

    for var_config in template_config.data_vars:
        short_name = var_config.attrs.short_name
        if short_name in ECMWF_SHORTNAME_EXEMPT:
            continue
        if short_name not in ecmwf_shortnames:
            suggestions = _format_ecmwf_suggestions(
                short_name, ecmwf_shortnames, ecmwf_shortname_to_id
            )
            invalid_shortnames.append(
                f"Variable '{var_config.name}' has short_name='{short_name}' "
                f"which is not in the ECMWF parameter database.\n{suggestions}"
            )

    assert not invalid_shortnames, (
        f"Data variables with invalid ECMWF short_name in dataset '{dataset.dataset_id}':\n\n"
        + "\n\n".join(invalid_shortnames)
    )


@pytest.mark.parametrize(
    "dataset", DYNAMICAL_DATASETS, ids=[d.dataset_id for d in DYNAMICAL_DATASETS]
)
def test_ecmwf_data_variable_longnames(
    dataset: DynamicalDataset[Any, Any],
    ecmwf_names: set[str],
    ecmwf_name_to_id: dict[str, int],
) -> None:
    """
    Ensure data variable long_name values match ECMWF parameter conventions.
    ECMWF long_name is the 'name' field in the ECMWF parameter database.
    """
    template_config = dataset.template_config

    invalid_longnames: list[str] = []

    for var_config in template_config.data_vars:
        long_name = var_config.attrs.long_name
        if long_name in ECMWF_LONGNAME_EXEMPT:
            continue
        if long_name not in ecmwf_names:
            suggestions = _format_ecmwf_suggestions(
                long_name, ecmwf_names, ecmwf_name_to_id
            )
            invalid_longnames.append(
                f"Variable '{var_config.name}' has long_name='{long_name}' "
                f"which is not in the ECMWF parameter database.\n{suggestions}"
            )

    assert not invalid_longnames, (
        f"Data variables with invalid ECMWF long_name in dataset '{dataset.dataset_id}':\n\n"
        + "\n\n".join(invalid_longnames)
    )
