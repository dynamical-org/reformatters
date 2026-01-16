import xml.etree.ElementTree as ET
from typing import Any

import cf_xarray  # noqa: F401 - needed for ds.cf accessor
import pytest
import requests
import xarray as xr

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common.dynamical_dataset import DynamicalDataset


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
    is_projected = "x" in ds.dims and "y" in ds.dims

    # Check latitude is recognized
    if "latitude" in ds.coords:
        assert "latitude" in ds.cf.coordinates, (
            f"latitude coordinate not recognized by cf_xarray. "
            f"Ensure it has standard_name='latitude', units='degrees_north'. "
            f"Current attrs: {dict(ds['latitude'].attrs)}"
        )
        # Verify latitude attrs are CF compliant
        lat_attrs = ds["latitude"].attrs
        assert lat_attrs.get("standard_name") == "latitude", (
            f"latitude missing standard_name='latitude', got: {lat_attrs.get('standard_name')}"
        )
        assert lat_attrs.get("units") == "degrees_north", (
            f"latitude missing units='degrees_north', got: {lat_attrs.get('units')}"
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
            f"Ensure it has standard_name='longitude', units='degrees_east'. "
            f"Current attrs: {dict(ds['longitude'].attrs)}"
        )
        # Verify longitude attrs are CF compliant
        lon_attrs = ds["longitude"].attrs
        assert lon_attrs.get("standard_name") == "longitude", (
            f"longitude missing standard_name='longitude', got: {lon_attrs.get('standard_name')}"
        )
        assert lon_attrs.get("units") == "degrees_east", (
            f"longitude missing units='degrees_east', got: {lon_attrs.get('units')}"
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
            f"Variable '{var_name}' has standard_name='{standard_name}' with units='{units}', "
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
