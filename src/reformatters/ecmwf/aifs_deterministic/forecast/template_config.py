from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import computed_field

from reformatters.common.config_models import (
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.template_config import (
    SPATIAL_REF_COORDS,
    TemplateConfig,
)
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar, EcmwfInternalAttrs


class EcmwfAifsForecastTemplateConfig(TemplateConfig[EcmwfDataVar]):
    dims: tuple[Dim, ...] = ("init_time", "lead_time", "latitude", "longitude")
    append_dim: AppendDim = "init_time"
    # Start from 2024-04-01 when PL u/v are available and the grid is regular 0.25 degree.
    # Earlier data (2024-02-29 to 2024-03-13) uses a reduced Gaussian grid.
    append_dim_start: Timestamp = pd.Timestamp("2024-04-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="ecmwf-aifs-deterministic-forecast-15-day-0-25-degree",
            dataset_version="0.1.0",
            name="ECMWF AIFS Deterministic Forecast, 15 day, 0.25 degree",
            description="Weather forecasts from the ECMWF Artificial Intelligence Forecasting System (AIFS) deterministic model.",
            attribution="ECMWF AIFS deterministic forecast data processed by dynamical.org from ECMWF Open Data.",
            spatial_domain="Global",
            spatial_resolution="0.25 degrees (~25km)",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:.0f} hours",
            forecast_domain="Forecast lead time 0-360 hours (0-15 days) ahead",
            forecast_resolution="Forecast step 6 hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            self.append_dim: self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "lead_time": pd.timedelta_range("0h", "360h", freq="6h"),
            "latitude": np.flip(np.arange(-90, 90.25, 0.25)),
            "longitude": np.arange(-180, 180, 0.25),
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        return {
            "valid_time": ds["init_time"] + ds["lead_time"],
            "ingested_forecast_length": (
                (self.append_dim,),
                np.full(ds[self.append_dim].size, np.timedelta64("NaT", "ns")),
            ),
            "expected_forecast_length": (
                (self.append_dim,),
                np.full(
                    ds[self.append_dim].size,
                    self.dimension_coordinates()["lead_time"].max(),
                    dtype="timedelta64[ns]",
                ),
            ),
            "spatial_ref": SPATIAL_REF_COORDS,
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        dim_coords = self.dimension_coordinates()
        append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()

        return [
            Coordinate(
                name=self.append_dim,
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    calendar="proleptic_gregorian",
                    units="seconds since 1970-01-01 00:00:00",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Forecast initialization time",
                    standard_name="forecast_reference_time",
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=dim_coords[self.append_dim].min().isoformat(), max="Present"
                    ),
                ),
            ),
            Coordinate(
                name="lead_time",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=-1,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=len(dim_coords["lead_time"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Forecast lead time",
                    standard_name="forecast_period",
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].min()),
                        max=str(dim_coords["lead_time"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="latitude",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["latitude"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Latitude",
                    standard_name="latitude",
                    units="degree_north",
                    axis="Y",
                    statistics_approximate=StatisticsApproximate(
                        min=float(dim_coords["latitude"].min()),
                        max=float(dim_coords["latitude"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="longitude",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["longitude"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Longitude",
                    standard_name="longitude",
                    units="degree_east",
                    axis="X",
                    statistics_approximate=StatisticsApproximate(
                        min=float(dim_coords["longitude"].min()),
                        max=float(dim_coords["longitude"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="valid_time",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    calendar="proleptic_gregorian",
                    units="seconds since 1970-01-01 00:00:00",
                    chunks=(
                        append_dim_coordinate_chunk_size,
                        len(dim_coords["lead_time"]),
                    ),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Valid time",
                    standard_name="time",
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=self.append_dim_start.isoformat(),
                        max="Present",
                    ),
                ),
            ),
            Coordinate(
                name="ingested_forecast_length",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=-1,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Ingested forecast length",
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].min()),
                        max=str(dim_coords["lead_time"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="expected_forecast_length",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=-1,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Expected forecast length",
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].min()),
                        max=str(dim_coords["lead_time"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="spatial_ref",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    chunks=(),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units=None,
                    statistics_approximate=None,
                    crs_wkt='GEOGCS["unknown",DATUM["unknown",SPHEROID["unknown",6371229,0]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Longitude",EAST],AXIS["Latitude",NORTH]]',
                    semi_major_axis=6371229.0,
                    semi_minor_axis=6371229.0,
                    inverse_flattening=0.0,
                    reference_ellipsoid_name="unknown",
                    longitude_of_prime_meridian=0.0,
                    prime_meridian_name="Greenwich",
                    geographic_crs_name="unknown",
                    horizontal_datum_name="unknown",
                    grid_mapping_name="latitude_longitude",
                    spatial_ref='GEOGCS["unknown",DATUM["unknown",SPHEROID["unknown",6371229,0]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Longitude",EAST],AXIS["Latitude",NORTH]]',
                    comment="This coordinate reference system matches the source data which follows WMO conventions of assuming the earth is a perfect sphere with a radius of 6,371,229m. It is similar to EPSG:4326, but EPSG:4326 uses a more accurate representation of the earth's shape.",
                ),
            ),
        ]

    @computed_field
    @property
    def data_vars(self) -> Sequence[EcmwfDataVar]:
        # ~4MB uncompressed, ~0.8MB compressed
        var_chunks: dict[Dim, int] = {
            "init_time": 4,  # 1 day of 6-hourly data
            "lead_time": 61,  # All lead times
            "latitude": 64,  # 12 chunks over 721 pixels
            "longitude": 64,  # 23 chunks over 1440 pixels
        }

        # ~961MB uncompressed, ~192MB compressed
        var_shards: dict[Dim, int] = {
            "init_time": var_chunks["init_time"] * 7,  # 7 days per shard
            "lead_time": var_chunks["lead_time"],
            "latitude": var_chunks["latitude"] * 6,  # 2 shards over 721 pixels
            "longitude": var_chunks["longitude"] * 6,  # 4 shards over 1440 pixels
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(var_chunks[d] for d in self.dims),
            shards=tuple(var_shards[d] for d in self.dims),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        default_keep_mantissa_bits = 7

        # Surface variables available from 2025-02-26 are marked with date_available.
        # All variables listed here are available from 2024-04-01 (append_dim_start) unless noted.
        expanded_vars_date = pd.Timestamp("2025-02-26T00:00")

        return [
            EcmwfDataVar(
                name="temperature_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="2t",
                    long_name="2 metre temperature",
                    units="degree_Celsius",
                    step_type="instant",
                    standard_name="air_temperature",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="Temperature [C]",
                    grib_description='2[m] HTGL="Specified height level above ground"',
                    grib_element="TMP",
                    grib_index_param="2t",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcmwfDataVar(
                name="dew_point_temperature_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="2d",
                    long_name="2 metre dewpoint temperature",
                    units="degree_Celsius",
                    step_type="instant",
                    standard_name="dew_point_temperature",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="Dew point temperature [C]",
                    grib_description='2[m] HTGL="Specified height level above ground"',
                    grib_element="DPT",
                    grib_index_param="2d",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcmwfDataVar(
                name="wind_u_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="10u",
                    long_name="10 metre U wind component",
                    units="m s-1",
                    step_type="instant",
                    standard_name="eastward_wind",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="u-component of wind [m/s]",
                    grib_description='10[m] HTGL="Specified height level above ground"',
                    grib_element="UGRD",
                    grib_index_param="10u",
                    keep_mantissa_bits=6,
                ),
            ),
            EcmwfDataVar(
                name="wind_v_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="10v",
                    long_name="10 metre V wind component",
                    units="m s-1",
                    step_type="instant",
                    standard_name="northward_wind",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="v-component of wind [m/s]",
                    grib_description='10[m] HTGL="Specified height level above ground"',
                    grib_element="VGRD",
                    grib_index_param="10v",
                    keep_mantissa_bits=6,
                ),
            ),
            EcmwfDataVar(
                name="wind_u_100m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="100u",
                    long_name="100 metre U wind component",
                    units="m s-1",
                    step_type="instant",
                    standard_name="eastward_wind",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="u-component of wind [m/s]",
                    grib_description='100[m] HTGL="Specified height level above ground"',
                    grib_element="UGRD",
                    grib_index_param="100u",
                    keep_mantissa_bits=6,
                    date_available=expanded_vars_date,
                ),
            ),
            EcmwfDataVar(
                name="wind_v_100m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="100v",
                    long_name="100 metre V wind component",
                    units="m s-1",
                    step_type="instant",
                    standard_name="northward_wind",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="v-component of wind [m/s]",
                    grib_description='100[m] HTGL="Specified height level above ground"',
                    grib_element="VGRD",
                    grib_index_param="100v",
                    keep_mantissa_bits=6,
                    date_available=expanded_vars_date,
                ),
            ),
            EcmwfDataVar(
                name="precipitation_rate_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="prate",
                    long_name="Total precipitation rate",
                    units="kg m-2 s-1",
                    step_type="instant",
                    standard_name="precipitation_flux",
                    comment="Instantaneous precipitation rate. Units equivalent to mm/s.",
                ),
                # Early GRIB master tables (v27) encode tp with generic product template codes.
                # Later versions (v34+) use "Total precipitation rate [kg/(m^2*s)]".
                # We use the early form here; read_data handles both.
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="(prodType 0, cat 1, subcat 193) [-]",
                    grib_description='2[-] SFC="Ground or water surface"',
                    grib_element="unknown",
                    grib_index_param="tp",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcmwfDataVar(
                name="convective_precipitation_rate_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="cprate",
                    long_name="Convective precipitation rate",
                    units="kg m-2 s-1",
                    step_type="instant",
                    standard_name="convective_precipitation_flux",
                    comment="Instantaneous convective precipitation rate. Units equivalent to mm/s.",
                ),
                # Early GRIB master tables (v27) encode cp with generic product template codes.
                # Later versions (v34+) use "Convective precipitation rate [kg/(m^2*s)]".
                # We use the early form here; read_data handles both.
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="(prodType 0, cat 1, subcat 195) [-]",
                    grib_description='2[-] SFC="Ground or water surface"',
                    grib_element="unknown",
                    grib_index_param="cp",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcmwfDataVar(
                name="downward_short_wave_radiation_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sdswrf",
                    long_name="Surface downward short-wave radiation flux",
                    units="W m-2",
                    step_type="instant",
                    standard_name="surface_downwelling_shortwave_flux_in_air",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="Downward short-wave radiation flux [W/(m^2)]",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_element="DSWRF",
                    grib_index_param="ssrd",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    date_available=expanded_vars_date,
                ),
            ),
            EcmwfDataVar(
                name="downward_long_wave_radiation_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sdlwrf",
                    long_name="Surface downward long-wave radiation flux",
                    units="W m-2",
                    step_type="instant",
                    standard_name="surface_downwelling_longwave_flux_in_air",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="Downward long-wave radiation flux [W/(m^2)]",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_element="DLWRF",
                    grib_index_param="strd",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    date_available=expanded_vars_date,
                ),
            ),
            EcmwfDataVar(
                name="pressure_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sp",
                    long_name="Surface pressure",
                    units="Pa",
                    step_type="instant",
                    standard_name="surface_air_pressure",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="Pressure [Pa]",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_element="PRES",
                    grib_index_param="sp",
                    keep_mantissa_bits=11,
                ),
            ),
            EcmwfDataVar(
                name="pressure_reduced_to_mean_sea_level",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="prmsl",
                    long_name="Pressure reduced to MSL",
                    units="Pa",
                    step_type="instant",
                    standard_name="air_pressure_at_mean_sea_level",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="Pressure [Pa]",
                    grib_description='0[-] MSL="Mean sea level"',
                    grib_element="PRES",
                    grib_index_param="msl",
                    keep_mantissa_bits=11,
                ),
            ),
            EcmwfDataVar(
                name="total_cloud_cover_atmosphere",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="tcc",
                    long_name="Total Cloud Cover",
                    units="percent",
                    step_type="instant",
                    standard_name="cloud_area_fraction",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="Total cloud cover [%]",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_element="TCDC",
                    grib_index_param="tcc",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    date_available=expanded_vars_date,
                ),
            ),
            EcmwfDataVar(
                name="precipitable_water_atmosphere",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="tcw",
                    long_name="Total column water",
                    units="kg m-2",
                    step_type="instant",
                    standard_name="atmosphere_mass_content_of_water_vapor",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="Total column water [kg/m^2]",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_element="TCWAT",
                    grib_index_param="tcw",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcmwfDataVar(
                name="geopotential_500hpa",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="z",
                    long_name="Geopotential",
                    units="m2 s-2",
                    step_type="instant",
                    standard_name="geopotential",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="Geopotential [(m^2)/(s^2)]",
                    grib_description='50000[Pa] ISBL="Isobaric surface"',
                    grib_element="GP",
                    grib_index_param="z",
                    grib_index_level_type="pl",
                    grib_index_level_value=500,
                    keep_mantissa_bits=11,
                ),
            ),
        ]
