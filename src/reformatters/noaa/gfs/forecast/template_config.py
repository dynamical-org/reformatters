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
from reformatters.noaa.models import NoaaDataVar, NoaaInternalAttrs


class NoaaGfsForecastTemplateConfig(TemplateConfig[NoaaDataVar]):
    dims: tuple[Dim, ...] = ("init_time", "lead_time", "latitude", "longitude")
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = pd.Timestamp("2021-05-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="noaa-gfs-forecast",
            dataset_version="0.2.5",
            name="NOAA GFS forecast",
            description="Weather forecasts from the Global Forecast System (GFS) operated by NOAA NWS NCEP.",
            attribution="NOAA NWS NCEP GFS data processed by dynamical.org from NOAA Open Data Dissemination archives.",
            spatial_domain="Global",
            spatial_resolution="0.25 degrees (~20km)",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:.0f} hours",
            forecast_domain="Forecast lead time 0-384 hours (0-16 days) ahead",
            forecast_resolution="Forecast step 0-120 hours: hourly, 123-384 hours: 3 hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        """
        Returns a dictionary of dimension names to coordinates for the dataset.
        """
        return {
            self.append_dim: self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "lead_time": (
                pd.timedelta_range("0h", "120h", freq="1h").union(
                    pd.timedelta_range("123h", "384h", freq="3h")
                )
            ),
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
                    ds["lead_time"].max(),
                    dtype="timedelta64[ns]",
                ),
            ),
            "spatial_ref": SPATIAL_REF_COORDS,
        }

    @computed_field  # type: ignore[prop-decorator]
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
                    units="degrees_north",
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
                    units="degrees_east",
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
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=self.append_dim_start.isoformat(),
                        max="Present + 16 days",
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
                    chunks=(),  # Scalar coordinate
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units=None,
                    statistics_approximate=None,
                    # Deterived by running `ds.rio.write_crs("+proj=longlat +a=6371229 +b=6371229 +no_defs +type=crs")["spatial_ref"].attrs
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[NoaaDataVar]:
        var_chunks: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 105,
            "latitude": 121,
            "longitude": 121,
        }
        var_shards: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 105 * 2,
            "latitude": 121 * 6,
            "longitude": 121 * 6,
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(var_chunks[d] for d in self.dims),
            shards=tuple(var_shards[d] for d in self.dims),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        default_window_reset_frequency = pd.Timedelta("6h")
        default_keep_mantissa_bits = 7

        return [
            NoaaDataVar(
                name="pressure_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sp",
                    long_name="Surface pressure",
                    units="Pa",
                    step_type="instant",
                    standard_name="surface_air_pressure",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="PRES",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=560,
                    keep_mantissa_bits=10,
                ),
            ),
            NoaaDataVar(
                name="temperature_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="t2m",
                    long_name="2 metre temperature",
                    units="C",
                    step_type="instant",
                    standard_name="air_temperature",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="TMP",
                    grib_description='2[m] HTGL="Specified height level above ground"',
                    grib_index_level="2 m above ground",
                    index_position=580,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            NoaaDataVar(
                name="relative_humidity_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="r2",
                    long_name="2 metre relative humidity",
                    units="%",
                    step_type="instant",
                    standard_name="relative_humidity",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="RH",
                    grib_description='2[m] HTGL="Specified height level above ground"',
                    grib_index_level="2 m above ground",
                    index_position=583,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            NoaaDataVar(
                name="maximum_temperature_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="tmax",
                    long_name="Maximum temperature",
                    units="C",
                    step_type="max",
                    comment="Maximum over the previous 1-6 hours, reset every 6-hour forecast step (00Z, 06Z, 12Z, 18Z).",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="TMAX",
                    grib_description='2[m] HTGL="Specified height level above ground"',
                    grib_index_level="2 m above ground",
                    index_position=585,
                    window_reset_frequency=default_window_reset_frequency,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            NoaaDataVar(
                name="minimum_temperature_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="tmin",
                    long_name="Minimum temperature",
                    units="C",
                    step_type="min",
                    comment="Minimum over the previous 1-6 hours, reset every 6-hour forecast step (00Z, 06Z, 12Z, 18Z).",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="TMIN",
                    grib_description='2[m] HTGL="Specified height level above ground"',
                    grib_index_level="2 m above ground",
                    index_position=586,
                    window_reset_frequency=default_window_reset_frequency,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            NoaaDataVar(
                name="wind_u_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="u10",
                    long_name="10 metre U wind component",
                    units="m/s",
                    step_type="instant",
                    standard_name="eastward_wind",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="UGRD",
                    grib_description='10[m] HTGL="Specified height level above ground"',
                    grib_index_level="10 m above ground",
                    index_position=587,
                    keep_mantissa_bits=6,
                ),
            ),
            NoaaDataVar(
                name="wind_v_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="v10",
                    long_name="10 metre V wind component",
                    units="m/s",
                    step_type="instant",
                    standard_name="northward_wind",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="VGRD",
                    grib_description='10[m] HTGL="Specified height level above ground"',
                    grib_index_level="10 m above ground",
                    index_position=588,
                    keep_mantissa_bits=6,
                ),
            ),
            NoaaDataVar(
                name="wind_u_100m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="u100",
                    long_name="100 metre U wind component",
                    standard_name="eastward_wind",
                    units="m/s",
                    step_type="instant",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="UGRD",
                    grib_description='100[m] HTGL="Specified height level above ground"',
                    grib_index_level="100 m above ground",
                    index_position=688,
                    keep_mantissa_bits=6,
                ),
            ),
            NoaaDataVar(
                name="wind_v_100m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="v100",
                    long_name="100 metre V wind component",
                    units="m/s",
                    step_type="instant",
                    standard_name="northward_wind",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="VGRD",
                    grib_index_level="100 m above ground",
                    grib_description='100[m] HTGL="Specified height level above ground"',
                    index_position=689,
                    keep_mantissa_bits=6,
                ),
            ),
            NoaaDataVar(
                name="percent_frozen_precipitation_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="cpofp",
                    long_name="Percent frozen precipitation",
                    units="%",
                    step_type="instant",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="CPOFP",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=590,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            NoaaDataVar(
                name="precipitation_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="tp",
                    long_name="Total Precipitation",
                    units="mm/s",
                    comment="Average precipitation rate since the previous forecast step.",
                    step_type="avg",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="APCP",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=595,
                    deaccumulate_to_rate=True,
                    window_reset_frequency=default_window_reset_frequency,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            NoaaDataVar(
                name="categorical_snow_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="csnow",
                    long_name="Categorical snow",
                    units="0=no; 1=yes",
                    step_type="avg",
                    comment="Presence/absence over the previous 1-6 hours, reset every 6-hour forecast step (00Z, 06Z, 12Z, 18Z).",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="CSNOW",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=604,
                    window_reset_frequency=default_window_reset_frequency,
                    keep_mantissa_bits="no-rounding",
                ),
            ),
            NoaaDataVar(
                name="categorical_ice_pellets_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="cicep",
                    long_name="Categorical ice pellets",
                    units="0=no; 1=yes",
                    step_type="avg",
                    comment="Presence/absence over the previous 1-6 hours, reset every 6-hour forecast step (00Z, 06Z, 12Z, 18Z).",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="CICEP",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=605,
                    window_reset_frequency=default_window_reset_frequency,
                    keep_mantissa_bits="no-rounding",
                ),
            ),
            NoaaDataVar(
                name="categorical_freezing_rain_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="cfrzr",
                    long_name="Categorical freezing rain",
                    units="0=no; 1=yes",
                    step_type="avg",
                    comment="Presence/absence over the previous 1-6 hours, reset every 6-hour forecast step (00Z, 06Z, 12Z, 18Z).",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="CFRZR",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=606,
                    window_reset_frequency=default_window_reset_frequency,
                    keep_mantissa_bits="no-rounding",
                ),
            ),
            NoaaDataVar(
                name="categorical_rain_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="crain",
                    long_name="Categorical rain",
                    units="0=no; 1=yes",
                    step_type="avg",
                    comment="Presence/absence over the previous 1-6 hours, reset every 6-hour forecast step (00Z, 06Z, 12Z, 18Z).",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="CRAIN",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=607,
                    window_reset_frequency=default_window_reset_frequency,
                    keep_mantissa_bits="no-rounding",
                ),
            ),
            NoaaDataVar(
                name="precipitable_water_atmosphere",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="pwat",
                    long_name="Precipitable water",
                    units="kg/(m^2)",
                    step_type="instant",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="PWAT",
                    grib_description='0[-] EATM="Entire atmosphere (considered as a single layer)"',
                    grib_index_level="entire atmosphere (considered as a single layer)",
                    index_position=625,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            NoaaDataVar(
                name="total_cloud_cover_atmosphere",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="tcc",
                    long_name="Total Cloud Cover",
                    units="%",
                    step_type="avg",
                    comment="Average over the previous 1-6 hours, reset every 6-hour forecast step (00Z, 06Z, 12Z, 18Z).",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="TCDC",
                    grib_description='0[-] EATM="Entire Atmosphere"',
                    grib_index_level="entire atmosphere",
                    index_position=635,
                    window_reset_frequency=default_window_reset_frequency,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            NoaaDataVar(
                name="geopotential_height_cloud_ceiling",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="gh",
                    long_name="Geopotential height",
                    units="gpm",
                    step_type="instant",
                    standard_name="geopotential_height",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="HGT",
                    grib_description='0[-] CEIL="Cloud ceiling"',
                    grib_index_level="cloud ceiling",
                    index_position=637,
                    keep_mantissa_bits=8,
                ),
            ),
            NoaaDataVar(
                name="downward_short_wave_radiation_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sdswrf",
                    long_name="Surface downward short-wave radiation flux",
                    units="W/(m^2)",
                    step_type="avg",
                    comment="Average over the previous 1-6 hours, reset every 6-hour forecast step (00Z, 06Z, 12Z, 18Z).",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="DSWRF",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=652,
                    window_reset_frequency=default_window_reset_frequency,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            NoaaDataVar(
                name="downward_long_wave_radiation_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sdlwrf",
                    long_name="Surface downward long-wave radiation flux",
                    units="W/(m^2)",
                    step_type="avg",
                    comment="Average over the previous 1-6 hours, reset every 6-hour forecast step (00Z, 06Z, 12Z, 18Z).",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="DLWRF",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=653,
                    window_reset_frequency=default_window_reset_frequency,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            NoaaDataVar(
                name="pressure_reduced_to_mean_sea_level",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="prmsl",
                    long_name="Pressure reduced to MSL",
                    units="Pa",
                    step_type="instant",
                ),
                internal_attrs=NoaaInternalAttrs(
                    grib_element="PRMSL",
                    grib_description='0[-] MSL="Mean sea level"',
                    grib_index_level="mean sea level",
                    index_position=0,
                    keep_mantissa_bits=10,
                ),
            ),
        ]
