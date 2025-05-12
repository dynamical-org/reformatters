from collections.abc import Sequence
from typing import Any
from pydantic import model_validator

import numpy as np
import pandas as pd
import xarray as xr

from reformatters.common.config_models import (
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVar,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.template_config import AppendDim, Dim, TemplateConfig
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.noaa.noaa_config_models import NOAADataVar, NOAAInternalAttrs


class GFSTemplateConfig(TemplateConfig):
    # ── Everything is now declared here inside the class ──────────────────────
    dataset_attributes: DatasetAttributes = DatasetAttributes(
        dataset_id="noaa-gfs-forecast",
        dataset_version="0.1.0",
        name="NOAA GFS forecast",
        description="Weather forecasts from the Global Forecast System (GFS) operated by NOAA NWS NCEP.",
        attribution="NOAA NWS NCEP GFS data processed by dynamical.org from NOAA Open Data Dissemination archives.",
        spatial_domain="Global",
        spatial_resolution="0.25 degrees (~20km)",
        time_domain="Forecasts initialized 2021-05-01T00:00 UTC to Present",
        time_resolution="Forecasts initialized every 6 hours.",
        forecast_domain="Forecast lead time 0-384 hours (0-16 days) ahead",
        forecast_resolution="Forecast step 0-120h: hourly, 123-384h: 3 hourly",
    )
    time_start: pd.Timestamp = pd.Timestamp("2021-05-01T00:00")
    time_frequency: pd.Timedelta = pd.Timedelta("6h")

    # dims and append_dim
    dims: tuple[Dim, ...] = ("init_time", "lead_time", "latitude", "longitude")
    append_dim: AppendDim = "init_time"

    # raw chunk & shard maps
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

    # --------------------------------------------------------------------
    # now build coords *inside* the class
    def build_coords(self) -> Sequence[Coordinate]:
        # assemble coordinate configs via the base-class helper
        dim_coords = self.dimension_coordinates()

        return [
            Coordinate(
                name=self.append_dim,
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    calendar="proleptic_gregorian",
                    units="seconds since 1970-01-01 00:00:00",
                    chunks=len(dim_coords[self.append_dim]),
                    shards=len(dim_coords[self.append_dim]),
                ),
                attrs=CoordinateAttrs(
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords[self.append_dim].min()), max="Present"
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
                    shards=len(dim_coords["lead_time"]),
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
                    shards=len(dim_coords["latitude"]),
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
                    shards=len(dim_coords["longitude"]),
                ),
                attrs=CoordinateAttrs(
                    units="degrees_east",
                    statistics_approximate=StatisticsApproximate(
                        min=float(dim_coords["longitude"].min()),
                        max=float(dim_coords["longitude"].max()),
                    ),
                ),
            ),
        ]


    # --------------------------------------------------------------------
    # similarly, build data_vars inside the class
    @property
    def data_vars(self) -> Sequence[DataVar[Any]]:
        base_chunks = tuple(self.var_chunks[d] for d in self.dims)
        base_shards = tuple(self.var_shards[d] for d in self.dims)

        return [
            NOAADataVar(
                name="pressure_surface",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="sp",
                    long_name="Surface pressure",
                    units="Pa",
                    step_type="instant",
                    standard_name="surface_air_pressure",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="PRES",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=560,
                    keep_mantissa_bits=10,
                ),
            ),
            NOAADataVar(
                name="temperature_2m",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="t2m",
                    long_name="2 metre temperature",
                    units="C",
                    step_type="instant",
                    standard_name="air_temperature",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="TMP",
                    grib_description='2[m] HTGL="Specified height level above ground"',
                    grib_index_level="2 m above ground",
                    index_position=580,
                    keep_mantissa_bits=7,
                ),
            ),
            NOAADataVar(
                name="relative_humidity_2m",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="r2",
                    long_name="2 metre relative humidity",
                    units="%",
                    step_type="instant",
                    standard_name="relative_humidity",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="RH",
                    grib_description='2[m] HTGL="Specified height level above ground"',
                    grib_index_level="2 m above ground",
                    index_position=583,
                    keep_mantissa_bits=7,
                ),
            ),
            NOAADataVar(
                name="maximum_temperature_2m",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="tmax",
                    long_name="Maximum temperature",
                    units="C",
                    step_type="max",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="TMAX",
                    grib_description='2[m] HTGL="Specified height level above ground"',
                    grib_index_level="2 m above ground",
                    index_position=585,
                    keep_mantissa_bits=7,
                ),
            ),
            NOAADataVar(
                name="minimum_temperature_2m",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="tmin",
                    long_name="Minimum temperature",
                    units="C",
                    step_type="min",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="TMIN",
                    grib_description='2[m] HTGL="Specified height level above ground"',
                    grib_index_level="2 m above ground",
                    index_position=586,
                    keep_mantissa_bits=7,
                ),
            ),
            NOAADataVar(
                name="wind_u_10m",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="u10",
                    long_name="10 metre U wind component",
                    units="m/s",
                    step_type="instant",
                    standard_name="eastward_wind",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="UGRD",
                    grib_description='10[m] HTGL="Specified height level above ground"',
                    grib_index_level="10 m above ground",
                    index_position=587,
                    keep_mantissa_bits=6,
                ),
            ),
            NOAADataVar(
                name="wind_v_10m",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="v10",
                    long_name="10 metre V wind component",
                    units="m/s",
                    step_type="instant",
                    standard_name="northward_wind",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="VGRD",
                    grib_description='10[m] HTGL="Specified height level above ground"',
                    grib_index_level="10 m above ground",
                    index_position=588,
                    keep_mantissa_bits=6,
                ),
            ),
            NOAADataVar(
                name="wind_u_100m",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="u100",
                    long_name="100 metre U wind component",
                    standard_name="eastward_wind",
                    units="m/s",
                    step_type="instant",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="UGRD",
                    grib_description='100[m] HTGL="Specified height level above ground"',
                    grib_index_level="100 m above ground",
                    index_position=688,
                    keep_mantissa_bits=6,
                ),
            ),
            NOAADataVar(
                name="wind_v_100m",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="v100",
                    long_name="100 metre V wind component",
                    units="m/s",
                    step_type="instant",
                    standard_name="northward_wind",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="VGRD",
                    grib_index_level="100 m above ground",
                    grib_description='100[m] HTGL="Specified height level above ground"',
                    index_position=689,
                    keep_mantissa_bits=6,
                ),
            ),
            NOAADataVar(
                name="percent_frozen_precipitation_surface",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="cpofp",
                    long_name="Percent frozen precipitation",
                    units="%",
                    step_type="instant",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="CPOFP",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=590,
                    keep_mantissa_bits=7,
                ),
            ),
            NOAADataVar(
                name="precipitation_surface",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="tp",
                    long_name="Total Precipitation",
                    units="mm/s",
                    comment="Average precipitation rate since the previous forecast step.",
                    step_type="avg",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="APCP",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=595,
                    include_lead_time_suffix=True,
                    deaccumulate_to_rates=True,
                    keep_mantissa_bits=7,
                ),
            ),
            NOAADataVar(
                name="categorical_snow_surface",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="csnow",
                    long_name="Categorical snow",
                    units="0=no; 1=yes",
                    step_type="avg",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="CSNOW",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=604,
                    keep_mantissa_bits="no-rounding",
                ),
            ),
            NOAADataVar(
                name="categorical_ice_pellets_surface",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="cicep",
                    long_name="Categorical ice pellets",
                    units="0=no; 1=yes",
                    step_type="avg",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="CICEP",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=605,
                    keep_mantissa_bits="no-rounding",
                ),
            ),
            NOAADataVar(
                name="categorical_freezing_rain_surface",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="cfrzr",
                    long_name="Categorical freezing rain",
                    units="0=no; 1=yes",
                    step_type="avg",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="CFRZR",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=606,
                    keep_mantissa_bits="no-rounding",
                ),
            ),
            NOAADataVar(
                name="categorical_rain_surface",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="crain",
                    long_name="Categorical rain",
                    units="0=no; 1=yes",
                    step_type="avg",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="CRAIN",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=607,
                    keep_mantissa_bits="no-rounding",
                ),
            ),
            NOAADataVar(
                name="precipitable_water_atmosphere",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="pwat",
                    long_name="Precipitable water",
                    units="kg/(m^2)",
                    step_type="instant",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="PWAT",
                    grib_description='0[-] EATM="Entire atmosphere (considered as a single layer)"',
                    grib_index_level="entire atmosphere (considered as a single layer)",
                    index_position=625,
                    keep_mantissa_bits=7,
                ),
            ),
            NOAADataVar(
                name="total_cloud_cover_atmosphere",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="tcc",
                    long_name="Total Cloud Cover",
                    units="%",
                    step_type="avg",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="TCDC",
                    grib_description='0[-] EATM="Entire Atmosphere"',
                    grib_index_level="entire atmosphere",
                    index_position=635,
                    keep_mantissa_bits=7,
                ),
            ),
            NOAADataVar(
                name="geopotential_height_cloud_ceiling",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="gh",
                    long_name="Geopotential height",
                    units="gpm",
                    step_type="instant",
                    standard_name="geopotential_height",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="HGT",
                    grib_description='0[-] CEIL="Cloud ceiling"',
                    grib_index_level="cloud ceiling",
                    index_position=637,
                    keep_mantissa_bits=8,
                ),
            ),
            NOAADataVar(
                name="downward_short_wave_radiation_flux_surface",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="sdswrf",
                    long_name="Surface downward short-wave radiation flux",
                    units="W/(m^2)",
                    step_type="avg",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="DSWRF",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=652,
                    keep_mantissa_bits=7,
                ),
            ),
            NOAADataVar(
                name="downward_long_wave_radiation_flux_surface",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="sdlwrf",
                    long_name="Surface downward long-wave radiation flux",
                    units="W/(m^2)",
                    step_type="avg",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="DLWRF",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=653,
                    keep_mantissa_bits=7,
                ),
            ),
            NOAADataVar(
                name="pressure_reduced_to_mean_sea_level",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=base_chunks,
                    shards=base_shards,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="prmsl",
                    long_name="Pressure reduced to MSL",
                    units="Pa",
                    step_type="instant",
                ),
                internal_attrs=NOAAInternalAttrs(
                    grib_element="PRMSL",
                    grib_description='0[-] MSL="Mean sea level"',
                    grib_index_level="mean sea level",
                    index_position=0,
                    keep_mantissa_bits=10,
                ),
            ),
        ]

    # --------------------------------------------------------------------

    def dimension_coordinates(self) -> dict[str, Any]:
        """
        Returns a dictionary of dimension names to coordinates for the dataset.
        """
        end = self.time_start + self.time_frequency
        return {
            self.append_dim: pd.date_range(
                self.time_start, end, freq=self.time_frequency, inclusive="left"
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
            "valid_time": ds[self.append_dim] + ds["lead_time"],
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
        }
