from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from reformatters.common.config_models import (
    Coordinate,
    CoordinateAttrs,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar, GEFSInternalAttrs


def get_shared_template_dimension_coordinates() -> dict[str, Any]:
    return {
        # latitude descends when north is up
        "latitude": np.flip(np.arange(-90, 90.25, 0.25)),
        "longitude": np.arange(-180, 180, 0.25),
    }


def get_shared_coordinate_configs() -> Sequence[Coordinate]:
    _dim_coords = get_shared_template_dimension_coordinates()

    return (
        Coordinate(
            name="latitude",
            encoding=Encoding(
                dtype="float64",
                fill_value=np.nan,
                compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                chunks=len(_dim_coords["latitude"]),
                shards=len(_dim_coords["latitude"]),
            ),
            attrs=CoordinateAttrs(
                units="degrees_north",
                statistics_approximate=StatisticsApproximate(
                    min=_dim_coords["latitude"].min(),
                    max=_dim_coords["latitude"].max(),
                ),
            ),
        ),
        Coordinate(
            name="longitude",
            encoding=Encoding(
                dtype="float64",
                fill_value=np.nan,
                compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                chunks=len(_dim_coords["longitude"]),
                shards=len(_dim_coords["longitude"]),
            ),
            attrs=CoordinateAttrs(
                units="degrees_east",
                statistics_approximate=StatisticsApproximate(
                    min=_dim_coords["longitude"].min(),
                    max=_dim_coords["longitude"].max(),
                ),
            ),
        ),
        Coordinate(
            name="spatial_ref",
            encoding=Encoding(
                dtype="int64",
                fill_value=0,
                chunks=1,  # Scalar coordinate
                shards=1,
            ),
            attrs=CoordinateAttrs(
                units="unitless",
                statistics_approximate=StatisticsApproximate(
                    min=0,
                    max=0,
                ),
            ),
        ),
    )


def get_shared_data_var_configs(
    chunks: tuple[int, ...], shards: tuple[int, ...]
) -> Sequence[GEFSDataVar]:
    assert len(chunks) == len(shards)

    encoding_float32 = Encoding(
        dtype="float32",
        fill_value=np.nan,
        chunks=chunks,
        shards=shards,
        compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
    )

    keep_mantissa_bits_default = 7
    keep_mantissa_bits_categorical: Literal["no-rounding"] = "no-rounding"

    return (
        GEFSDataVar(
            name="pressure_surface",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="sp",
                long_name="Surface pressure",
                units="Pa",
                step_type="instant",
                standard_name="surface_air_pressure",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="PRES",
                grib_description='0[-] SFC="Ground or water surface"',
                grib_index_level="surface",
                gefs_file_type="s+a",
                index_position=4,
                keep_mantissa_bits=10,
            ),
        ),
        GEFSDataVar(
            name="temperature_2m",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="t2m",
                long_name="2 metre temperature",
                units="C",
                step_type="instant",
                standard_name="air_temperature",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="TMP",
                grib_description='2[m] HTGL="Specified height level above ground"',
                grib_index_level="2 m above ground",
                gefs_file_type="s+a",
                index_position=10,
                keep_mantissa_bits=keep_mantissa_bits_default,
            ),
        ),
        GEFSDataVar(
            name="relative_humidity_2m",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="r2",
                long_name="2 metre relative humidity",
                units="%",
                step_type="instant",
                standard_name="relative_humidity",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="RH",
                grib_description='2[m] HTGL="Specified height level above ground"',
                grib_index_level="2 m above ground",
                gefs_file_type="s+a",
                index_position=12,
                keep_mantissa_bits=keep_mantissa_bits_default,
            ),
        ),
        GEFSDataVar(
            name="maximum_temperature_2m",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="tmax",
                long_name="Maximum temperature",
                units="C",
                step_type="max",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="TMAX",
                grib_description='2[m] HTGL="Specified height level above ground"',
                grib_index_level="2 m above ground",
                gefs_file_type="s+a",
                index_position=13,
                keep_mantissa_bits=keep_mantissa_bits_default,
            ),
        ),
        GEFSDataVar(
            name="minimum_temperature_2m",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="tmin",
                long_name="Minimum temperature",
                units="C",
                step_type="min",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="TMIN",
                grib_description='2[m] HTGL="Specified height level above ground"',
                grib_index_level="2 m above ground",
                gefs_file_type="s+a",
                index_position=14,
                keep_mantissa_bits=keep_mantissa_bits_default,
            ),
        ),
        GEFSDataVar(
            name="wind_u_10m",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="u10",
                long_name="10 metre U wind component",
                units="m/s",
                step_type="instant",
                standard_name="eastward_wind",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="UGRD",
                grib_description='10[m] HTGL="Specified height level above ground"',
                grib_index_level="10 m above ground",
                gefs_file_type="s+a",
                index_position=15,
                keep_mantissa_bits=6,
            ),
        ),
        GEFSDataVar(
            name="wind_v_10m",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="v10",
                long_name="10 metre V wind component",
                units="m/s",
                step_type="instant",
                standard_name="northward_wind",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="VGRD",
                grib_description='10[m] HTGL="Specified height level above ground"',
                grib_index_level="10 m above ground",
                gefs_file_type="s+a",
                index_position=16,
                keep_mantissa_bits=6,
            ),
        ),
        GEFSDataVar(
            name="wind_u_100m",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="u100",
                long_name="100 metre U wind component",
                standard_name="eastward_wind",
                units="m/s",
                comment="All lead times of this variable are interpolated from a 0.5 degree grid.",
                step_type="instant",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="UGRD",
                grib_description='100[m] HTGL="Specified height level above ground"',
                grib_index_level="100 m above ground",
                gefs_file_type="b",
                index_position=357,
                keep_mantissa_bits=6,
            ),
        ),
        GEFSDataVar(
            name="wind_v_100m",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="v100",
                long_name="100 metre V wind component",
                units="m/s",
                comment="All lead times of this variable are interpolated from a 0.5 degree grid.",
                step_type="instant",
                standard_name="northward_wind",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="VGRD",
                grib_description='100[m] HTGL="Specified height level above ground"',
                grib_index_level="100 m above ground",
                gefs_file_type="b",
                index_position=358,
                keep_mantissa_bits=6,
            ),
        ),
        GEFSDataVar(
            name="percent_frozen_precipitation_surface",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="cpofp",
                long_name="Percent frozen precipitation",
                units="%",
                step_type="instant",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="CPOFP",
                grib_description='0[-] SFC="Ground or water surface"',
                grib_index_level="surface",
                gefs_file_type="s+b",
                index_position=17,
                keep_mantissa_bits=keep_mantissa_bits_default,
            ),
        ),
        GEFSDataVar(
            name="precipitation_surface",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="tp",
                long_name="Total Precipitation",
                units="mm/s",
                comment="Average precipitation rate since the previous forecast step.",
                step_type="avg",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="APCP",
                grib_description='0[-] SFC="Ground or water surface"',
                grib_index_level="surface",
                gefs_file_type="s+a",
                index_position=18,
                include_lead_time_suffix=True,
                deaccumulate_to_rates=True,
                keep_mantissa_bits=keep_mantissa_bits_default,
            ),
        ),
        GEFSDataVar(
            name="categorical_snow_surface",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="csnow",
                long_name="Categorical snow",
                units="0=no; 1=yes",
                step_type="avg",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="CSNOW",
                grib_description='0[-] SFC="Ground or water surface"',
                grib_index_level="surface",
                gefs_file_type="s+a",
                index_position=19,
                keep_mantissa_bits=keep_mantissa_bits_categorical,
            ),
        ),
        GEFSDataVar(
            name="categorical_ice_pellets_surface",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="cicep",
                long_name="Categorical ice pellets",
                units="0=no; 1=yes",
                step_type="avg",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="CICEP",
                grib_description='0[-] SFC="Ground or water surface"',
                grib_index_level="surface",
                gefs_file_type="s+a",
                index_position=20,
                keep_mantissa_bits=keep_mantissa_bits_categorical,
            ),
        ),
        GEFSDataVar(
            name="categorical_freezing_rain_surface",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="cfrzr",
                long_name="Categorical freezing rain",
                units="0=no; 1=yes",
                step_type="avg",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="CFRZR",
                grib_description='0[-] SFC="Ground or water surface"',
                grib_index_level="surface",
                gefs_file_type="s+a",
                index_position=21,
                keep_mantissa_bits=keep_mantissa_bits_categorical,
            ),
        ),
        GEFSDataVar(
            name="categorical_rain_surface",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="crain",
                long_name="Categorical rain",
                units="0=no; 1=yes",
                step_type="avg",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="CRAIN",
                grib_description='0[-] SFC="Ground or water surface"',
                grib_index_level="surface",
                gefs_file_type="s+a",
                index_position=22,
                keep_mantissa_bits=keep_mantissa_bits_categorical,
            ),
        ),
        GEFSDataVar(
            name="precipitable_water_atmosphere",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="pwat",
                long_name="Precipitable water",
                units="kg/(m^2)",
                step_type="instant",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="PWAT",
                grib_description='0[-] EATM="Entire atmosphere (considered as a single layer)"',
                grib_index_level="entire atmosphere (considered as a single layer)",
                gefs_file_type="s+a",
                index_position=27,
                keep_mantissa_bits=keep_mantissa_bits_default,
            ),
        ),
        GEFSDataVar(
            name="total_cloud_cover_atmosphere",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="tcc",
                long_name="Total Cloud Cover",
                units="%",
                comment="Average value in the last 6 hour period (00, 06, 12, 18 UTC) or 3 hour period (03, 09, 15, 21 UTC).",
                step_type="avg",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="TCDC",
                grib_description='0[-] EATM="Entire Atmosphere"',
                grib_index_level="entire atmosphere",
                gefs_file_type="s+a",
                index_position=28,
                keep_mantissa_bits=keep_mantissa_bits_default,
            ),
        ),
        GEFSDataVar(
            name="geopotential_height_cloud_ceiling",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="gh",
                long_name="Geopotential height",
                units="gpm",
                step_type="instant",
                standard_name="geopotential_height",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="HGT",
                grib_description='0[-] CEIL="Cloud ceiling"',
                grib_index_level="cloud ceiling",
                gefs_file_type="s+b",
                index_position=29,
                keep_mantissa_bits=8,
            ),
        ),
        GEFSDataVar(
            name="downward_short_wave_radiation_flux_surface",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="sdswrf",
                long_name="Surface downward short-wave radiation flux",
                units="W/(m^2)",
                comment="Average value in the last 6 hour period (00, 06, 12, 18 UTC) or 3 hour period (03, 09, 15, 21 UTC).",
                step_type="avg",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="DSWRF",
                grib_description='0[-] SFC="Ground or water surface"',
                grib_index_level="surface",
                gefs_file_type="s+a",
                index_position=30,
                keep_mantissa_bits=keep_mantissa_bits_default,
            ),
        ),
        GEFSDataVar(
            name="downward_long_wave_radiation_flux_surface",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="sdlwrf",
                long_name="Surface downward long-wave radiation flux",
                units="W/(m^2)",
                comment="Average value in the last 6 hour period (00, 06, 12, 18 UTC) or 3 hour period (03, 09, 15, 21 UTC).",
                step_type="avg",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="DLWRF",
                grib_description='0[-] SFC="Ground or water surface"',
                grib_index_level="surface",
                gefs_file_type="s+a",
                index_position=31,
                keep_mantissa_bits=keep_mantissa_bits_default,
            ),
        ),
        GEFSDataVar(
            name="pressure_reduced_to_mean_sea_level",
            encoding=encoding_float32,
            attrs=DataVarAttrs(
                short_name="prmsl",
                long_name="Pressure reduced to MSL",
                units="Pa",
                step_type="instant",
            ),
            internal_attrs=GEFSInternalAttrs(
                grib_element="PRMSL",
                grib_description='0[-] MSL="Mean sea level"',
                grib_index_level="mean sea level",
                gefs_file_type="s+a",
                index_position=38,
                keep_mantissa_bits=10,
            ),
        ),
    )
