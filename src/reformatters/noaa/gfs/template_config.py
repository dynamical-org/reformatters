from collections.abc import Sequence

import numpy as np
import pandas as pd

from reformatters.common.config_models import (
    Coordinate,
    CoordinateAttrs,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.template_config import TemplateConfig
from reformatters.common.zarr import (
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.noaa.models import NoaaDataVar, NoaaInternalAttrs


class NoaaGfsCommonTemplateConfig(TemplateConfig[NoaaDataVar]):
    """Common template configuration for GFS datasets."""

    def _latitude_longitude_coordinates(self) -> dict[str, np.ndarray]:
        return {
            "latitude": np.flip(np.arange(-90, 90.25, 0.25)),
            "longitude": np.arange(-180, 180, 0.25),
        }

    @property
    def coords(self) -> Sequence[Coordinate]:
        dim_coords = self._latitude_longitude_coordinates()

        return [
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

    def get_data_vars(self, encoding: Encoding) -> Sequence[NoaaDataVar]:
        """Return the data variables for the dataset with the given encoding."""
        default_window_reset_frequency = pd.Timedelta("6h")
        default_keep_mantissa_bits = 7

        return [
            NoaaDataVar(
                name="pressure_surface",
                encoding=encoding,
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="t2m",
                    long_name="2 metre temperature",
                    units="degree_Celsius",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="r2",
                    long_name="2 metre relative humidity",
                    units="percent",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="tmax",
                    standard_name="air_temperature",
                    long_name="Maximum temperature",
                    units="degree_Celsius",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="tmin",
                    standard_name="air_temperature",
                    long_name="Minimum temperature",
                    units="degree_Celsius",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="u10",
                    long_name="10 metre U wind component",
                    units="m s-1",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="v10",
                    long_name="10 metre V wind component",
                    units="m s-1",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="u100",
                    long_name="100 metre U wind component",
                    standard_name="eastward_wind",
                    units="m s-1",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="v100",
                    long_name="100 metre V wind component",
                    units="m s-1",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="cpofp",
                    long_name="Percent frozen precipitation",
                    units="percent",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="tp",
                    standard_name="precipitation_flux",
                    long_name="Total Precipitation",
                    units="kg m-2 s-1",
                    comment="Average precipitation rate since the previous forecast step. Units equivalent to mm/s.",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="csnow",
                    long_name="Categorical snow",
                    units="1",
                    step_type="avg",
                    comment="Presence/absence over the previous 1-6 hours, reset every 6-hour forecast step (00Z, 06Z, 12Z, 18Z). 0=no; 1=yes.",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="cicep",
                    long_name="Categorical ice pellets",
                    units="1",
                    step_type="avg",
                    comment="Presence/absence over the previous 1-6 hours, reset every 6-hour forecast step (00Z, 06Z, 12Z, 18Z). 0=no; 1=yes.",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="cfrzr",
                    long_name="Categorical freezing rain",
                    units="1",
                    step_type="avg",
                    comment="Presence/absence over the previous 1-6 hours, reset every 6-hour forecast step (00Z, 06Z, 12Z, 18Z). 0=no; 1=yes.",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="crain",
                    long_name="Categorical rain",
                    units="1",
                    step_type="avg",
                    comment="Presence/absence over the previous 1-6 hours, reset every 6-hour forecast step (00Z, 06Z, 12Z, 18Z). 0=no; 1=yes.",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="pwat",
                    standard_name="atmosphere_mass_content_of_water_vapor",
                    long_name="Precipitable water",
                    units="kg m-2",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="tcc",
                    standard_name="cloud_area_fraction",
                    long_name="Total Cloud Cover",
                    units="percent",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="gh",
                    long_name="Geopotential height",
                    units="m",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="sdswrf",
                    standard_name="surface_downwelling_shortwave_flux_in_air",
                    long_name="Surface downward short-wave radiation flux",
                    units="W m-2",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="sdlwrf",
                    standard_name="surface_downwelling_longwave_flux_in_air",
                    long_name="Surface downward long-wave radiation flux",
                    units="W m-2",
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
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="prmsl",
                    standard_name="air_pressure_at_mean_sea_level",
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
