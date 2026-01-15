from collections.abc import Sequence

import numpy as np
import pandas as pd
import pyproj
from pydantic import computed_field

from reformatters.common.config_models import (
    Coordinate,
    CoordinateAttrs,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import (
    Array1D,
    Array2D,
)
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.noaa.hrrr.hrrr_config_models import (
    NoaaHrrrDataVar,
    NoaaHrrrInternalAttrs,
)


class NoaaHrrrCommonTemplateConfig(TemplateConfig[NoaaHrrrDataVar]):
    @computed_field  # type: ignore[prop-decorator]
    @property
    def coords(self) -> Sequence[Coordinate]:
        y_coords, x_coords = self._y_x_coordinates()

        return [
            Coordinate(
                name="x",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(x_coords),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="X coordinate of projection",
                    standard_name="projection_x_coordinate",
                    units="m",
                    axis="X",
                    statistics_approximate=StatisticsApproximate(
                        min=-2700000.0,
                        max=2700000.0,
                    ),
                ),
            ),
            Coordinate(
                name="y",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(y_coords),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Y coordinate of projection",
                    standard_name="projection_y_coordinate",
                    units="m",
                    axis="Y",
                    statistics_approximate=StatisticsApproximate(
                        min=-1600000.0,
                        max=1600000.0,
                    ),
                ),
            ),
            Coordinate(
                name="latitude",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=(len(y_coords), len(x_coords)),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Latitude",
                    standard_name="latitude",
                    units="degrees_north",
                    statistics_approximate=StatisticsApproximate(
                        min=21.138123,
                        max=52.615653,
                    ),
                ),
            ),
            Coordinate(
                name="longitude",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=(len(y_coords), len(x_coords)),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Longitude",
                    standard_name="longitude",
                    units="degrees_east",
                    statistics_approximate=StatisticsApproximate(
                        min=-134.09548,
                        max=-60.917192,
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
                    # Derived from opening a sample HRRR file, see
                    # tests/noaa/hrrr/template_config_test.py::test_spatial_info_matches_file
                    GeoTransform="-2699020.142521929 3000.0 0.0 1588193.847443335 0.0 -3000.0",
                    crs_wkt='PROJCS["unnamed",GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",SPHEROID["Sphere",6371229,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["latitude_of_origin",38.5],PARAMETER["central_meridian",-97.5],PARAMETER["standard_parallel_1",38.5],PARAMETER["standard_parallel_2",38.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]',
                    false_easting=0.0,
                    false_northing=0.0,
                    geographic_crs_name="Coordinate System imported from GRIB file",
                    grid_mapping_name="lambert_conformal_conic",
                    horizontal_datum_name="unnamed",
                    inverse_flattening=0.0,
                    latitude_of_projection_origin=38.5,
                    longitude_of_central_meridian=-97.5,
                    longitude_of_prime_meridian=0.0,
                    prime_meridian_name="Greenwich",
                    projected_crs_name="unnamed",
                    reference_ellipsoid_name="Sphere",
                    semi_major_axis=6371229.0,
                    semi_minor_axis=6371229.0,
                    spatial_ref='PROJCS["unnamed",GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",SPHEROID["Sphere",6371229,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["latitude_of_origin",38.5],PARAMETER["central_meridian",-97.5],PARAMETER["standard_parallel_1",38.5],PARAMETER["standard_parallel_2",38.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]',
                    standard_parallel=(38.5, 38.5),
                ),
            ),
        ]

    def get_data_vars(self, encoding: Encoding) -> Sequence[NoaaHrrrDataVar]:
        default_window_reset_frequency = pd.Timedelta("1h")
        default_keep_mantissa_bits = 7

        return [
            NoaaHrrrDataVar(
                name="composite_reflectivity",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="refc",
                    long_name="Composite reflectivity",
                    units="dBZ",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="REFC",
                    grib_description='0[-] EATM="Entire Atmosphere"',
                    index_position=1,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    grib_index_level="entire atmosphere",
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="temperature_2m",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="t2m",
                    long_name="2 metre temperature",
                    units="degree_Celsius",
                    step_type="instant",
                    standard_name="air_temperature",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="TMP",
                    grib_description='2[m] HTGL="Specified height level above ground"',
                    grib_index_level="2 m above ground",
                    index_position=71,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="wind_u_10m",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="u10",
                    long_name="10 metre U wind component",
                    units="m/s",
                    step_type="instant",
                    standard_name="eastward_wind",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="UGRD",
                    grib_description='10[m] HTGL="Specified height level above ground"',
                    grib_index_level="10 m above ground",
                    index_position=77,
                    keep_mantissa_bits=6,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="wind_v_10m",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="v10",
                    long_name="10 metre V wind component",
                    units="m/s",
                    step_type="instant",
                    standard_name="northward_wind",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="VGRD",
                    grib_description='10[m] HTGL="Specified height level above ground"',
                    grib_index_level="10 m above ground",
                    index_position=78,
                    keep_mantissa_bits=6,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="precipitation_surface",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="tp",
                    standard_name="precipitation_flux",
                    long_name="Total Precipitation",
                    units="mm/s",
                    comment="Average precipitation rate since the previous forecast step.",
                    step_type="avg",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="APCP",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=84,
                    deaccumulate_to_rate=True,
                    window_reset_frequency=default_window_reset_frequency,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="precipitable_water_atmosphere",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="pwat",
                    standard_name="atmosphere_mass_content_of_water_vapor",
                    long_name="Precipitable water",
                    units="kg m-2",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="PWAT",
                    grib_description='0[-] EATM="Entire atmosphere (considered as a single layer)"',
                    grib_index_level="entire atmosphere (considered as a single layer)",
                    index_position=107,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="total_cloud_cover_atmosphere",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="tcc",
                    standard_name="cloud_area_fraction",
                    long_name="Total Cloud Cover",
                    units="percent",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="TCDC",
                    grib_description='0[-] EATM="Entire Atmosphere"',
                    grib_index_level="entire atmosphere",
                    index_position=116,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="downward_short_wave_radiation_flux_surface",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="sdswrf",
                    standard_name="surface_downwelling_shortwave_flux_in_air",
                    long_name="Surface downward short-wave radiation flux",
                    units="W m-2",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="DSWRF",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=123,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="downward_long_wave_radiation_flux_surface",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="sdlwrf",
                    standard_name="surface_downwelling_longwave_flux_in_air",
                    long_name="Surface downward long-wave radiation flux",
                    units="W m-2",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="DLWRF",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=124,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="pressure_reduced_to_mean_sea_level",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="prmsl",
                    standard_name="air_pressure_at_mean_sea_level",
                    long_name="Pressure reduced to MSL",
                    units="Pa",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="MSLMA",
                    grib_description='0[-] MSL="Mean sea level"',
                    grib_index_level="mean sea level",
                    index_position=41,
                    keep_mantissa_bits=13,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="percent_frozen_precipitation_surface",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="cpofp",
                    long_name="Percent frozen precipitation",
                    units="percent",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="CPOFP",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=82,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="pressure_surface",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="sp",
                    standard_name="surface_air_pressure",
                    long_name="Surface pressure",
                    units="Pa",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="PRES",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=62,
                    keep_mantissa_bits=10,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="categorical_ice_pellets_surface",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="cicep",
                    long_name="Categorical ice pellets",
                    units="1",
                    comment="0=no; 1=yes",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="CICEP",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=91,
                    keep_mantissa_bits="no-rounding",
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="categorical_snow_surface",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="csnow",
                    long_name="Categorical snow",
                    units="1",
                    comment="0=no; 1=yes",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="CSNOW",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=90,
                    keep_mantissa_bits="no-rounding",
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="categorical_freezing_rain_surface",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="cfrzr",
                    long_name="Categorical freezing rain",
                    units="1",
                    comment="0=no; 1=yes",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="CFRZR",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=92,
                    keep_mantissa_bits="no-rounding",
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="categorical_rain_surface",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="crain",
                    long_name="Categorical rain",
                    units="1",
                    comment="0=no; 1=yes",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="CRAIN",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_index_level="surface",
                    index_position=93,
                    keep_mantissa_bits="no-rounding",
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="relative_humidity_2m",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="r2",
                    standard_name="relative_humidity",
                    long_name="2 metre relative humidity",
                    units="percent",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="RH",
                    grib_description='2[m] HTGL="Specified height level above ground"',
                    grib_index_level="2 m above ground",
                    index_position=75,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="geopotential_height_cloud_ceiling",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="gh",
                    standard_name="geopotential_height",
                    long_name="Geopotential height",
                    units="gpm",
                    step_type="instant",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="HGT",
                    grib_description='0[-] CEIL="Cloud ceiling"',
                    grib_index_level="cloud ceiling",
                    index_position=117,
                    keep_mantissa_bits=8,
                    hrrr_file_type="sfc",
                ),
            ),
            # HRRR provides 80m but not 100m winds
            NoaaHrrrDataVar(
                name="wind_u_80m",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="u80",
                    long_name="U-component of wind (80 m above ground)",
                    units="m/s",
                    step_type="instant",
                    standard_name="eastward_wind",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="UGRD",
                    grib_description='80[m] HTGL="Specified height level above ground"',
                    grib_index_level="80 m above ground",
                    index_position=60,
                    keep_mantissa_bits=6,
                    hrrr_file_type="sfc",
                ),
            ),
            NoaaHrrrDataVar(
                name="wind_v_80m",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="v80",
                    long_name="V-component of wind (80 m above ground)",
                    units="m/s",
                    step_type="instant",
                    standard_name="northward_wind",
                ),
                internal_attrs=NoaaHrrrInternalAttrs(
                    grib_element="VGRD",
                    grib_description='80[m] HTGL="Specified height level above ground"',
                    grib_index_level="80 m above ground",
                    index_position=61,
                    keep_mantissa_bits=6,
                    hrrr_file_type="sfc",
                ),
            ),
        ]

    def _spatial_info(
        self,
    ) -> tuple[
        tuple[int, int], tuple[float, float, float, float], tuple[float, float], str
    ]:
        """
        Returns (shape, bounds, resolution, crs proj4 string).
        Useful for deriving x, y and latitude, longitude coordinates.
        See tests/noaa/hrrr/template_config_test.py::test_spatial_info_matches_file
        """
        return (
            (1059, 1799),
            (
                -2699020.142521929,
                -1588806.152556665,
                2697979.857478071,
                1588193.847443335,
            ),
            (3000.0, -3000.0),
            "+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs=True",
        )

    def _y_x_coordinates(self) -> tuple[Array1D[np.float64], Array1D[np.float64]]:
        shape, bounds, resolution, _crs = self._spatial_info()
        dx, dy = resolution
        left, _bottom, _right, top = bounds
        ny, nx = shape
        # add 1/2 a pixel to corner of bounds to get pixel center
        y_coords = (top + (0.5 * dy)) + (np.arange(ny) * dy)
        x_coords = (left + (0.5 * dx)) + (np.arange(nx) * dx)
        # astype is no-op for type checker
        return y_coords.astype(np.float64), x_coords.astype(np.float64)

    def _latitude_longitude_coordinates(
        self, x_coords: Array1D[np.float64], y_coords: Array1D[np.float64]
    ) -> tuple[Array2D[np.float32], Array2D[np.float32]]:
        # Create 2D latitude and longitude grids
        # x, y are the spatial dimensions of this dataset
        # latitude and longitude there
        _, _, _, crs = self._spatial_info()
        xs, ys = np.meshgrid(x_coords, y_coords)
        lons, lats = pyproj.Proj(crs)(xs, ys, inverse=True)
        # Dropping to 32 bit precision still gets us < 1 meter precision and
        # makes each array about 6MB vs 15MB for float64.
        lats = lats.astype(np.float32)
        lons = lons.astype(np.float32)
        return lats, lons
