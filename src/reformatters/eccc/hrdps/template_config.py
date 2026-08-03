import functools
from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd
from pydantic import computed_field
from pyproj import CRS, Transformer

from reformatters.common.config_models import (
    Coordinate,
    CoordinateAttrs,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.deaccumulation import RADIATION_INVALID_BELOW_THRESHOLD
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import Array1D, Array2D, Timedelta
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.eccc.hrdps.hrdps_config_models import (
    EcccHrdpsDataVar,
    EcccHrdpsInternalAttrs,
)

# HRDPS runs at 00, 06, 12, and 18 UTC. The forecast dataset's
# append_dim_frequency and the analysis dataset's deaccumulation reset both
# follow from this.
HRDPS_INIT_FREQUENCY: Final[Timedelta] = pd.Timedelta("6h")

# Extracted from an HRDPS continental GRIB2 file by rasterio/GDAL, see
# tests/eccc/hrdps/template_config_test.py::test_spatial_info_matches_file.
# A rotated latitude-longitude grid on a perfect sphere (R=6,371,229m) with the
# GRIB-convention southern pole at (-36.08852, -114.694858).
HRDPS_CRS_WKT = (
    'GEOGCRS["Coordinate System imported from GRIB file",'
    'BASEGEOGCRS["Coordinate System imported from GRIB file",'
    'DATUM["unnamed",ELLIPSOID["Sphere",6371229,0,LENGTHUNIT["metre",1,ID["EPSG",9001]]]],'
    'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]]],'
    'DERIVINGCONVERSION["Pole rotation (GRIB convention)",'
    'METHOD["Pole rotation (GRIB convention)"],'
    'PARAMETER["Latitude of the southern pole (GRIB convention)",-36.08852,'
    'ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]],'
    'PARAMETER["Longitude of the southern pole (GRIB convention)",-114.694858,'
    'ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]],'
    'PARAMETER["Axis rotation (GRIB convention)",0,'
    'ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]]],'
    "CS[ellipsoidal,2],"
    'AXIS["latitude",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]],'
    'AXIS["longitude",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]]]'
)


class EcccHrdpsCommonTemplateConfig(TemplateConfig[EcccHrdpsDataVar]):
    @computed_field
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
                    long_name="Longitude in rotated pole grid",
                    standard_name="grid_longitude",
                    units="degrees",
                    axis="X",
                    statistics_approximate=StatisticsApproximate(
                        min=float(x_coords.min()),
                        max=float(x_coords.max()),
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
                    long_name="Latitude in rotated pole grid",
                    standard_name="grid_latitude",
                    units="degrees",
                    axis="Y",
                    statistics_approximate=StatisticsApproximate(
                        min=float(y_coords.min()),
                        max=float(y_coords.max()),
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
                    units="degree_north",
                    statistics_approximate=StatisticsApproximate(
                        min=27.284597,
                        max=70.611480,
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
                    units="degree_east",
                    statistics_approximate=StatisticsApproximate(
                        min=-152.730666,
                        max=-40.708561,
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
                    # Derived by opening a sample HRDPS file, see
                    # tests/eccc/hrdps/template_config_test.py::test_spatial_info_matches_file
                    crs_wkt=HRDPS_CRS_WKT,
                    spatial_ref=HRDPS_CRS_WKT,
                    GeoTransform="-14.832470000590822 0.022500001181567565 0.0 16.711251000775796 0.0 -0.02250000155159038",
                    comment=(
                        "Rotated latitude-longitude grid on a perfect sphere with a radius "
                        "of 6,371,229m, extracted from grib. The y and x dimension "
                        "coordinates are latitude and longitude in the rotated system; the "
                        "2D latitude and longitude coordinates give the true geographic "
                        "position of each grid point."
                    ),
                ),
            ),
        ]

    def get_data_vars(self, encoding: Encoding) -> Sequence[EcccHrdpsDataVar]:
        default_keep_mantissa_bits = 7

        return [
            EcccHrdpsDataVar(
                name="temperature_2m",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="2t",
                    long_name="2 metre temperature",
                    units="degree_Celsius",
                    step_type="instant",
                    standard_name="air_temperature",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="TMP_AGL-2m",
                    grib_element="TMP",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcccHrdpsDataVar(
                name="dew_point_temperature_2m",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="2d",
                    standard_name="dew_point_temperature",
                    long_name="2 metre dewpoint temperature",
                    units="degree_Celsius",
                    step_type="instant",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="DPT_AGL-2m",
                    grib_element="DPT",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcccHrdpsDataVar(
                name="relative_humidity_2m",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="2r",
                    standard_name="relative_humidity",
                    long_name="2 metre relative humidity",
                    units="percent",
                    step_type="instant",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="RH_AGL-2m",
                    grib_element="RH",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcccHrdpsDataVar(
                name="specific_humidity_2m",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="q",
                    standard_name="specific_humidity",
                    long_name="Specific humidity",
                    units="1",
                    step_type="instant",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="SPFH_AGL-2m",
                    grib_element="SPFH",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcccHrdpsDataVar(
                name="wind_u_10m",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="10u",
                    long_name="10 metre U wind component",
                    units="m s-1",
                    step_type="instant",
                    standard_name="eastward_wind",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="UGRD_AGL-10m",
                    grib_element="UGRD",
                    keep_mantissa_bits=6,
                ),
            ),
            EcccHrdpsDataVar(
                name="wind_v_10m",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="10v",
                    long_name="10 metre V wind component",
                    units="m s-1",
                    step_type="instant",
                    standard_name="northward_wind",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="VGRD_AGL-10m",
                    grib_element="VGRD",
                    keep_mantissa_bits=6,
                ),
            ),
            EcccHrdpsDataVar(
                name="wind_u_80m",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="80u",
                    long_name="80 metre U wind component",
                    units="m s-1",
                    step_type="instant",
                    standard_name="eastward_wind",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="UGRD_AGL-80m",
                    grib_element="UGRD",
                    keep_mantissa_bits=6,
                ),
            ),
            EcccHrdpsDataVar(
                name="wind_v_80m",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="80v",
                    long_name="80 metre V wind component",
                    units="m s-1",
                    step_type="instant",
                    standard_name="northward_wind",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="VGRD_AGL-80m",
                    grib_element="VGRD",
                    keep_mantissa_bits=6,
                ),
            ),
            EcccHrdpsDataVar(
                name="instantaneous_wind_gust_10m",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="i10fg",
                    standard_name="wind_speed_of_gust",
                    long_name="Instantaneous 10 metre wind gust",
                    units="m s-1",
                    step_type="instant",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="GUST_AGL-10m",
                    grib_element="GUST",
                    keep_mantissa_bits=6,
                ),
            ),
            EcccHrdpsDataVar(
                name="precipitation_surface",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="prate",
                    standard_name="precipitation_flux",
                    long_name="Precipitation rate",
                    units="kg m-2 s-1",
                    comment="Average precipitation rate since the previous forecast step. Units equivalent to mm/s.",
                    step_type="avg",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="APCP_Sfc",
                    grib_element="APCP",
                    include_lead_time_suffix=True,
                    deaccumulate_to_rate=True,
                    keep_mantissa_bits=8,
                ),
            ),
            EcccHrdpsDataVar(
                name="downward_short_wave_radiation_flux_surface",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="sdswrf",
                    standard_name="surface_downwelling_shortwave_flux_in_air",
                    long_name="Surface downward short-wave radiation flux",
                    units="W m-2",
                    step_type="avg",
                    comment="Average value since the previous forecast step.",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="DSWRF_Sfc",
                    grib_element="DSWRF",
                    deaccumulate_to_rate=True,
                    deaccumulation_invalid_below_threshold_rate=RADIATION_INVALID_BELOW_THRESHOLD,
                    # JPEG2000 packing noise in the flat night-time accumulation drives clamping.
                    deaccumulation_expected_clamp_fraction=0.25,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcccHrdpsDataVar(
                name="downward_long_wave_radiation_flux_surface",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="sdlwrf",
                    standard_name="surface_downwelling_longwave_flux_in_air",
                    long_name="Surface downward long-wave radiation flux",
                    units="W m-2",
                    step_type="avg",
                    comment="Average value since the previous forecast step.",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="DLWRF_Sfc",
                    grib_element="DLWRF",
                    deaccumulate_to_rate=True,
                    deaccumulation_invalid_below_threshold_rate=RADIATION_INVALID_BELOW_THRESHOLD,
                    deaccumulation_expected_clamp_fraction=0.25,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcccHrdpsDataVar(
                name="pressure_surface",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="sp",
                    standard_name="surface_air_pressure",
                    long_name="Surface pressure",
                    units="Pa",
                    step_type="instant",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="PRES_Sfc",
                    grib_element="PRES",
                    keep_mantissa_bits=10,
                ),
            ),
            EcccHrdpsDataVar(
                name="pressure_reduced_to_mean_sea_level",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="prmsl",
                    standard_name="air_pressure_at_mean_sea_level",
                    long_name="Pressure reduced to MSL",
                    units="Pa",
                    step_type="instant",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="PRMSL_MSL",
                    grib_element="PRMSL",
                    keep_mantissa_bits=10,
                ),
            ),
            EcccHrdpsDataVar(
                name="total_cloud_cover_atmosphere",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="tcc",
                    standard_name="cloud_area_fraction",
                    long_name="Total cloud cover",
                    units="percent",
                    step_type="instant",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="TCDC_Sfc",
                    grib_element="TCDC",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcccHrdpsDataVar(
                name="snow_thickness_surface",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="sde",
                    standard_name="surface_snow_thickness",
                    long_name="Snow depth",
                    units="m",
                    step_type="instant",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="SNOD_Sfc",
                    grib_element="SNOD",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcccHrdpsDataVar(
                name="snow_water_equivalent_surface",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="sd",
                    standard_name="lwe_thickness_of_surface_snow_amount",
                    long_name="Snow depth water equivalent",
                    units="m",
                    step_type="instant",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="SDWE_Sfc",
                    grib_element="SDWE",
                    # Source GRIB is in kg m-2 (= mm lwe); convert to m
                    scale_factor=0.001,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcccHrdpsDataVar(
                name="convective_available_potential_energy_atmosphere",
                encoding=encoding,
                attrs=DataVarAttrs(
                    short_name="cape",
                    standard_name="atmosphere_convective_available_potential_energy",
                    long_name="Convective available potential energy",
                    units="J kg-1",
                    step_type="instant",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    variable_name_in_filename="CAPE_Sfc",
                    grib_element="CAPE",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
        ]

    def _spatial_info(
        self,
    ) -> tuple[
        tuple[int, int], tuple[float, float, float, float], tuple[float, float], str
    ]:
        """
        Returns (shape, bounds, resolution, crs wkt string).
        Useful for deriving x, y and latitude, longitude coordinates.
        See tests/eccc/hrdps/template_config_test.py::test_spatial_info_matches_file
        """
        return (
            (1290, 2540),
            (
                -14.832470000590822,
                -12.313751000775795,
                42.31753300059079,
                16.711251000775796,
            ),
            (0.022500001181567565, -0.02250000155159038),
            HRDPS_CRS_WKT,
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
        _, _, _, crs = self._spatial_info()
        return _latitude_longitude_grids(crs, x_coords.tobytes(), y_coords.tobytes())


# The unrotation costs ~400ms on the full HRDPS grid and every HRDPS dataset shares
# one grid, while a template build requests it once per zarr group. Callers must
# not mutate the returned (cached) arrays.
@functools.cache
def _latitude_longitude_grids(
    crs_wkt: str, x_bytes: bytes, y_bytes: bytes
) -> tuple[Array2D[np.float32], Array2D[np.float32]]:
    x_coords = np.frombuffer(x_bytes, dtype=np.float64)
    y_coords = np.frombuffer(y_bytes, dtype=np.float64)
    xs, ys = np.meshgrid(x_coords, y_coords)
    rotated_crs = CRS.from_wkt(crs_wkt)
    # The derived rotated CRS's source is the plain (unrotated) sphere lat/lon CRS.
    assert rotated_crs.source_crs is not None
    transformer = Transformer.from_crs(
        rotated_crs, rotated_crs.source_crs, always_xy=True
    )
    lons, lats = transformer.transform(xs, ys)
    # Dropping to 32 bit precision still gets us < 1 meter precision and
    # makes each array about 13MB vs 26MB for float64.
    return lats.astype(np.float32), lons.astype(np.float32)
