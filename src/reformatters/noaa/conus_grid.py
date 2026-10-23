from collections.abc import Sequence
from typing import Generic, TypeVar

import numpy as np
from pydantic import computed_field

from reformatters.common.config_models import (
    Coordinate,
    CoordinateAttrs,
    DataVar,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.projection import latitude_longitude_grids, y_x_coordinates
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import Array1D, Array2D
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)

DATA_VAR = TypeVar("DATA_VAR", bound=DataVar)


class NoaaConusTemplateConfig(TemplateConfig[DATA_VAR], Generic[DATA_VAR]):
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
                    units="degree_north",
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
                    units="degree_east",
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
        return y_x_coordinates(shape, bounds, resolution)

    def _latitude_longitude_coordinates(
        self, x_coords: Array1D[np.float64], y_coords: Array1D[np.float64]
    ) -> tuple[Array2D[np.float32], Array2D[np.float32]]:
        _, _, _, crs = self._spatial_info()
        return latitude_longitude_grids(crs, x_coords, y_coords)
