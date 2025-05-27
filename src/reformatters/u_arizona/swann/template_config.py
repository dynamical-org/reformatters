from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import (
    BaseInternalAttrs,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVar,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.template_config import (
    TemplateConfig,
)
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)


class SWANNInternalAttrs(BaseInternalAttrs):
    netcdf_var_name: str


class SWANNDataVar(DataVar[SWANNInternalAttrs]):
    pass


class SWANNTemplateConfig(TemplateConfig[SWANNDataVar]):
    dims: tuple[Dim, ...] = ("time", "latitude", "longitude")
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = pd.Timestamp("1981-10-01")
    append_dim_frequency: Timedelta = pd.Timedelta("1D")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="u-arizona-swann",
            dataset_version="0.1.0",
            name="University of Arizona SWANN Snow",
            description="Daily 4 km Gridded SWE and Snow Depth from Assimilated In-Situ and Modeled Data over the Conterminous US, Version 1",
            attribution=(
                "Broxton, P., X. Zeng, and N. Dawson. 2019. "
                "Daily 4 km Gridded SWE and Snow Depth from Assimilated In-Situ and Modeled Data over the Conterminous US, Version 1. "
                "Boulder, Colorado USA. NASA  National Snow and Ice Data Center Distributed Active Archive Center. "
                "https://doi.org/10.5067/0GGPB220EX6A."
            ),
            spatial_domain="Conterminous US",
            spatial_resolution="4 km",
            time_domain="1981-10-01 to Present",
            time_resolution="Daily",
            forecast_domain=None,
            forecast_resolution=None,
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        # Pixel sizes from GDAL info for the 4km SWE/Depth dataset
        # These values come from the NetCDF file's GeoTransform:
        # > gdalinfo "netcdf:UA_SWE_Depth_4km_v1_20241001_stable.nc:SWE" | grep "Pixel Size"
        # > Pixel Size = (0.041666666666667,-0.041666667692123)
        lon_pixel_size = 0.041666666666667
        lat_pixel_size = -0.041666667692123  # negative because latitude decreases

        latitude_max = 49.91666793823242
        latitude_min = 24.08333396911621
        longitude_max = -66.5
        longitude_min = -125.0

        epsilon = 1e-5

        # Generate coordinate arrays using the actual data boundaries from the NetCDF file
        # These values come from the actual data coordinates in the NetCDF file:
        # - lat: 24.08333396911621 to 49.91666793823242
        # - lon: -125.0 to -66.5
        # We use these exact values to ensure our zarr output matches the source data
        latitude = np.arange(latitude_max, latitude_min - epsilon, lat_pixel_size)
        longitude = np.arange(longitude_min, longitude_max + epsilon, lon_pixel_size)

        return {
            self.append_dim: self.append_dim_coordinates(
                self.append_dim_start + pd.Timedelta(days=1)
            ),
            "latitude": latitude,
            "longitude": longitude,
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
                        min=pd.Timestamp(self.append_dim_start).isoformat(),
                        max="Present",
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
                    # The following were obtained by running ds.rio.write_crs("EPSG:4269"), see test_get_template_spatial_ref.
                    crs_wkt='GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4269"]]',
                    semi_major_axis=6378137.0,
                    semi_minor_axis=6356752.314140356,
                    inverse_flattening=298.257222101,
                    reference_ellipsoid_name="GRS 1980",
                    longitude_of_prime_meridian=0.0,
                    prime_meridian_name="Greenwich",
                    geographic_crs_name="NAD83",
                    horizontal_datum_name="North American Datum 1983",
                    grid_mapping_name="latitude_longitude",
                    spatial_ref='GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4269"]]',
                    comment="EPSG:4269",
                ),
            ),
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[SWANNDataVar]:
        # Chunking selected to target ~1.5mb compressed chunks (we are assuming ~20% compression)
        var_chunks: dict[Dim, int] = {
            "time": 365,
            "latitude": 32,
            "longitude": 32,
        }
        # Sharding selected to target ~300mb compressed shards (we are assuming ~20% compression)
        var_shards: dict[Dim, int] = {
            "time": var_chunks["time"] * 5,
            "latitude": var_chunks["latitude"] * 15,
            "longitude": var_chunks["longitude"] * 15,
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(var_chunks[d] for d in self.dims),
            shards=tuple(var_shards[d] for d in self.dims),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        default_keep_mantissa_bits = 8

        return [
            SWANNDataVar(
                name="snow_water_equivalent",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="snow_water_equivalent",
                    long_name="Snow water equivalent",
                    standard_name="lwe_thickness_of_surface_snow_amount",
                    units="mm h20",
                    step_type="instant",
                ),
                internal_attrs=SWANNInternalAttrs(
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    netcdf_var_name="SWE",
                ),
            ),
            SWANNDataVar(
                name="snow_depth",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="snow_depth",
                    long_name="Snow depth",
                    standard_name="surface_snow_thickness",
                    units="mm snow thickness",
                    step_type="instant",
                ),
                internal_attrs=SWANNInternalAttrs(
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    netcdf_var_name="DEPTH",
                ),
            ),
        ]
