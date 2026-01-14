from collections.abc import Sequence
from pathlib import Path
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
    SPATIAL_REF_COORDS,  # noqa: F401
    TemplateConfig,
)
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_2BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)

# Constants for QA data variable
QA_NETCDF_VAR_NAME = "QA"
QA_FILL_VALUE = -32767
QA_ENCODING_FILL_VALUE = -32767


class NoaaNdviCdrInternalAttrs(BaseInternalAttrs):
    """
    Variable specific attributes used internally to drive processing.
    Not written to the dataset.
    """

    netcdf_var_name: str
    fill_value: float
    scale_factor: float | None = None
    add_offset: float | None = None
    valid_range: tuple[float, float] | None = None
    read_data_fill_value: float | None = None


class NoaaNdviCdrDataVar(DataVar[NoaaNdviCdrInternalAttrs]):
    pass


class NoaaNdviCdrAnalysisTemplateConfig(TemplateConfig[NoaaNdviCdrDataVar]):
    dims: tuple[Dim, ...] = ("time", "latitude", "longitude")
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = pd.Timestamp("1981-06-24")
    append_dim_frequency: Timedelta = pd.Timedelta("1D")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="noaa-ndvi-cdr-analysis",
            dataset_version="0.1.0",
            name="NOAA Normalized Difference Vegetation Index CDR",
            description="Daily Normalized Difference Vegetation Index (NDVI) derived from NOAA Climate Data Record (CDR) using AVHRR (1981-2014) and VIIRS (2014-present) satellite data.",
            attribution=(
                "Vermote, Eric; NOAA CDR Program. (2022): NOAA Climate Data Record (CDR) of VIIRS Normalized Difference Vegetation Index (NDVI), Version 1. "
                "Vermote, Eric; NOAA CDR Program. (2019): NOAA Climate Data Record (CDR) of AVHRR Normalized Difference Vegetation Index (NDVI), Version 5. "
                "NOAA National Centers for Environmental Information."
            ),
            spatial_domain="Global",
            spatial_resolution="0.05 degrees (~5km)",
            time_domain="1981-06-24 to Present",
            time_resolution="Daily",
            forecast_domain=None,
            forecast_resolution=None,
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        """
        Returns a dictionary of dimension names to coordinates for the dataset.
        """
        # This file was created from the lat/lon values in one of the source NetCDF files:
        #   AVHRR-Land_v005_AVH13C1_NOAA-07_19810625_c20170610042839.nc
        # We have confirmed that the lat/lon values match the VIIRS NetCDF files.
        #
        # We use values from a sample source file to set the latitude and longitude coordinates
        # because the values in the NetCDF do not have a consistent delta.
        lat_lon_values = np.load(
            Path(__file__).parent / "./source_lat_lon_coordinate_values.npz"
        )

        return {
            self.append_dim: self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "latitude": lat_lon_values["latitude"],
            "longitude": lat_lon_values["longitude"],
        }

    @computed_field  # type: ignore[prop-decorator]
    @property
    def coords(self) -> Sequence[Coordinate]:
        """Define metadata and encoding for each coordinate."""
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
                    long_name="Time",
                    standard_name="time",
                    axis="T",
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=self.append_dim_start.isoformat(),
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
                    long_name="Latitude",
                    standard_name="latitude",
                    units="degrees_north",
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
                    units="degrees_east",
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
                # The following were obtained by running ds.rio.write_crs("EPSG:4326").crs.attrs, see test_get_template_spatial_ref.
                attrs=CoordinateAttrs(
                    units=None,
                    statistics_approximate=None,
                    crs_wkt='GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]',
                    geographic_crs_name="WGS 84",
                    grid_mapping_name="latitude_longitude",
                    horizontal_datum_name="World Geodetic System 1984",
                    inverse_flattening=298.257223563,
                    longitude_of_prime_meridian=0.0,
                    prime_meridian_name="Greenwich",
                    reference_ellipsoid_name="WGS 84",
                    semi_major_axis=6378137.0,
                    semi_minor_axis=6356752.314245179,
                    spatial_ref='GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]',
                    comment="EPSG:4326 - WGS 84 Geographic Coordinate System",
                ),
            ),
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[NoaaNdviCdrDataVar]:
        """Define metadata and encoding for each data variable."""
        # Chunking selected to target ~3mb compressed chunks
        var_chunks: dict[Dim, int] = {
            "time": 365,
            "latitude": 100,
            "longitude": 100,
        }
        # Sharding selected to target ~140mb compressed shards (assuming compression to ~20%)
        # Note: We previously targeted larger shards (time * 5), but had trouble getting jobs to
        # on the larger nodes before they got evicted.
        var_shards: dict[Dim, int] = {
            "time": var_chunks["time"] * 2,
            "latitude": var_chunks["latitude"] * 5,
            "longitude": var_chunks["longitude"] * 5,
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=0,
            chunks=tuple(var_chunks[d] for d in self.dims),
            shards=tuple(var_shards[d] for d in self.dims),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        encoding_int16_default = Encoding(
            dtype="int16",
            fill_value=QA_ENCODING_FILL_VALUE,
            chunks=tuple(var_chunks[d] for d in self.dims),
            shards=tuple(var_shards[d] for d in self.dims),
            compressors=[BLOSC_2BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        default_keep_mantissa_bits = 8

        return [
            NoaaNdviCdrDataVar(
                name="ndvi_raw",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="ndvi_raw",
                    long_name="normalized_difference_vegetation_index",
                    units="unitless",
                    step_type="instant",
                    comment="Raw NDVI values from the NOAA CDR",
                ),
                internal_attrs=NoaaNdviCdrInternalAttrs(
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    netcdf_var_name="NDVI",
                    fill_value=-9999.0,
                    scale_factor=0.0001,
                    add_offset=0.0,
                    valid_range=(-1000, 10000),
                    read_data_fill_value=np.nan,
                ),
            ),
            NoaaNdviCdrDataVar(
                name="ndvi_usable",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="ndvi_usable",
                    long_name="normalized_difference_vegetation_index",
                    units="unitless",
                    step_type="instant",
                    comment="NDVI values that have been quality checked",
                ),
                internal_attrs=NoaaNdviCdrInternalAttrs(
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    netcdf_var_name="NDVI",
                    fill_value=-9999.0,
                    scale_factor=0.0001,
                    add_offset=0.0,
                    valid_range=(-1000, 10000),
                    read_data_fill_value=np.nan,
                ),
            ),
            NoaaNdviCdrDataVar(
                name="qa",
                encoding=encoding_int16_default,
                attrs=DataVarAttrs(
                    short_name="qa",
                    long_name="quality_assurance",
                    units="categorical",
                    step_type="instant",
                ),
                internal_attrs=NoaaNdviCdrInternalAttrs(
                    keep_mantissa_bits="no-rounding",
                    netcdf_var_name=QA_NETCDF_VAR_NAME,
                    fill_value=QA_FILL_VALUE,
                ),
            ),
        ]
