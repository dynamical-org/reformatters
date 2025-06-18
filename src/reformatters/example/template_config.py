from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import computed_field

from reformatters.common.config_models import (
    BaseInternalAttrs,
    Coordinate,
    CoordinateAttrs,  # noqa: F401
    DatasetAttributes,
    DataVar,
    DataVarAttrs,  # noqa: F401
    Encoding,  # noqa: F401
    StatisticsApproximate,  # noqa: F401
)
from reformatters.common.template_config import (
    SPATIAL_REF_COORDS,  # noqa: F401
    TemplateConfig,
)
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,  # noqa: F401
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,  # noqa: F401
)


class ExampleInternalAttrs(BaseInternalAttrs):
    """
    Variable specific attributes used internally to drive processing.
    Not written to the dataset.
    """

    # For example,
    # grib_element: str


class ExampleDataVar(DataVar[ExampleInternalAttrs]):
    pass


class ExampleTemplateConfig(TemplateConfig[ExampleDataVar]):
    dims: tuple[Dim, ...] = ("init_time", "lead_time", "latitude", "longitude")
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = pd.Timestamp("2020-01-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        # return DatasetAttributes(
        #     dataset_id="producer-model-variant",
        #     dataset_version="0.1.0",
        #     name="Producer Model Variant",
        #     description="Weather data from the Model operated by Producer.",
        #     attribution="Producer Model Variant data processed by dynamical.org from Producer Model.",
        #     spatial_domain="Global",
        #     spatial_resolution="0.25 degrees (~20km)",
        #     time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
        #     time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:.0f} hours",
        #     forecast_domain="Forecast lead time 0-384 hours (0-16 days) ahead",
        #     forecast_resolution="Forecast step 0-120 hours: hourly, 123-384 hours: 3 hourly",
        # )
        raise NotImplementedError("Subclasses implement `dataset_attributes`")

    def dimension_coordinates(self) -> dict[str, Any]:
        """
        Returns a dictionary of dimension names to coordinates for the dataset.
        """
        # return {
        #     self.append_dim: self.append_dim_coordinates(
        #         self.append_dim_start + self.append_dim_frequency
        #     ),
        #     "lead_time": (
        #         pd.timedelta_range("0h", "120h", freq="1h").union(
        #             pd.timedelta_range("123h", "384h", freq="3h")
        #         )
        #     ),
        #     "latitude": np.flip(np.arange(-90, 90.25, 0.25)),
        #     "longitude": np.arange(-180, 180, 0.25),
        # }
        raise NotImplementedError("Subclasses implement `dimension_coordinates`")

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        """
        Return a dictionary of non-dimension coordinates for the dataset.
        Called whenever len(ds.append_dim) changes.
        """
        # Non-dimension coordinates are additional labels for data along
        # one or more dimensions. Use them to make it easier to use and
        # understand your dataset.
        # return {
        #     "valid_time": ds["init_time"] + ds["lead_time"],
        #     "ingested_forecast_length": (
        #         (self.append_dim,),
        #         np.full(ds[self.append_dim].size, np.timedelta64("NaT", "ns")),
        #     ),
        #     "spatial_ref": SPATIAL_REF_COORDS,
        # }
        raise NotImplementedError("Subclasses implement `derive_coordinates`")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def coords(self) -> Sequence[Coordinate]:
        """Define metadata and encoding for each coordinate."""
        # dim_coords = self.dimension_coordinates()
        # append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()

        # return [
        #     Coordinate(
        #         name=self.append_dim,
        #         encoding=Encoding(
        #             dtype="int64",
        #             fill_value=0,
        #             compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
        #             calendar="proleptic_gregorian",
        #             units="seconds since 1970-01-01 00:00:00",
        #             chunks=append_dim_coordinate_chunk_size,
        #             shards=None,
        #         ),
        #         attrs=CoordinateAttrs(
        #             units="seconds since 1970-01-01 00:00:00",
        #             statistics_approximate=StatisticsApproximate(
        #                 min=dim_coords[self.append_dim].min().isoformat(), max="Present"
        #             ),
        #         ),
        #     ),
        #     Coordinate(
        #         name="lead_time",
        #         encoding=Encoding(
        #             dtype="int64",
        #             fill_value=-1,
        #             compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
        #             units="seconds",
        #             chunks=len(dim_coords["lead_time"]),
        #             shards=None,
        #         ),
        #         attrs=CoordinateAttrs(
        #             units="seconds",
        #             statistics_approximate=StatisticsApproximate(
        #                 min=str(dim_coords["lead_time"].min()),
        #                 max=str(dim_coords["lead_time"].max()),
        #             ),
        #         ),
        #     ),
        #     Coordinate(
        #         name="latitude",
        #         encoding=Encoding(
        #             dtype="float64",
        #             fill_value=np.nan,
        #             compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
        #             chunks=len(dim_coords["latitude"]),
        #             shards=None,
        #         ),
        #         attrs=CoordinateAttrs(
        #             units="degrees_north",
        #             statistics_approximate=StatisticsApproximate(
        #                 min=float(dim_coords["latitude"].min()),
        #                 max=float(dim_coords["latitude"].max()),
        #             ),
        #         ),
        #     ),
        #     Coordinate(
        #         name="longitude",
        #         encoding=Encoding(
        #             dtype="float64",
        #             fill_value=np.nan,
        #             compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
        #             chunks=len(dim_coords["longitude"]),
        #             shards=None,
        #         ),
        #         attrs=CoordinateAttrs(
        #             units="degrees_east",
        #             statistics_approximate=StatisticsApproximate(
        #                 min=float(dim_coords["longitude"].min()),
        #                 max=float(dim_coords["longitude"].max()),
        #             ),
        #         ),
        #     ),
        #     Coordinate(
        #         name="valid_time",
        #         encoding=Encoding(
        #             dtype="int64",
        #             fill_value=0,
        #             compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
        #             calendar="proleptic_gregorian",
        #             units="seconds since 1970-01-01 00:00:00",
        #             chunks=(
        #                 append_dim_coordinate_chunk_size,
        #                 len(dim_coords["lead_time"]),
        #             ),
        #             shards=None,
        #         ),
        #         attrs=CoordinateAttrs(
        #             units="seconds since 1970-01-01 00:00:00",
        #             statistics_approximate=StatisticsApproximate(
        #                 min=self.append_dim_start.isoformat(),
        #                 max="Present + 16 days",
        #             ),
        #         ),
        #     ),
        #     Coordinate(
        #         name="ingested_forecast_length",
        #         encoding=Encoding(
        #             dtype="int64",
        #             fill_value=-1,
        #             compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
        #             units="seconds",
        #             chunks=append_dim_coordinate_chunk_size,
        #             shards=None,
        #         ),
        #         attrs=CoordinateAttrs(
        #             units="seconds",
        #             statistics_approximate=StatisticsApproximate(
        #                 min=str(dim_coords["lead_time"].min()),
        #                 max=str(dim_coords["lead_time"].max()),
        #             ),
        #         ),
        #     ),
        #     Coordinate(
        #         name="spatial_ref",
        #         encoding=Encoding(
        #             dtype="int64",
        #             fill_value=0,
        #             chunks=(),  # Scalar coordinate
        #             shards=None,
        #         ),
        #         attrs=CoordinateAttrs(
        #             units=None,
        #             statistics_approximate=None,
        #             # Deterived by running `ds.rio.write_crs("+proj=longlat +a=6371229 +b=6371229 +no_defs +type=crs")["spatial_ref"].attrs
        #             crs_wkt='GEOGCS["unknown",DATUM["unknown",SPHEROID["unknown",6371229,0]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Longitude",EAST],AXIS["Latitude",NORTH]]',
        #             semi_major_axis=6371229.0,
        #             semi_minor_axis=6371229.0,
        #             inverse_flattening=0.0,
        #             reference_ellipsoid_name="unknown",
        #             longitude_of_prime_meridian=0.0,
        #             prime_meridian_name="Greenwich",
        #             geographic_crs_name="unknown",
        #             horizontal_datum_name="unknown",
        #             grid_mapping_name="latitude_longitude",
        #             spatial_ref='GEOGCS["unknown",DATUM["unknown",SPHEROID["unknown",6371229,0]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Longitude",EAST],AXIS["Latitude",NORTH]]',
        #             comment="This coordinate reference system matches the source data which follows WMO conventions of assuming the earth is a perfect sphere with a radius of 6,371,229m. It is similar to EPSG:4326, but EPSG:4326 uses a more accurate representation of the earth's shape.",
        #         ),
        #     ),
        # ]
        raise NotImplementedError("Subclasses implement `coords`")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[ExampleDataVar]:
        """Define metadata and encoding for each data variable."""
        # # Data variable chunking and sharding
        # #
        # # Aim for one of these roughly equivalent quantities:
        # # 1-2mb chunks compressed
        # # 4-8mb uncompressed
        # # 4-8 million float32 values
        # var_chunks: dict[Dim, int] = {
        #     "init_time": 1,
        #     "lead_time": 105,
        #     "latitude": 121,
        #     "longitude": 121,
        # }
        # # Aim for one of these roughly equivalent quantities:
        # # 64-256MB shards compressed
        # # 256-1024MB uncompressed
        # # 256 million to 1 billion float32 values
        # var_shards: dict[Dim, int] = {
        #     "init_time": 1,
        #     "lead_time": 105 * 2,
        #     "latitude": 121 * 6,
        #     "longitude": 121 * 6,
        # }

        # encoding_float32_default = Encoding(
        #     dtype="float32",
        #     fill_value=np.nan,
        #     chunks=tuple(var_chunks[d] for d in self.dims),
        #     shards=tuple(var_shards[d] for d in self.dims),
        #     compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        # )

        # default_keep_mantissa_bits = 7

        # # return [
        #     ExampleDataVar(
        #         name="temperature_2m",
        #         encoding=encoding_float32_default,
        #         attrs=DataVarAttrs(
        #             short_name="t2m",
        #             long_name="2 metre temperature",
        #             units="C",
        #             step_type="instant",
        #             standard_name="air_temperature",
        #         ),
        #         internal_attrs=ExampleInternalAttrs(
        #             grib_element="TMP",
        #             grib_comment='2[m] HTGL="Specified height level above ground"',
        #             grib_index_level="2 m above ground",
        #             index_position=580,
        #             keep_mantissa_bits=default_keep_mantissa_bits,
        #         ),
        #     ),
        #     ExampleDataVar(
        #         name="precipitation_surface",
        #         encoding=encoding_float32_default,
        #         attrs=DataVarAttrs(
        #             short_name="tp",
        #             long_name="Total Precipitation",
        #             units="mm/s",
        #             comment="Average precipitation rate since the previous forecast step.",
        #             step_type="avg",
        #         ),
        #         internal_attrs=ExampleInternalAttrs(
        #             grib_element="APCP",
        #             grib_comment='0[-] SFC="Ground or water surface"',
        #             grib_index_level="surface",
        #             index_position=595,
        #             include_lead_time_suffix=True,
        #             deaccumulate_to_rate=True,
        #             window_reset_frequency=pd.Timedelta("6h"),
        #             keep_mantissa_bits=default_keep_mantissa_bits,
        #         ),
        #     ),
        #     ExampleDataVar(
        #         name="pressure_surface",
        #         encoding=encoding_float32_default,
        #         attrs=DataVarAttrs(
        #             short_name="sp",
        #             long_name="Surface pressure",
        #             units="Pa",
        #             step_type="instant",
        #             standard_name="surface_air_pressure",
        #         ),
        #         internal_attrs=ExampleInternalAttrs(
        #             grib_element="PRES",
        #             grib_comment='0[-] SFC="Ground or water surface"',
        #             grib_index_level="surface",
        #             index_position=560,
        #             keep_mantissa_bits=10,
        #         ),
        #     ),
        #     ExampleDataVar(
        #         name="categorical_snow_surface",
        #         encoding=encoding_float32_default,
        #         attrs=DataVarAttrs(
        #             short_name="csnow",
        #             long_name="Categorical snow",
        #             units="0=no; 1=yes",
        #             step_type="avg",
        #         ),
        #         internal_attrs=ExampleInternalAttrs(
        #             grib_element="CSNOW",
        #             grib_comment='0[-] SFC="Ground or water surface"',
        #             grib_index_level="surface",
        #             index_position=604,
        #             window_reset_frequency=pd.Timedelta("6h"),
        #             keep_mantissa_bits="no-rounding",
        #         ),
        #     ),
        #     ExampleDataVar(
        #         name="total_cloud_cover_atmosphere",
        #         encoding=encoding_float32_default,
        #         attrs=DataVarAttrs(
        #             short_name="tcc",
        #             long_name="Total Cloud Cover",
        #             units="%",
        #             step_type="avg",
        #         ),
        #         internal_attrs=ExampleInternalAttrs(
        #             grib_element="TCDC",
        #             grib_comment='0[-] EATM="Entire Atmosphere"',
        #             grib_index_level="entire atmosphere",
        #             index_position=635,
        #             window_reset_frequency=pd.Timedelta("6h"),
        #             keep_mantissa_bits=default_keep_mantissa_bits,
        #         ),
        #     ),
        # ]
        raise NotImplementedError("Subclasses implement `data_vars`")
