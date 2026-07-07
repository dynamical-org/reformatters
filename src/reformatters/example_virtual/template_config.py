from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import computed_field

from reformatters.common.config_models import (
    ROOT,
    BaseInternalAttrs,
    Coordinate,
    CoordinateAttrs,  # noqa: F401
    DatasetAttributes,
    DataVar,
    DataVarAttrs,  # noqa: F401
    Encoding,  # noqa: F401
    Group,
    StatisticsApproximate,  # noqa: F401
)
from reformatters.common.template_config import (
    SPATIAL_REF_COORDS,  # noqa: F401
    TemplateConfig,
)
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,  # noqa: F401
)

# A virtual dataset's chunks are *references* to messages in source files (decoded
# at read time), not bytes we rechunk and rewrite. That changes only the DATA
# VARIABLE encoding (see `data_vars` below); the coordinates are small arrays we
# materialize normally, so this template's coords are identical to a materialized
# dataset's. The per-variable serializer that decodes a raw GRIB message lives in
# the data var encoding, e.g.:
# from gribberish.zarr import GribberishCodec


class ExampleInternalAttrs(BaseInternalAttrs):
    """
    Variable specific attributes used internally to drive processing.
    Not written to the dataset.
    """

    # The serializer needs to know which GRIB message to decode out of each file.
    # For example,
    # grib_element: str


class ExampleDataVar(DataVar[ExampleInternalAttrs]):
    pass


class ExampleSpatialTemplateConfig(TemplateConfig[ExampleDataVar]):
    # Single-level dataset: all vars live at the root. To add a vertical group, add an
    # entry whose key is the group/dimension name and whose dims are the root dims plus
    # that dimension, then set group=... on the group's DataVars (their zarr path becomes
    # "<group>/<name>"); each ref then routes to its group array by var.path. E.g.:
    #   "pressure_level": ("init_time", "lead_time", "latitude", "longitude", "pressure_level"),
    # See tests/common/virtual_multi_group_test.py for a worked multi-group virtual dataset.
    dims: dict[Group, tuple[Dim, ...]] = {
        ROOT: ("init_time", "lead_time", "latitude", "longitude")
    }
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = pd.Timestamp("2020-01-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        # return DatasetAttributes(
        #     dataset_id="producer-model-variant",
        #     dataset_version="0.1.0",
        #     name="Producer Model Variant",
        #     description="Weather data from the Model operated by Producer.",
        #     attribution="Producer Model Variant data processed by dynamical.org from Producer Model.",
        #     license="CC-BY-4.0",
        #     spatial_domain="Global",
        #     spatial_resolution="0.25 degrees (~20km)",
        #     time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
        #     time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:.0f} hours",
        #     forecast_domain="Forecast lead time 0-240 hours (0-10 days) ahead",
        #     forecast_resolution="Forecast step 3 hourly",
        # )
        raise NotImplementedError("Subclasses implement `dataset_attributes`")

    def dimension_coordinates(self) -> dict[str, Any]:
        """
        Returns a dictionary of dimension names to coordinates for the dataset.
        """
        # Virtual chunks decode the raw source message, so the grid here must be the
        # source file's NATIVE grid - one chunk per message means no regridding is
        # possible. (A materialized dataset should also align with the native grid unless
        # a different output grid meaningfully improves usability.) The GribberishCodec
        # `north_up=True` option (see `data_vars`) makes every decoded message north-first
        # (row 0 = largest latitude), matching GDAL's automatic flip baked into our
        # materialized datasets, so order latitude/y descending here.
        # return {
        #     self.append_dim: self.append_dim_coordinates(
        #         self.append_dim_start + self.append_dim_frequency
        #     ),
        #     "lead_time": pd.timedelta_range("0h", "240h", freq="3h"),
        #     "latitude": np.flip(np.arange(-90, 90.25, 0.25)),
        #     "longitude": np.arange(0, 360, 0.25),
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
        #     "expected_forecast_length": (
        #         ("init_time",),
        #         np.full(ds["init_time"].size, np.timedelta64(240, "h")),
        #     ),
        #     "spatial_ref": SPATIAL_REF_COORDS,
        # }
        raise NotImplementedError("Subclasses implement `derive_coordinates`")

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        """Define metadata and encoding for each coordinate.

        Coordinates are small materialized arrays - their encoding is the same as in
        a materialized dataset. Only the data variable encoding differs for a virtual
        dataset (see `data_vars`).
        """
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
        #             long_name="Forecast initialization time",
        #             standard_name="forecast_reference_time",
        #             units="seconds since 1970-01-01 00:00:00",
        #             statistics_approximate=StatisticsApproximate(
        #                 min=dim_coords[self.append_dim].min().isoformat(), max="Present"
        #             ),
        #         ),
        #     ),
        #     Coordinate(
        #         name="lead_time",
        #         encoding=Encoding(
        #             dtype="float64",
        #             fill_value=float("nan"),
        #             compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
        #             units="seconds",
        #             chunks=len(dim_coords["lead_time"]),
        #             shards=None,
        #         ),
        #         attrs=CoordinateAttrs(
        #             long_name="Forecast lead time",
        #             standard_name="forecast_period",
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
        #             long_name="Latitude",
        #             standard_name="latitude",
        #             units="degree_north",
        #             axis="Y",
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
        #             long_name="Longitude",
        #             standard_name="longitude",
        #             units="degree_east",
        #             axis="X",
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
        #             long_name="Valid time",
        #             standard_name="time",
        #             units="seconds since 1970-01-01 00:00:00",
        #             statistics_approximate=StatisticsApproximate(
        #                 min=self.append_dim_start.isoformat(),
        #                 max="Present + 10 days",
        #             ),
        #         ),
        #     ),
        #     Coordinate(
        #         name="expected_forecast_length",
        #         encoding=Encoding(
        #             dtype="float64",
        #             fill_value=float("nan"),
        #             compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
        #             units="seconds",
        #             chunks=append_dim_coordinate_chunk_size,
        #             shards=None,
        #         ),
        #         attrs=CoordinateAttrs(
        #             long_name="Expected forecast length",
        #             units="seconds",
        #             statistics_approximate=StatisticsApproximate(
        #                 min=str(dim_coords["lead_time"].max()),
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
        #             # Derived by running `ds.rio.write_crs("+proj=longlat +a=6371229 +b=6371229 +no_defs +type=crs")["spatial_ref"].attrs
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

    @computed_field
    @property
    def data_vars(self) -> Sequence[ExampleDataVar]:
        """Define metadata and encoding for each data variable.

        This is THE method that diverges from a materialized dataset. Two virtual rules:

        1. One chunk per source message. Each virtual chunk is exactly one GRIB message,
           so chunk size is 1 along every per-message dim (init_time, lead_time, and
           ensemble_member if present) and full-width along the message's spatial dims,
           with shards=None. The geometry of these chunks is what the write loop uses to
           place each reference, so it must match the source file layout exactly.

        2. We never re-encode the bytes for storage (no compressors of our own); a
           per-variable `serializer` (a zarr v3 ArrayBytesCodec, e.g. GribberishCodec)
           decodes the raw GRIB message at read time. Declare `dtype` as whatever that
           codec produces (GribberishCodec decodes to float64) to avoid a cast.

        Served values are otherwise the RAW source values. A transform the materialized
        pipeline applies splits two ways here:
        - Pointwise (per-cell) conversions - e.g. K -> degC, a unit rescale - CAN be done
          on read by chaining a zarr ScaleOffset array->array codec in `filters`; keep the
          materialized variable's name/units so it stays a drop-in.
        - Cross-chunk transforms (deaccumulating precip to a rate, temporal differencing)
          cannot be done on read - serve the raw quantity under its own name/units instead.
        """
        # dim_coords = self.dimension_coordinates()
        #
        # # One chunk per GRIB message: 1 along init_time and lead_time, full spatial extent.
        # message_chunks: tuple[int, ...] = (
        #     1,  # init_time
        #     1,  # lead_time
        #     len(dim_coords["latitude"]),
        #     len(dim_coords["longitude"]),
        # )
        #
        # virtual_encoding = Encoding(
        #     dtype="float64",  # GribberishCodec decodes to float64 natively
        #     fill_value=np.nan,
        #     chunks=message_chunks,
        #     shards=None,
        #     compressors=(),  # no compression of our own; bytes stay as the source wrote them
        #     filters=(),  # or e.g. (ScaleOffset(offset=-273.15, scale=1.0).to_dict(),) to serve degC
        #     # north_up=True flips each message north-first (row 0 = largest latitude);
        #     # set it on every GribberishCodec so all our datasets share one orientation.
        #     serializer=GribberishCodec(var="TMP", north_up=True).to_dict(),
        # )

        # return [
        #     ExampleDataVar(
        #         name="temperature_2m",
        #         encoding=virtual_encoding,
        #         attrs=DataVarAttrs(
        #             short_name="2t",
        #             long_name="2 metre temperature",
        #             # Raw GRIB temperature is Kelvin; the materialized dataset converts
        #             # to degree_Celsius on read, but a virtual chunk serves it untouched.
        #             units="K",
        #             step_type="instant",
        #             standard_name="air_temperature",
        #         ),
        #         internal_attrs=ExampleInternalAttrs(
        #             grib_element="TMP",
        #         ),
        #     ),
        #     ExampleDataVar(
        #         name="pressure_surface",
        #         encoding=virtual_encoding,
        #         attrs=DataVarAttrs(
        #             short_name="sp",
        #             long_name="Surface pressure",
        #             units="Pa",
        #             step_type="instant",
        #             standard_name="surface_air_pressure",
        #         ),
        #         internal_attrs=ExampleInternalAttrs(
        #             grib_element="PRES",
        #         ),
        #     ),
        # ]
        raise NotImplementedError("Subclasses implement `data_vars`")
