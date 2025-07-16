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


class DwdIconEuInternalAttrs(BaseInternalAttrs):
    """Variable specific attributes used internally to drive processing.

    Not written to the dataset.
    """

    grib_element: str
    grib_description: str


class DwdIconEuDataVar(DataVar[DwdIconEuInternalAttrs]):
    pass


class DwdIconEuForecastTemplateConfig(TemplateConfig[DwdIconEuDataVar]):
    dims: tuple[Dim, ...] = ("init_time", "lead_time", "latitude", "longitude")
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = pd.Timestamp("2025-08-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="dwd-icon_eu-forecast",
            dataset_version="0.1.0",
            name="DWD ICON-EU Forecast",
            description="High-resolution weather forecasts for Europe from the ICON-EU model operated by Deutscher Wetterdienst (DWD).",
            attribution="DWD ICON-EU data processed by dynamical.org from DWD.",
            spatial_domain="Europe",
            spatial_resolution="0.0625 degrees (~7km)",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:.0f} hours",
            forecast_domain="Forecast lead time 0-120 hours (0-5 days) ahead",
            forecast_resolution="Forecast step 0-78 hours: hourly, 81-120 hours: 3 hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        """Returns a dictionary of dimension names to coordinates for the
        dataset."""
        return {
            self.append_dim: self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "lead_time": (
                pd.timedelta_range("0h", "78h", freq="1h").union(
                    pd.timedelta_range("81h", "120h", freq="3h")
                )
            ),
            # TODO: Continue checking Gemini's output from here (downwards):
            "latitude": np.linspace(70.5, 29.5, 657),
            "longitude": np.linspace(-23.5, 62.5, 1377),
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        """Return a dictionary of non-dimension coordinates for the dataset.

        Called whenever len(ds.append_dim) changes.
        """
        return {
            "valid_time": ds["init_time"] + ds["lead_time"],
            "ingested_forecast_length": (
                (self.append_dim,),
                np.full(ds[self.append_dim].size, np.timedelta64("NaT", "ns")),
            ),
            "spatial_ref": SPATIAL_REF_COORDS,
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
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=dim_coords[self.append_dim].min().isoformat(), max="Present"
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
                    shards=None,
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
                name="valid_time",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    calendar="proleptic_gregorian",
                    units="seconds since 1970-01-01 00:00:00",
                    chunks=(
                        append_dim_coordinate_chunk_size,
                        len(dim_coords["lead_time"]),
                    ),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=self.append_dim_start.isoformat(),
                        max="Present + 5 days",
                    ),
                ),
            ),
            Coordinate(
                name="ingested_forecast_length",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=-1,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
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
                    # Deterived by running `ds.rio.write_crs("+proj=longlat +a=6371229 +b=6371229 +no_defs +type=crs")["spatial_ref"].attrs
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[DwdIconEuDataVar]:
        """Define metadata and encoding for each data variable."""
        # Data variable chunking and sharding
        #
        # Aim for one of these roughly equivalent quantities:
        # 1-2mb chunks compressed
        # 4-8mb uncompressed
        # 4-8 million float32 values
        var_chunks: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 121,
            "latitude": 73,
            "longitude": 153,
        }
        # Aim for one of these roughly equivalent quantities:
        # 64-256MB shards compressed
        # 256-1024MB uncompressed
        # 256 million to 1 billion float32 values
        var_shards: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 121,
            "latitude": 657,
            "longitude": 1377,
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(var_chunks[d] for d in self.dims),
            shards=tuple(var_shards[d] for d in self.dims),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        default_keep_mantissa_bits = 7

        return [
            DwdIconEuDataVar(
                name="alb_rad",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="Surface albedo",
                    units="%",
                    step_type="avg",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="ALB_RAD",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="aswdifd_s",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="Surface downward diffuse short-wave radiation",
                    units="W m**-2",
                    step_type="avg",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="ASWDIFD_S",
                    deaccumulate_to_rate=True,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="aswdir_s",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="Surface downward direct short-wave radiation",
                    units="W m**-2",
                    step_type="avg",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="ASWDIR_S",
                    deaccumulate_to_rate=True,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="cape_con",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="Convective Available Potential Energy",
                    units="J kg**-1",
                    step_type="instant",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="CAPE_CON",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="clch",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="High cloud cover",
                    units="%",
                    step_type="instant",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="CLCH",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="clcl",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="Low cloud cover",
                    units="%",
                    step_type="instant",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="CLCL",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="clcm",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="Medium cloud cover",
                    units="%",
                    step_type="instant",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="CLCM",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="clct",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="Total Cloud Cover",
                    units="%",
                    step_type="avg",
                    standard_name="cloud_area_fraction",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="CLCT",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="h_snow",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="Snow depth",
                    units="m",
                    step_type="instant",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="H_SNOW",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="pmsl",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="Mean sea level pressure",
                    units="Pa",
                    step_type="instant",
                    standard_name="air_pressure_at_mean_sea_level",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="PMSL",
                    keep_mantissa_bits=10,
                ),
            ),
            DwdIconEuDataVar(
                name="relhum_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="2m Relative Humidity",
                    units="%",
                    step_type="instant",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="RELHUM_2M",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="runoff_g",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="Grid-scale runoff",
                    units="kg m**-2",
                    step_type="accum",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="RUNOFF_G",
                    deaccumulate_to_rate=True,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="t_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="2 metre temperature",
                    units="C",
                    step_type="instant",
                    standard_name="air_temperature",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="T_2M",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="tot_prec",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="Total Precipitation",
                    units="mm/s",
                    comment="Average precipitation rate since the previous forecast step.",
                    step_type="avg",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="TOT_PREC",
                    deaccumulate_to_rate=True,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="u_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="10 metre U wind component",
                    units="m s**-1",
                    step_type="instant",
                    standard_name="eastward_wind",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="U_10M",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="v_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="10 metre V wind component",
                    units="m s**-1",
                    step_type="instant",
                    standard_name="northward_wind",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="V_10M",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="vmax_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="10m Wind Gust Speed",
                    units="m s**-1",
                    step_type="max",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="VMAX_10M",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="w_snow",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    long_name="Water equivalent of snow depth",
                    units="kg m**-2",
                    step_type="instant",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    grib_element="W_SNOW",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
        ]
