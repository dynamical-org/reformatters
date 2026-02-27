from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
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
from reformatters.common.template_config import SPATIAL_REF_COORDS, TemplateConfig
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)

# MRMS v12.0 launched October 14, 2020 at 20:00 UTC, introducing MultiSensor_QPE products.
# Before this date, GaugeCorr_QPE is available via Iowa Mesonet.
MRMS_V12_START = pd.Timestamp("2020-10-14T20:00")


class NoaaMrmsInternalAttrs(BaseInternalAttrs):
    mrms_product: str
    # Pre-v12 product name on Iowa Mesonet (e.g. GaugeCorr_QPE_01H for precipitation_surface)
    mrms_product_pre_v12: str | None = None
    mrms_level: str = "00.00"
    # Product only available from this time onwards; earlier times emit NaN
    available_from: Timestamp | None = None
    # For deaccumulation: MRMS hourly QPE is a 1-hour fixed window accumulation
    window_reset_frequency: Timedelta | None = None


class NoaaMrmsDataVar(DataVar[NoaaMrmsInternalAttrs]):
    pass


class NoaaMrmsConusAnalysisHourlyTemplateConfig(TemplateConfig[NoaaMrmsDataVar]):
    dims: tuple[Dim, ...] = ("time", "latitude", "longitude")
    append_dim: AppendDim = "time"
    # MRMS became operational at NCEP in September 2014.
    # Iowa Mesonet archive starts October 2014.
    append_dim_start: Timestamp = pd.Timestamp("2014-10-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("1h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="noaa-mrms-conus-analysis-hourly",
            dataset_version="0.1.0",
            name="NOAA MRMS CONUS analysis, hourly",
            description="Hourly precipitation analysis from the Multi-Radar Multi-Sensor (MRMS) system operated by NOAA NWS NCEP.",
            attribution="NOAA NWS NCEP MRMS data processed by dynamical.org from NOAA Open Data Dissemination and Iowa Mesonet archives.",
            spatial_domain="Continental United States",
            spatial_resolution="0.01 degrees (~1km)",
            time_domain=f"{self.append_dim_start} UTC to Present",
            time_resolution="1 hour",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            "time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "latitude": np.arange(54.995, 20.005 - 0.001, -0.01),
            "longitude": np.arange(-129.995, -60.005 + 0.001, 0.01),
        }

    def derive_coordinates(
        self,
        ds: xr.Dataset,  # noqa: ARG002
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        return {
            "spatial_ref": SPATIAL_REF_COORDS,
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        dim_coords = self.dimension_coordinates()
        append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()

        return [
            Coordinate(
                name="time",
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
                        min=self.append_dim_start.isoformat(), max="Present"
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
                    # MRMS uses IAU 1965 spheroid (semi_major=6378160, inv_flattening=298.25)
                    crs_wkt='GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",SPHEROID["Spheroid imported from GRIB file",6378160,298.253916296469]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]',
                    semi_major_axis=6378160.0,
                    semi_minor_axis=6356775.0,
                    inverse_flattening=298.253916296469,
                    reference_ellipsoid_name="Spheroid imported from GRIB file",
                    longitude_of_prime_meridian=0.0,
                    prime_meridian_name="Greenwich",
                    geographic_crs_name="Coordinate System imported from GRIB file",
                    horizontal_datum_name="unnamed",
                    grid_mapping_name="latitude_longitude",
                    spatial_ref='GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",SPHEROID["Spheroid imported from GRIB file",6378160,298.253916296469]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]',
                    comment="MRMS uses the IAU 1965 spheroid. Difference from WGS84 is sub-pixel at 0.01 degree resolution.",
                ),
            ),
        ]

    @computed_field
    @property
    def data_vars(self) -> Sequence[NoaaMrmsDataVar]:
        # MRMS grid: 3500 lat x 7000 lon, hourly
        # Precipitation data compresses very well (lots of zeros)
        var_chunks: dict[Dim, int] = {
            "time": 72,  # 3 days of hourly data
            "latitude": 175,  # 20 chunks over 3500 pixels
            "longitude": 175,  # 40 chunks over 7000 pixels
        }

        var_shards: dict[Dim, int] = {
            "time": 72 * 30,  # 2160 hours = 90 days
            "latitude": 175 * 4,  # 5 shards over 3500 pixels
            "longitude": 175 * 4,  # 10 shards over 7000 pixels
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(var_chunks[d] for d in self.dims),
            shards=tuple(var_shards[d] for d in self.dims),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        default_keep_mantissa_bits = 7
        # MRMS hourly QPE resets every hour (each file is an independent 1-hour accumulation)
        qpe_window_reset_frequency = pd.Timedelta("1h")

        return [
            NoaaMrmsDataVar(
                name="precipitation_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="prate",
                    standard_name="precipitation_flux",
                    long_name="Precipitation rate",
                    units="kg m-2 s-1",
                    step_type="avg",
                    comment="Average precipitation rate over the previous hour. Derived from MultiSensor_QPE_01H_Pass2 from October 2020, GaugeCorr_QPE_01H before. Units equivalent to mm/s.",
                ),
                internal_attrs=NoaaMrmsInternalAttrs(
                    mrms_product="MultiSensor_QPE_01H_Pass2",
                    mrms_product_pre_v12="GaugeCorr_QPE_01H",
                    deaccumulate_to_rate=True,
                    window_reset_frequency=qpe_window_reset_frequency,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            NoaaMrmsDataVar(
                name="precipitation_pass_1_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="prate",
                    standard_name="precipitation_flux",
                    long_name="Precipitation rate",
                    units="kg m-2 s-1",
                    step_type="avg",
                    comment="Average precipitation rate over the previous hour, pass 1 (lower latency, fewer gauges). Derived from MultiSensor_QPE_01H_Pass1. Available from October 2020 onward. Units equivalent to mm/s.",
                ),
                internal_attrs=NoaaMrmsInternalAttrs(
                    mrms_product="MultiSensor_QPE_01H_Pass1",
                    available_from=MRMS_V12_START,
                    deaccumulate_to_rate=True,
                    window_reset_frequency=qpe_window_reset_frequency,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            NoaaMrmsDataVar(
                name="precipitation_radar_only_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="rdpr",
                    standard_name="precipitation_flux",
                    long_name="Precipitation rate from radar",
                    units="kg m-2 s-1",
                    step_type="avg",
                    comment="Average radar-only precipitation rate over the previous hour (no gauge correction). Derived from RadarOnly_QPE_01H. Units equivalent to mm/s.",
                ),
                internal_attrs=NoaaMrmsInternalAttrs(
                    mrms_product="RadarOnly_QPE_01H",
                    deaccumulate_to_rate=True,
                    window_reset_frequency=qpe_window_reset_frequency,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            NoaaMrmsDataVar(
                name="categorical_precipitation_type_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="ptype",
                    long_name="Precipitation type",
                    units="1",
                    step_type="instant",
                    comment="Precipitation type flag. 0=no precipitation, 1=warm stratiform rain, 3=snow, 6=convective, 7=hail, 10=cold stratiform rain, 91=tropical.",
                ),
                internal_attrs=NoaaMrmsInternalAttrs(
                    mrms_product="PrecipFlag",
                    keep_mantissa_bits="no-rounding",
                ),
            ),
        ]
