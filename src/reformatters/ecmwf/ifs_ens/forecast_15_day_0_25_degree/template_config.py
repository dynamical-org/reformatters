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
from reformatters.common.template_config import (
    SPATIAL_REF_COORDS,
    TemplateConfig,
)
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)


class EcmwfIfsEnsInternalAttrs(BaseInternalAttrs):
    """
    Variable specific attributes used internally to drive processing.
    Not written to the dataset.
    """

    # TODO (skipping this for now, will come back and add when developing processing for real)
    # grib_band_index: int
    # grib_element_name: str
    # grib_description: str
    # grib_index_level: str


class EcmwfIfsEnsDataVar(DataVar[EcmwfIfsEnsInternalAttrs]):
    pass


class EcmwfIfsEnsForecast15Day025DegreeTemplateConfig(
    TemplateConfig[EcmwfIfsEnsDataVar]
):
    dims: tuple[Dim, ...] = (
        "init_time",
        "lead_time",
        "ensemble_member",
        "latitude",
        "longitude",
    )
    append_dim: AppendDim = "init_time"
    # forecasts available from same s3 bucket since 2023-01-18, but only with 0.4deg resolution from dataset start through 2024-01-31.
    append_dim_start: Timestamp = pd.Timestamp("2024-02-01T00:00")
    # starting with just 0z forecasts for now. 12z also available with same forecast length; 6 & 18z also available with 144hr max lead time.
    append_dim_frequency: Timedelta = pd.Timedelta("24h")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="ecmwf-ifs-ens-forecast-15-day-0-25-degree",
            dataset_version="0.1.0",
            name="ECMWF IFS Ensemble (ENS) Forecast, 15 day, 0.25 degree",
            description="Ensemble weather forecasts from the ECMWF Integrated Forecasting System (IFS) - 15 day forecasts, 0.25 degree resolution.",
            attribution="ECMWF IFS Ensemble Forecast data processed by dynamical.org from ECMWF Open Data.",  # TODO correct all this
            spatial_domain="Global",
            spatial_resolution="0.25 degrees (~20km)",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:.0f} hours",
            forecast_domain="Forecast lead time 0-360 hours (0-15 days) ahead",
            forecast_resolution="Forecast step 0-144 hours: 3 hourly, 144-360 hours: 6 hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        """
        Returns a dictionary of dimension names to coordinates for the dataset.
        """
        return {
            self.append_dim: self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "lead_time": (
                pd.timedelta_range("0h", "145h", freq="3h").union(
                    pd.timedelta_range("150h", "361h", freq="6h")
                )
            ),
            "ensemble_member": np.arange(1, 51),
            "latitude": np.flip(np.arange(-90, 90.25, 0.25)),
            "longitude": np.arange(-180, 180, 0.25),
        }

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
        return {
            "valid_time": ds["init_time"] + ds["lead_time"],
            "ingested_forecast_length": (
                (self.append_dim,),
                np.full(ds[self.append_dim].size, np.timedelta64("NaT", "ns")),
            ),
            # "expected_forecast_length": (
            #     (self.append_dim,),
            #     np.full(
            #         ds[self.append_dim].size,
            #         ds["lead_time"].max(),
            #         dtype="timedelta64[ns]",
            #     ),
            # ),
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
                name="ensemble_member",
                encoding=Encoding(
                    dtype="int32",
                    fill_value=-1,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["ensemble_member"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units="realization",  # TODO what does this mean lol I stole it from gefs
                    statistics_approximate=StatisticsApproximate(
                        min=int(dim_coords["ensemble_member"].min()),
                        max=int(dim_coords["ensemble_member"].max()),
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
                        max="Present + 15 days",
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
                    # Derived by running `ds.rio.write_crs("+proj=longlat +a=6371229 +b=6371229 +no_defs +type=crs")["spatial_ref"].attrs
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
            # TODO: add expected forecast length?
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[EcmwfIfsEnsDataVar]:
        """Define metadata and encoding for each data variable."""
        # Data variable chunking and sharding
        #
        # Aim for one of these roughly equivalent quantities:
        # 1-2mb chunks compressed
        # 4-8mb uncompressed
        # 4-8 million float32 values
        # TODO check the math on these chunks & shards being reasonable. reference slack notes
        var_chunks: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 61,  # Updated for ECMWF 61-step forecast # TODO should be 83? 85?
            "ensemble_member": 50,  # All ensemble members
            "latitude": 144,  # ~721/5 for reasonable chunk size
            "longitude": 144,  # ~1440/10 for reasonable chunk size
        }
        # Aim for one of these roughly equivalent quantities:
        # 64-256MB shards compressed
        # 256-1024MB uncompressed
        # 256 million to 1 billion float32 values
        var_shards: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 61 * 2,  # TODO should be 83/85 * 2??
            "ensemble_member": 50,
            "latitude": 144 * 5,
            "longitude": 144 * 5,
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
            EcmwfIfsEnsDataVar(
                name="temperature_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="t2m",
                    long_name="2 metre temperature",
                    units="K",
                    step_type="instant",
                    standard_name="air_temperature",
                ),
                internal_attrs=EcmwfIfsEnsInternalAttrs(
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcmwfIfsEnsDataVar(
                name="wind_u_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="u10",
                    long_name="10 metre U wind component",
                    units="m s**-1",
                    step_type="instant",
                    standard_name="eastward_wind",
                ),
                internal_attrs=EcmwfIfsEnsInternalAttrs(
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcmwfIfsEnsDataVar(
                name="wind_v_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="v10",
                    long_name="10 metre V wind component",
                    units="m s**-1",
                    step_type="instant",
                    standard_name="northward_wind",
                ),
                internal_attrs=EcmwfIfsEnsInternalAttrs(
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcmwfIfsEnsDataVar(
                name="precipitation_total",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="tp",
                    long_name="Total precipitation",
                    units="m",
                    step_type="accum",
                    comment="Accumulated precipitation since forecast start time.",
                ),
                internal_attrs=EcmwfIfsEnsInternalAttrs(
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
        ]


"""
Band 39 Block=1440x1 Type=Float64, ColorInterp=Undefined
  Description = 0[-] SFC="Ground or water surface"
  Metadata:
    GRIB_COMMENT=Total precipitation rate [kg/(m^2*s)]
    GRIB_DISCIPLINE=0(Meteorological)
    GRIB_ELEMENT=TPRATE
    GRIB_FORECAST_SECONDS=0
    GRIB_IDS=CENTER=98(ECMWF) SUBCENTER=0 MASTER_TABLE=34 LOCAL_TABLE=0 SIGNF_REF_TIME=1(Start_of_Forecast) REF_TIME=2025-03-01T00:00:00Z PROD_STATUS=0(Operational) TYPE=4(Perturbed_forecast)
    GRIB_PDS_PDTN=1
    GRIB_PDS_TEMPLATE_ASSEMBLED_VALUES=1 52 4 255 158 0 0 1 0 1 0 0 255 -127 -2147483647 255 34 51
    GRIB_PDS_TEMPLATE_NUMBERS=1 52 4 255 158 0 0 0 1 0 0 0 0 1 0 0 0 0 0 255 255 255 255 255 255 255 34 51
    GRIB_REF_TIME=1740787200
    GRIB_SHORT_NAME=0-SFC
    GRIB_UNIT=[kg/(m^2*s)]
    GRIB_VALID_TIME=1740787200


Band 6165 Block=1440x1 Type=Float64, ColorInterp=Undefined
  Description = 0[-] SFC="Ground or water surface"
  Metadata:
    GRIB_COMMENT=Total precipitation rate [kg/(m^2*s)]
    GRIB_DISCIPLINE=0(Meteorological)
    GRIB_ELEMENT=TPRATE
    GRIB_FORECAST_SECONDS=0
    GRIB_IDS=CENTER=98(ECMWF) SUBCENTER=0 MASTER_TABLE=34 LOCAL_TABLE=0 SIGNF_REF_TIME=1(Start_of_Forecast) REF_TIME=2025-03-01T00:00:00Z PROD_STATUS=0(Operational) TYPE=4(Perturbed_forecast)
    GRIB_PDS_PDTN=1
    GRIB_PDS_TEMPLATE_ASSEMBLED_VALUES=1 52 4 255 158 0 0 1 0 1 0 0 255 -127 -2147483647 255 46 51
    GRIB_PDS_TEMPLATE_NUMBERS=1 52 4 255 158 0 0 0 1 0 0 0 0 1 0 0 0 0 0 255 255 255 255 255 255 255 46 51
    GRIB_REF_TIME=1740787200
    GRIB_SHORT_NAME=0-SFC
    GRIB_UNIT=[kg/(m^2*s)]
    GRIB_VALID_TIME=1740787200

"""
