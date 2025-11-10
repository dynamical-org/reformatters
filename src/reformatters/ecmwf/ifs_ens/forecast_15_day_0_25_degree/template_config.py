from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import computed_field

from reformatters.common.config_models import (
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.deaccumulation import (
    PRECIPITATION_RATE_INVALID_BELOW_THRESHOLD,
    RADIATION_INVALID_BELOW_THRESHOLD,
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
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar, EcmwfInternalAttrs


class EcmwfIfsEnsForecast15Day025DegreeTemplateConfig(TemplateConfig[EcmwfDataVar]):
    dims: tuple[Dim, ...] = (
        "init_time",
        "lead_time",
        "ensemble_member",
        "latitude",
        "longitude",
    )
    append_dim: AppendDim = "init_time"
    # While this dataset has a longer history, April 2024 is when we have enough 0.25 degree resolution data to start processing, with
    # the correct number of longitude values (1440), and the majority of the variables we are interested in are available.
    append_dim_start: Timestamp = pd.Timestamp("2024-04-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("24h")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="ecmwf-ifs-ens-forecast-15-day-0-25-degree",
            dataset_version="0.1.0",
            name="ECMWF IFS ENS Forecast, 15 day, 0.25 degree",
            description="Ensemble weather forecasts from the ECMWF Integrated Forecasting System (IFS).",
            attribution="ECMWF IFS ENS Forecast data processed by dynamical.org from ECMWF Open Data.",
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
            # single control member (0) + 50 perturbed members (1-50)
            "ensemble_member": np.arange(0, 51),
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
        return {
            "valid_time": ds["init_time"] + ds["lead_time"],
            "ingested_forecast_length": (
                (self.append_dim,),
                np.full(ds[self.append_dim].size, np.timedelta64("NaT", "ns")),
            ),
            "expected_forecast_length": (
                (self.append_dim,),
                np.full(
                    ds[self.append_dim].size,
                    self.dimension_coordinates()["lead_time"].max(),
                    dtype="timedelta64[ns]",
                ),
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
                name="ensemble_member",
                encoding=Encoding(
                    dtype="int16",
                    fill_value=-1,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["ensemble_member"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units="realization",
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
                        max="Present",
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
                name="expected_forecast_length",
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
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[EcmwfDataVar]:
        """Define metadata and encoding for each data variable."""
        # Data variable chunking and sharding
        # Roughly ~17.5MB uncompressed, ~3.5MB compressed
        var_chunks: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 85,  # All lead times
            "ensemble_member": 51,  # All ensemble members
            "latitude": 32,  # 23 chunks over 721 pixels
            "longitude": 32,  # 45 chunks over 1440 pixels
        }
        # Roughly ~1775MB uncompressed, ~355MB compressed
        var_shards: dict[Dim, int] = {
            "init_time": var_chunks["init_time"],
            "lead_time": var_chunks["lead_time"],
            "ensemble_member": var_chunks["ensemble_member"],
            "latitude": var_chunks["latitude"] * 10,
            "longitude": var_chunks["longitude"] * 10,
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
            EcmwfDataVar(
                name="pressure_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sp",
                    long_name="Surface pressure",
                    units="Pa",
                    step_type="instant",
                    standard_name="surface_air_pressure",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="Pressure [Pa]",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_element="PRES",
                    grib_index_param="sp",
                    keep_mantissa_bits=11,
                ),
            ),
            EcmwfDataVar(
                name="temperature_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="t2m",
                    long_name="2 metre temperature",
                    units="C",
                    step_type="instant",
                    standard_name="air_temperature",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="Temperature [C]",
                    grib_description='2[m] HTGL="Specified height level above ground"',
                    grib_element="TMP",
                    grib_index_param="2t",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcmwfDataVar(
                name="wind_u_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="u10",
                    long_name="10 metre U wind component",
                    units="m/s",
                    step_type="instant",
                    standard_name="eastward_wind",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="u-component of wind [m/s]",
                    grib_description='10[m] HTGL="Specified height level above ground"',
                    grib_element="UGRD",
                    grib_index_param="10u",
                    keep_mantissa_bits=6,
                ),
            ),
            EcmwfDataVar(
                name="wind_v_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="v10",
                    long_name="10 metre V wind component",
                    units="m/s",
                    step_type="instant",
                    standard_name="northward_wind",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="v-component of wind [m/s]",
                    grib_description='10[m] HTGL="Specified height level above ground"',
                    grib_element="VGRD",
                    grib_index_param="10v",
                    keep_mantissa_bits=6,
                ),
            ),
            EcmwfDataVar(
                name="wind_u_100m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="u100",
                    long_name="100 metre U wind component",
                    standard_name="eastward_wind",
                    units="m/s",
                    step_type="instant",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="u-component of wind [m/s]",
                    grib_description='100[m] HTGL="Specified height level above ground"',
                    grib_element="UGRD",
                    grib_index_param="100u",
                    keep_mantissa_bits=6,
                ),
            ),
            EcmwfDataVar(
                name="wind_v_100m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="v100",
                    long_name="100 metre V wind component",
                    units="m/s",
                    step_type="instant",
                    standard_name="northward_wind",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="v-component of wind [m/s]",
                    grib_description='100[m] HTGL="Specified height level above ground"',
                    grib_element="VGRD",
                    grib_index_param="100v",
                    keep_mantissa_bits=6,
                ),
            ),
            EcmwfDataVar(
                name="precipitation_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="tp",
                    long_name="Total precipitation",
                    units="mm/s",
                    step_type="avg",
                    comment="Average precipitation rate since the previous forecast step.",
                ),
                # The metadata for precipitation surface in the grib files is not correctly populated.
                # We know that comment (prodType 0, cat 1, subcat 193) [-] is correct for precipitation surface,
                # so we use that. We have included the other set of grib metadata fields here for completeness.
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="(prodType 0, cat 1, subcat 193) [-]",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_element="unknown",
                    grib_index_param="tp",
                    deaccumulate_to_rate=True,
                    scale_factor=1000,  # The raw data is in meters so we will need to scale to mm
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    window_reset_frequency=pd.Timedelta.max,  # accumulate over the full lead time dimension, never resetting
                    deaccumulation_invalid_below_threshold_rate=PRECIPITATION_RATE_INVALID_BELOW_THRESHOLD,
                ),
            ),
            EcmwfDataVar(
                name="categorical_precipitation_type_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="ptype",
                    long_name="Categorical precipitation type at surface",
                    units="0=No precipitation; 1=Rain; 2=Thunderstorm; 3=Freezing rain; 4=Mixed/ice; 5=Snow; 6=Wet snow; 7=Mixture of rain and snow; 8=Ice pellets; 9=Graupel; 10=Hail; 11=Drizzle; 12=Freezing drizzle; 13-191=Reserved; 192-254=Reserved for local use; 255=Missing",
                    step_type="instant",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="Precipitation type [0=No precipitation; 1=Rain; 2=Thunderstorm; 3=Freezing rain; 4=Mixed/ice; 5=Snow; 6=Wet snow; 7=Mixture of rain and snow; 8=Ice pellets; 9=Graupel; 10=Hail; 11=Drizzle; 12=Freezing drizzle; 13-191=Reserved; 192-254=Reserved for local use; 255=Missing]",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_element="PTYPE",
                    grib_index_param="ptype",
                    keep_mantissa_bits="no-rounding",
                    date_available=pd.Timestamp("2024-11-13T00:00"),
                ),
            ),
            EcmwfDataVar(
                name="downward_long_wave_radiation_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sdlwrf",
                    long_name="Surface downward long-wave radiation flux",
                    units="W/(m^2)",
                    step_type="avg",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="Downward long-wave radiation flux [W/(m^2)]",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_element="DLWRF",
                    grib_index_param="strd",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    deaccumulate_to_rate=True,
                    window_reset_frequency=pd.Timedelta.max,
                    deaccumulation_invalid_below_threshold_rate=RADIATION_INVALID_BELOW_THRESHOLD,
                ),
            ),
            EcmwfDataVar(
                name="downward_short_wave_radiation_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sdswrf",
                    long_name="Surface downward short-wave radiation flux",
                    units="W/(m^2)",
                    step_type="avg",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="Downward short-wave radiation flux [W/(m^2)]",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_element="DSWRF",
                    grib_index_param="ssrd",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    deaccumulate_to_rate=True,
                    window_reset_frequency=pd.Timedelta.max,
                    deaccumulation_invalid_below_threshold_rate=RADIATION_INVALID_BELOW_THRESHOLD,
                ),
            ),
            EcmwfDataVar(
                name="pressure_reduced_to_mean_sea_level",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="prmsl",
                    long_name="Pressure reduced to MSL",
                    units="Pa",
                    step_type="instant",
                    standard_name="air_pressure_at_mean_sea_level",
                ),
                internal_attrs=EcmwfInternalAttrs(
                    grib_comment="Pressure [Pa]",
                    grib_description='0[-] MSL="Mean sea level"',
                    grib_element="PRES",
                    grib_index_param="msl",
                    keep_mantissa_bits=11,
                ),
            ),
        ]
