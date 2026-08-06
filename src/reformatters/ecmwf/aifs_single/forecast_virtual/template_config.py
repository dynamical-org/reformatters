from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from gribberish.zarr import GribberishCodec
from pydantic import computed_field
from zarr.codecs import ScaleOffset

from reformatters.common.config_models import (
    ROOT,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVarAttrs,
    Encoding,
    Group,
    StatisticsApproximate,
)
from reformatters.common.template_config import (
    SPATIAL_REF_COORDS,
    TemplateConfig,
)
from reformatters.common.types import (
    AppendDim,
    CodecConfig,
    Dims,
    Timedelta,
    Timestamp,
)
from reformatters.common.zarr import BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar, EcmwfInternalAttrs

_GRID_NLAT = 721
_GRID_NLON = 1440

# Descending like HRRR's pressure_level group; 10 hPa exists only from
# AIFS_2026_UPGRADE_DATE (q never has it).
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]

# The open-data format change: the source path segment switches aifs/ -> aifs-single/,
# many variables are added, and tp/cp switch from metres to kg m-2.
AIFS_SINGLE_FORMAT_CHANGE_DATE = pd.Timestamp("2025-02-26T00:00")
# Model upgrade adding fscov and the 10 hPa pressure level.
AIFS_2026_UPGRADE_DATE = pd.Timestamp("2026-05-13T00:00")

# ScaleOffset decodes on read as value / scale + offset. Temperatures are served in
# degree_Celsius to match the materialized ecmwf-aifs-single-forecast; geopotential
# (m2 s-2) is divided by standard gravity to serve geopotential height in metres,
# matching the materialized geopotential_height_* variables.
_KELVIN_TO_CELSIUS = ScaleOffset(offset=-273.15, scale=1.0).to_dict()
_GEOPOTENTIAL_TO_HEIGHT = ScaleOffset(offset=0.0, scale=9.80665).to_dict()
# Params whose source values are Kelvin; their vars default to the K->C filter.
_CELSIUS_PARAMS = frozenset({"2t", "2d", "skt", "sot", "t"})

_STATIC_COMMENT = "Time-invariant field published at lead time 0 only."


class EcmwfAifsSingleVirtualInternalAttrs(EcmwfInternalAttrs):
    # The source publishes this variable only at lead time 0 (static fields).
    lead_0_only: bool = False


class EcmwfAifsSingleVirtualDataVar(EcmwfDataVar):
    internal_attrs: EcmwfAifsSingleVirtualInternalAttrs


class EcmwfAifsSingleForecastVirtualTemplateConfig(
    TemplateConfig[EcmwfAifsSingleVirtualDataVar]
):
    """Virtual, spatially-chunked (map-optimized) ECMWF AIFS Single forecast template.

    Chunks are references to GRIB messages in ECMWF's open data archive decoded at
    read time, so the grid is the native 0.25 degree grid with one chunk per message.
    Covers every source variable: single-level and soil variables at the root plus a
    pressure_level group. See docs/virtual_datasets.md.
    """

    dims: Dims = {
        ROOT: ("init_time", "lead_time", "latitude", "longitude"),
        "pressure_level": (
            "init_time",
            "lead_time",
            "latitude",
            "longitude",
            "pressure_level",
        ),
    }
    append_dim: AppendDim = "init_time"
    # Earlier AIFS data (2024-02-29 to 2024-03-31) uses a reduced Gaussian grid.
    append_dim_start: Timestamp = pd.Timestamp("2024-04-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="ecmwf-aifs-single-forecast-virtual",
            dataset_version="0.1.0",
            name="ECMWF AIFS Single forecast, virtual",
            description="Weather forecasts from the ECMWF Artificial Intelligence Forecasting System (AIFS) Single model.",
            attribution="ECMWF AIFS Single forecast data processed by dynamical.org from ECMWF Open Data.",
            license="CC-BY-4.0",
            spatial_domain="Global",
            spatial_resolution="0.25 degrees (~20km)",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:.0f} hours",
            forecast_domain="Forecast lead time 0-360 hours (0-15 days) ahead",
            forecast_resolution="6 hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            self.append_dim: self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "lead_time": pd.timedelta_range("0h", "360h", freq="6h"),
            "latitude": np.flip(np.arange(-90, 90.25, 0.25)),
            "longitude": np.arange(-180, 180, 0.25),
            "pressure_level": np.array(PRESSURE_LEVELS, dtype=np.int64),
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        return {
            "valid_time": ds["init_time"] + ds["lead_time"],
            "expected_forecast_length": (
                (self.append_dim,),
                np.full(
                    ds[self.append_dim].size,
                    self.dimension_coordinates()["lead_time"].max(),
                    dtype="timedelta64[us]",
                ),
            ),
            "spatial_ref": SPATIAL_REF_COORDS,
        }

    @computed_field
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
                    long_name="Forecast initialization time",
                    standard_name="forecast_reference_time",
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=dim_coords[self.append_dim].min().isoformat(), max="Present"
                    ),
                ),
            ),
            Coordinate(
                name="lead_time",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=float("nan"),
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=len(dim_coords["lead_time"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Forecast lead time",
                    standard_name="forecast_period",
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
                name="pressure_level",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=-1,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["pressure_level"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Pressure level",
                    standard_name="air_pressure",
                    units="hPa",
                    axis="Z",
                    positive="down",
                    statistics_approximate=StatisticsApproximate(
                        min=int(dim_coords["pressure_level"].min()),
                        max=int(dim_coords["pressure_level"].max()),
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
                    long_name="Valid time",
                    standard_name="time",
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=self.append_dim_start.isoformat(),
                        max="Present + 15 days",
                    ),
                ),
            ),
            Coordinate(
                name="expected_forecast_length",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=float("nan"),
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Expected forecast length",
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].max()),
                        max=str(dim_coords["lead_time"].max()),
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

    @computed_field
    @property
    def data_vars(self) -> Sequence[EcmwfAifsSingleVirtualDataVar]:
        return [*_root_data_vars(), *_pressure_data_vars()]


def _virtual_encoding(
    index_param: str, group: Group, filters: Sequence[CodecConfig]
) -> Encoding:
    """One chunk per GRIB message: chunk 1 along init_time/lead_time/pressure_level,
    full latitude/longitude, no shards, no compressors. GribberishCodec decodes the
    raw message and any array->array filters (K->C, geopotential->height) chain on
    read."""
    if group is ROOT:
        chunks: tuple[int, ...] = (1, 1, _GRID_NLAT, _GRID_NLON)
    else:
        chunks = (1, 1, _GRID_NLAT, _GRID_NLON, 1)
    return Encoding(
        # GribberishCodec decodes to float64 natively; declaring float64 avoids a cast.
        dtype="float64",
        fill_value=np.nan,
        chunks=chunks,
        shards=None,
        compressors=(),
        filters=filters,
        serializer=GribberishCodec(
            var=index_param, adjust_longitude_range=True, north_up=True
        ).to_dict(),
    )


def _var(
    name: str,
    *,
    param: str,
    level_type: str,
    level_value: float,
    group: Group,
    step_type: str,
    short_name: str,
    long_name: str,
    units: str,
    standard_name: str | None,
    comment: str | None,
    date_available: Timestamp | None,
    lead_0_only: bool,
    filters: Sequence[CodecConfig] | None,
) -> EcmwfAifsSingleVirtualDataVar:
    # Default to the K->C filter for temperature params; a var may override with
    # explicit filters (e.g. geopotential -> height).
    resolved_filters: Sequence[CodecConfig] = (
        filters
        if filters is not None
        else ([_KELVIN_TO_CELSIUS] if param in _CELSIUS_PARAMS else ())
    )
    return EcmwfAifsSingleVirtualDataVar(
        name=name,
        group=group,
        encoding=_virtual_encoding(param, group, resolved_filters),
        attrs=DataVarAttrs(
            short_name=short_name,
            long_name=long_name,
            units=units,
            standard_name=standard_name,
            step_type=step_type,  # ty: ignore[invalid-argument-type]
            comment=comment,
        ),
        internal_attrs=EcmwfAifsSingleVirtualInternalAttrs(
            grib_index_param=param,
            grib_index_level_type=level_type,  # ty: ignore[invalid-argument-type]
            grib_index_level_value=level_value,
            date_available=date_available,
            lead_0_only=lead_0_only,
            # Virtual chunks are never rewritten, so no rounding; the remaining
            # required fields drive materialized GRIB reads and are unused here.
            keep_mantissa_bits="no-rounding",
            grib_comment="",
            grib_element="",
            grib_description="",
        ),
    )


def _root_var(
    name: str,
    *,
    param: str,
    level_type: str = "sfc",
    level_value: float = float("nan"),
    step_type: str = "instant",
    short_name: str,
    long_name: str,
    units: str,
    standard_name: str | None = None,
    comment: str | None = None,
    date_available: Timestamp | None = None,
    lead_0_only: bool = False,
    filters: Sequence[CodecConfig] | None = None,
) -> EcmwfAifsSingleVirtualDataVar:
    return _var(
        name,
        param=param,
        level_type=level_type,
        level_value=level_value,
        group=ROOT,
        step_type=step_type,
        short_name=short_name,
        long_name=long_name,
        units=units,
        standard_name=standard_name,
        comment=comment,
        date_available=date_available,
        lead_0_only=lead_0_only,
        filters=filters,
    )


def _pressure_var(
    name: str,
    *,
    param: str,
    short_name: str,
    long_name: str,
    units: str,
    standard_name: str | None = None,
    comment: str | None = None,
    filters: Sequence[CodecConfig] | None = None,
) -> EcmwfAifsSingleVirtualDataVar:
    return _var(
        name,
        param=param,
        level_type="pl",
        level_value=float("nan"),
        group="pressure_level",
        step_type="instant",
        short_name=short_name,
        long_name=long_name,
        units=units,
        standard_name=standard_name,
        comment=comment,
        date_available=None,
        lead_0_only=False,
        filters=filters,
    )


def _root_data_vars() -> list[EcmwfAifsSingleVirtualDataVar]:
    return [
        _root_var(
            "pressure_surface",
            param="sp",
            short_name="sp",
            long_name="Surface pressure",
            units="Pa",
            standard_name="surface_air_pressure",
        ),
        _root_var(
            "pressure_reduced_to_mean_sea_level",
            param="msl",
            short_name="prmsl",
            long_name="Pressure reduced to MSL",
            units="Pa",
            standard_name="air_pressure_at_mean_sea_level",
        ),
        _root_var(
            "temperature_2m",
            param="2t",
            short_name="2t",
            long_name="2 metre temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        _root_var(
            "dew_point_temperature_2m",
            param="2d",
            short_name="2d",
            long_name="2 metre dewpoint temperature",
            units="degree_Celsius",
            standard_name="dew_point_temperature",
        ),
        _root_var(
            "skin_temperature_surface",
            param="skt",
            short_name="skt",
            long_name="Skin temperature",
            units="degree_Celsius",
            standard_name="surface_temperature",
        ),
        _root_var(
            "wind_u_10m",
            param="10u",
            short_name="10u",
            long_name="10 metre U wind component",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        _root_var(
            "wind_v_10m",
            param="10v",
            short_name="10v",
            long_name="10 metre V wind component",
            units="m s-1",
            standard_name="northward_wind",
        ),
        _root_var(
            "wind_u_100m",
            param="100u",
            short_name="100u",
            long_name="100 metre U wind component",
            units="m s-1",
            standard_name="eastward_wind",
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
        ),
        _root_var(
            "wind_v_100m",
            param="100v",
            short_name="100v",
            long_name="100 metre V wind component",
            units="m s-1",
            standard_name="northward_wind",
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
        ),
        # AIFS publishes cloud cover already in percent, so it needs no scaling filter
        # (unlike IFS ENS, whose 0-1 fraction is scaled by 100).
        _root_var(
            "total_cloud_cover_atmosphere",
            param="tcc",
            short_name="tcc",
            long_name="Total cloud cover",
            units="percent",
            standard_name="cloud_area_fraction",
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
        ),
        _root_var(
            "low_cloud_cover",
            param="lcc",
            short_name="lcc",
            long_name="Low cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
        ),
        _root_var(
            "medium_cloud_cover",
            param="mcc",
            short_name="mcc",
            long_name="Medium cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
        ),
        _root_var(
            "high_cloud_cover",
            param="hcc",
            short_name="hcc",
            long_name="High cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
        ),
        _root_var(
            "total_column_water_atmosphere",
            param="tcw",
            short_name="tcw",
            long_name="Total column water",
            units="kg m-2",
            standard_name="atmosphere_mass_content_of_water",
        ),
        _root_var(
            "snow_area_fraction_surface",
            param="fscov",
            # short/long names match the equivalent HRRR variable; the source
            # parameter is ECMWF fscov "Fraction of snow cover" (paramId 260289).
            short_name="snowc",
            long_name="Snow cover",
            units="1",
            standard_name="surface_snow_area_fraction",
            date_available=AIFS_2026_UPGRADE_DATE,
        ),
        # tp/cp exist before AIFS_SINGLE_FORMAT_CHANGE_DATE too, but in metres;
        # date_available serves only the unit-consistent kg m-2 era.
        _root_var(
            "total_precipitation_run_total_surface",
            param="tp",
            step_type="accum",
            short_name="tp",
            long_name="Total precipitation",
            units="kg m-2",
            standard_name="precipitation_amount",
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
        ),
        _root_var(
            "convective_precipitation_run_total_surface",
            param="cp",
            step_type="accum",
            short_name="cp",
            long_name="Convective precipitation",
            units="kg m-2",
            standard_name="convective_precipitation_amount",
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
        ),
        _root_var(
            "snowfall_water_equivalent_run_total_surface",
            param="sf",
            step_type="accum",
            short_name="sf",
            long_name="Snowfall water equivalent",
            units="kg m-2",
            standard_name="snowfall_amount",
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
        ),
        _root_var(
            "runoff_water_equivalent_run_total_surface",
            param="rowe",
            step_type="accum",
            short_name="rowe",
            long_name="Runoff water equivalent (surface plus subsurface)",
            units="kg m-2",
            standard_name="runoff_amount",
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
        ),
        _root_var(
            "downward_short_wave_radiation_run_total_surface",
            param="ssrd",
            step_type="accum",
            short_name="ssrd",
            long_name="Surface short-wave (solar) radiation downwards",
            units="W s m-2",
            standard_name="integral_wrt_time_of_surface_downwelling_shortwave_flux_in_air",
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
        ),
        _root_var(
            "downward_long_wave_radiation_run_total_surface",
            param="strd",
            step_type="accum",
            short_name="strd",
            long_name="Surface long-wave (thermal) radiation downwards",
            units="W s m-2",
            standard_name="integral_wrt_time_of_surface_downwelling_longwave_flux_in_air",
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
        ),
        _root_var(
            "soil_temperature_layer_1",
            param="sot",
            level_type="sol",
            level_value=1,
            short_name="sot",
            long_name="Soil temperature",
            units="degree_Celsius",
            standard_name="soil_temperature",
            comment="ECMWF soil level 1, the uppermost soil layer.",
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
        ),
        _root_var(
            "soil_temperature_layer_2",
            param="sot",
            level_type="sol",
            level_value=2,
            short_name="sot",
            long_name="Soil temperature",
            units="degree_Celsius",
            standard_name="soil_temperature",
            comment="ECMWF soil level 2, the second soil layer from the surface.",
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
        ),
        _root_var(
            "volumetric_soil_moisture_layer_1",
            param="vsw",
            level_type="sol",
            level_value=1,
            short_name="vsw",
            long_name="Volumetric soil moisture",
            units="1",
            standard_name="volume_fraction_of_condensed_water_in_soil",
            comment="ECMWF soil level 1, the uppermost soil layer.",
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
        ),
        _root_var(
            "volumetric_soil_moisture_layer_2",
            param="vsw",
            level_type="sol",
            level_value=2,
            short_name="vsw",
            long_name="Volumetric soil moisture",
            units="1",
            standard_name="volume_fraction_of_condensed_water_in_soil",
            comment="ECMWF soil level 2, the second soil layer from the surface.",
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
        ),
        _root_var(
            "land_sea_mask_surface",
            param="lsm",
            short_name="lsm",
            long_name="Land-sea mask",
            units="1",
            standard_name="land_area_fraction",
            comment=f"Fraction (0-1) of the grid box that is land. {_STATIC_COMMENT}",
            lead_0_only=True,
        ),
        _root_var(
            "geopotential_height_surface",
            param="z",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            comment=f"Surface (orography) geopotential height. {_STATIC_COMMENT}",
            lead_0_only=True,
            filters=[_GEOPOTENTIAL_TO_HEIGHT],
        ),
        _root_var(
            "standard_deviation_of_sub_gridscale_orography_surface",
            param="sdor",
            short_name="sdor",
            long_name="Standard deviation of sub-gridscale orography",
            units="m",
            comment=_STATIC_COMMENT,
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
            lead_0_only=True,
        ),
        _root_var(
            "slope_of_sub_gridscale_orography_surface",
            param="slor",
            short_name="slor",
            long_name="Slope of sub-gridscale orography",
            units="1",
            comment=_STATIC_COMMENT,
            date_available=AIFS_SINGLE_FORMAT_CHANGE_DATE,
            lead_0_only=True,
        ),
    ]


def _pressure_data_vars() -> list[EcmwfAifsSingleVirtualDataVar]:
    return [
        _pressure_var(
            "geopotential_height",
            param="z",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            filters=[_GEOPOTENTIAL_TO_HEIGHT],
        ),
        _pressure_var(
            "temperature",
            param="t",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        _pressure_var(
            "wind_u",
            param="u",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        _pressure_var(
            "wind_v",
            param="v",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        _pressure_var(
            "vertical_velocity",
            param="w",
            short_name="w",
            long_name="Vertical velocity",
            units="Pa s-1",
            standard_name="lagrangian_tendency_of_air_pressure",
        ),
        _pressure_var(
            "specific_humidity",
            param="q",
            short_name="q",
            long_name="Specific humidity",
            units="1",
            standard_name="specific_humidity",
            comment="The source provides no 10 hPa level for this variable; that level is always NaN.",
        ),
    ]
