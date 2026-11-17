from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from reformatters.common.config_models import (
    BaseInternalAttrs,
    Coordinate,
    CoordinateAttrs,
    DataVar,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.deaccumulation import (
    DIRECT_RADIATION_INVALID_BELOW_THRESHOLD,
    RADIATION_INVALID_BELOW_THRESHOLD,
    AccumulationType,
)
from reformatters.common.types import Timedelta
from reformatters.common.zarr import BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE


class DwdIconEuInternalAttrs(BaseInternalAttrs):
    """Variable specific attributes used internally to drive processing.

    Not written to the dataset.

    Attributes:
        variable_name_in_filename (str): The name used in ICON-EU's GRIB filename for this variable.
            For example, `alb_rad` (for `surface_albedo`).
        deaccumulation_invalid_below_threshold_rate (float | None): Threshold passed through to
            `deaccumulate_to_rates_inplace` when `deaccumulate_to_rate` is True. Used, for example,
            to tolerate the larger negative noise produced by lossy-compressed radiation fields.
        deaccumulation_expected_clamp_fraction (float | None): Override for the fraction of values
            expected to be clamped to 0.
        deaccumulation_type (AccumulationType): Whether the source values are cumulative totals
            ("accumulated", default) or running-mean rates whose averaging window grows from
            forecast start ("running_mean", used for ICON-EU averaged radiation fields).
    """

    variable_name_in_filename: str
    window_reset_frequency: Timedelta | None = None
    scale_factor: float | None = None
    deaccumulation_invalid_below_threshold_rate: float | None = None
    deaccumulation_expected_clamp_fraction: float | None = None
    deaccumulation_type: AccumulationType = "accumulated"


class DwdIconEuDataVar(DataVar[DwdIconEuInternalAttrs]):
    pass


def forecast_dimension_coordinates(init_times: pd.DatetimeIndex) -> dict[str, Any]:
    return {
        "init_time": init_times,
        "lead_time": pd.timedelta_range("0h", "78h", freq="1h").union(
            pd.timedelta_range("81h", "120h", freq="3h")
        ),
    }


def derive_forecast_coordinates(
    ds: xr.Dataset,
) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
    return {
        "valid_time": ds["init_time"] + ds["lead_time"],
        "ingested_forecast_length": (
            ("init_time",),
            np.full(ds["init_time"].size, np.timedelta64("NaT", "us")),
        ),
        "expected_forecast_length": (
            ("init_time",),
            np.full(
                ds["init_time"].size,
                ds["lead_time"].max(),
                dtype="timedelta64[us]",
            ),
        ),
    }


def forecast_coordinate_configs(
    init_times: pd.DatetimeIndex,
    append_dim_coordinate_chunk_size: int,
) -> Sequence[Coordinate]:
    dim_coords = forecast_dimension_coordinates(init_times)
    return [
        Coordinate(
            name="init_time",
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
                    min=dim_coords["init_time"].min().isoformat(), max="Present"
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
                    min=init_times.min().isoformat(),
                    max="Present + 5 days",
                ),
            ),
        ),
        Coordinate(
            name="ingested_forecast_length",
            encoding=Encoding(
                dtype="float64",
                fill_value=float("nan"),
                compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                units="seconds",
                chunks=append_dim_coordinate_chunk_size,
                shards=None,
            ),
            attrs=CoordinateAttrs(
                long_name="Ingested forecast length",
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
                    min=str(dim_coords["lead_time"].min()),
                    max=str(dim_coords["lead_time"].max()),
                ),
            ),
        ),
    ]


def icon_eu_data_vars(encoding: Encoding) -> Sequence[DwdIconEuDataVar]:
    default_keep_mantissa_bits = 7

    # Some of the `comment` text is taken from the DWD Database Reference PDF:
    # https://www.dwd.de/DWD/forschung/nwv/fepub/icon_database_main.pdf
    return [
        DwdIconEuDataVar(
            name="downward_diffuse_short_wave_radiation_flux_surface",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="aswdifd_s",
                long_name="Surface diffuse short-wave radiation flux",
                units="W m-2",
                step_type="avg",
                standard_name="surface_diffuse_downwelling_shortwave_flux_in_air",
                comment="Average flux since the previous forecast step.",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="aswdifd_s",
                keep_mantissa_bits=default_keep_mantissa_bits,
                deaccumulate_to_rate=True,
                window_reset_frequency=pd.Timedelta.max,
                deaccumulation_invalid_below_threshold_rate=RADIATION_INVALID_BELOW_THRESHOLD,
                # GRIB precision jitter in the running mean clamps every dark grid cell to 0,
                # reaching ~28% of an init's values in February and more at midwinter.
                deaccumulation_expected_clamp_fraction=0.4,
                deaccumulation_type="running_mean",
            ),
        ),
        DwdIconEuDataVar(
            name="downward_direct_short_wave_radiation_flux_surface",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="aswdir_s",
                standard_name="surface_direct_downwelling_shortwave_flux_in_air",
                long_name="Surface direct short-wave radiation flux",
                units="W m-2",
                step_type="avg",
                comment="Average flux since the previous forecast step.",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="aswdir_s",
                keep_mantissa_bits=default_keep_mantissa_bits,
                deaccumulate_to_rate=True,
                window_reset_frequency=pd.Timedelta.max,
                deaccumulation_invalid_below_threshold_rate=DIRECT_RADIATION_INVALID_BELOW_THRESHOLD,
                # GRIB precision jitter in the running mean clamps every dark grid cell to 0,
                # reaching ~28% of an init's values in February and more at midwinter.
                deaccumulation_expected_clamp_fraction=0.4,
                deaccumulation_type="running_mean",
            ),
        ),
        DwdIconEuDataVar(
            name="convective_available_potential_energy_atmosphere",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="cape",
                standard_name="atmosphere_convective_available_potential_energy",
                long_name="Convective available potential energy",
                units="J kg-1",
                step_type="instant",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="cape_con",
                keep_mantissa_bits=default_keep_mantissa_bits,
            ),
        ),
        DwdIconEuDataVar(
            name="cloud_cover_high",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="hcc",
                standard_name="cloud_area_fraction_in_atmosphere_layer",
                long_name="High cloud cover",
                units="percent",
                step_type="instant",
                comment="Cloud cover (0 - 400 hPa).",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="clch",
                keep_mantissa_bits=default_keep_mantissa_bits,
            ),
        ),
        DwdIconEuDataVar(
            name="cloud_cover_low",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="lcc",
                standard_name="cloud_area_fraction_in_atmosphere_layer",
                long_name="Low cloud cover",
                units="percent",
                step_type="instant",
                comment="Cloud cover (800 hPa - surface).",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="clcl",
                keep_mantissa_bits=default_keep_mantissa_bits,
            ),
        ),
        DwdIconEuDataVar(
            name="cloud_cover_medium",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="mcc",
                standard_name="cloud_area_fraction_in_atmosphere_layer",
                long_name="Medium cloud cover",
                units="percent",
                step_type="instant",
                comment="Cloud cover (400 - 800 hPa).",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="clcm",
                keep_mantissa_bits=default_keep_mantissa_bits,
            ),
        ),
        DwdIconEuDataVar(
            name="total_cloud_cover_atmosphere",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="tcc",
                standard_name="cloud_area_fraction",
                long_name="Total cloud cover",
                units="percent",
                step_type="instant",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="clct",
                keep_mantissa_bits=default_keep_mantissa_bits,
            ),
        ),
        DwdIconEuDataVar(
            name="snow_thickness_surface",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="sde",
                long_name="Snow depth",
                standard_name="surface_snow_thickness",
                units="m",
                step_type="instant",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="h_snow",
                keep_mantissa_bits=default_keep_mantissa_bits,
            ),
        ),
        DwdIconEuDataVar(
            name="pressure_reduced_to_mean_sea_level",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="prmsl",
                standard_name="air_pressure_at_mean_sea_level",
                long_name="Pressure reduced to MSL",
                units="Pa",
                step_type="instant",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="pmsl",
                keep_mantissa_bits=10,
            ),
        ),
        DwdIconEuDataVar(
            name="relative_humidity_2m",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="2r",
                long_name="2 metre relative humidity",
                units="percent",
                step_type="instant",
                standard_name="relative_humidity",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="relhum_2m",
                keep_mantissa_bits=default_keep_mantissa_bits,
            ),
        ),
        DwdIconEuDataVar(
            name="temperature_2m",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="2t",
                long_name="2 metre temperature",
                units="degree_Celsius",
                step_type="instant",
                standard_name="air_temperature",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="t_2m",
                keep_mantissa_bits=default_keep_mantissa_bits,
            ),
        ),
        DwdIconEuDataVar(
            name="precipitation_surface",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="prate",
                standard_name="precipitation_flux",
                long_name="Precipitation rate",
                units="kg m-2 s-1",
                step_type="avg",
                comment="Average precipitation rate since the previous forecast step. Units equivalent to mm/s.",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="tot_prec",
                keep_mantissa_bits=8,
                deaccumulate_to_rate=True,
                window_reset_frequency=pd.Timedelta.max,  # accumulates over full lead time, never resetting
            ),
        ),
        DwdIconEuDataVar(
            name="wind_u_10m",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="10u",
                long_name="10 metre U wind component",
                units="m s-1",
                step_type="instant",
                standard_name="eastward_wind",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="u_10m",
                keep_mantissa_bits=default_keep_mantissa_bits,
            ),
        ),
        DwdIconEuDataVar(
            name="wind_v_10m",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="10v",
                long_name="10 metre V wind component",
                units="m s-1",
                step_type="instant",
                standard_name="northward_wind",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="v_10m",
                keep_mantissa_bits=default_keep_mantissa_bits,
            ),
        ),
        DwdIconEuDataVar(
            name="wind_gust_10m",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="10fg",
                long_name="Maximum 10 metre wind gust since previous post-processing",
                standard_name="wind_speed_of_gust",
                units="m s-1",
                step_type="max",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="vmax_10m",
                keep_mantissa_bits=default_keep_mantissa_bits,
            ),
        ),
        DwdIconEuDataVar(
            name="snow_water_equivalent_surface",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="sd",
                standard_name="lwe_thickness_of_surface_snow_amount",
                long_name="Snow depth water equivalent",
                units="m",
                step_type="instant",
                comment="Set to 0 over water surfaces and snow-free land points.",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="w_snow",
                # Source GRIB is in kg m-2 (= mm lwe); convert to m
                scale_factor=0.001,
                keep_mantissa_bits=default_keep_mantissa_bits,
            ),
        ),
        DwdIconEuDataVar(
            name="dew_point_temperature_2m",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="2d",
                standard_name="dew_point_temperature",
                long_name="2 metre dewpoint temperature",
                units="degree_Celsius",
                step_type="instant",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="td_2m",
                keep_mantissa_bits=default_keep_mantissa_bits,
            ),
        ),
        DwdIconEuDataVar(
            name="pressure_surface",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="sp",
                standard_name="surface_air_pressure",
                long_name="Surface pressure",
                units="Pa",
                step_type="instant",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="ps",
                keep_mantissa_bits=10,
            ),
        ),
        DwdIconEuDataVar(
            name="precipitable_water_atmosphere",
            encoding=encoding,
            attrs=DataVarAttrs(
                short_name="pwat",
                standard_name="atmosphere_mass_content_of_water_vapor",
                long_name="Precipitable water",
                units="kg m-2",
                step_type="instant",
            ),
            internal_attrs=DwdIconEuInternalAttrs(
                variable_name_in_filename="tqv",
                keep_mantissa_bits=default_keep_mantissa_bits,
            ),
        ),
    ]
