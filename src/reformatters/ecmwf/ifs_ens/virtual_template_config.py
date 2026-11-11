import functools
from collections.abc import Sequence
from typing import Any, Literal, assert_never

import numpy as np
import pandas as pd
from gribberish.zarr import GribberishCodec
from pydantic import computed_field
from zarr.codecs import ScaleOffset

from reformatters.common.config_models import (
    ROOT,
    Coordinate,
    CoordinateAttrs,
    DataVarAttrs,
    Encoding,
    Group,
    StatisticsApproximate,
)
from reformatters.common.types import CodecConfig, Dims, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.ecmwf.ecmwf_config_models import (
    EcmwfDataVar,
    EcmwfInternalAttrs,
    _resolve_grib_index_param,
)
from reformatters.ecmwf.ecmwf_template_config import (
    Ecmwf025DegreeForecastTemplateConfig,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.source_file_coord import (
    IFS_CYCLE_50R1_CUTOVER,
)

IFS_ENS_025_DEGREE_START = pd.Timestamp("2024-04-01T00")
IFS_CYCLE_49R1_CUTOVER = pd.Timestamp("2024-11-12T06")
PRESSURE_LEVEL_EXPANSION = pd.Timestamp("2025-01-14T06")
PRESSURE_LEVELS = (10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
EARLY_PRESSURE_LEVELS = (50, 200, 250, 300, 500, 700, 850, 925, 1000)


def available_pressure_levels(init_time: Timestamp) -> tuple[int, ...]:
    if init_time < PRESSURE_LEVEL_EXPANSION:
        return EARLY_PRESSURE_LEVELS
    if init_time < IFS_CYCLE_50R1_CUTOVER:
        return PRESSURE_LEVELS[1:]
    return PRESSURE_LEVELS


class EcmwfIfsEnsVirtualInternalAttrs(EcmwfInternalAttrs):
    date_unavailable: Timestamp | None = None
    legacy_soil_param: str | None = None
    lead_0_only: bool = False
    control_only: bool = False


class EcmwfIfsEnsVirtualDataVar(EcmwfDataVar):
    internal_attrs: EcmwfIfsEnsVirtualInternalAttrs

    def available(self, init_time: Timestamp, lead_time: pd.Timedelta) -> bool:
        attrs = self.internal_attrs
        return (
            (attrs.date_available is None or init_time >= attrs.date_available)
            and (attrs.date_unavailable is None or init_time < attrs.date_unavailable)
            and (lead_time == pd.Timedelta(0) or not attrs.lead_0_only)
            and (lead_time != pd.Timedelta(0) or self.has_hour_0_values())
        )

    def source_selector(
        self, init_time: Timestamp, lead_time: pd.Timedelta
    ) -> tuple[str, str, int | None]:
        attrs = self.internal_attrs
        if attrs.legacy_soil_param is not None and init_time < IFS_CYCLE_49R1_CUTOVER:
            return attrs.legacy_soil_param, "sfc", None
        param = _resolve_grib_index_param(
            self, lead_time
        ).internal_attrs.grib_index_param
        value = attrs.grib_index_level_value
        return (
            param,
            attrs.grib_index_level_type,
            None if pd.isna(value) else int(value),
        )


class EcmwfIfsEnsForecastVirtualTemplateConfig(
    Ecmwf025DegreeForecastTemplateConfig[EcmwfIfsEnsVirtualDataVar]
):
    resolution: Literal["0p25", "0p1"]
    forecast_days: Literal[6, 15]
    append_dim_start: Timestamp = IFS_ENS_025_DEGREE_START
    dims: Dims = {
        ROOT: ("init_time", "lead_time", "ensemble_member", "latitude", "longitude"),
        "pressure_level": (
            "init_time",
            "lead_time",
            "ensemble_member",
            "latitude",
            "longitude",
            "pressure_level",
        ),
    }

    def dimension_coordinates(self) -> dict[str, Any]:
        match self.resolution:
            case "0p25":
                latitude = np.flip(np.arange(-90, 90.25, 0.25))
                longitude = np.arange(-180, 180, 0.25)
            case "0p1":
                latitude = np.arange(900, -901, -1) / 10
                longitude = np.arange(-1800, 1800) / 10
            case unexpected:
                assert_never(unexpected)
        match self.forecast_days:
            case 6:
                lead_time = pd.timedelta_range("0h", "144h", freq="3h")
            case 15:
                lead_time = pd.timedelta_range("0h", "144h", freq="3h").union(
                    pd.timedelta_range("150h", "360h", freq="6h")
                )
            case unexpected:
                assert_never(unexpected)
        return {
            self.append_dim: self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "lead_time": lead_time,
            "ensemble_member": np.arange(51),
            "latitude": latitude,
            "longitude": longitude,
            "pressure_level": np.array(PRESSURE_LEVELS, dtype=np.int64),
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        return [
            *super().coords,
            Coordinate(
                name="ensemble_member",
                encoding=Encoding(
                    dtype="int16",
                    fill_value=-1,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=51,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Ensemble member",
                    standard_name="realization",
                    units="1",
                    statistics_approximate=StatisticsApproximate(min=0, max=50),
                ),
            ),
            Coordinate(
                name="pressure_level",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=-1,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(PRESSURE_LEVELS),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Pressure level",
                    standard_name="air_pressure",
                    units="hPa",
                    axis="Z",
                    positive="down",
                    statistics_approximate=StatisticsApproximate(min=10, max=1000),
                ),
            ),
        ]

    @computed_field
    @property
    def data_vars(self) -> Sequence[EcmwfIfsEnsVirtualDataVar]:
        match self.resolution:
            case "0p25":
                grid = (721, 1440)
            case "0p1":
                grid = (1801, 3600)
            case unexpected:
                assert_never(unexpected)
        return [
            var
            for var in _data_vars(grid)
            if var.internal_attrs.date_unavailable is None
            or var.internal_attrs.date_unavailable > self.append_dim_start
        ]


def _var(
    name: str,
    *,
    param: str,
    long_name: str,
    units: str,
    grid: tuple[int, int],
    short_name: str | None = None,
    standard_name: str | None = None,
    group: Group = ROOT,
    step_type: Literal["instant", "accum", "max", "min"] = "instant",
    comment: str | None = None,
    date_available: Timestamp | None = None,
    date_unavailable: Timestamp | None = None,
    level_value: float = float("nan"),
    legacy_soil_param: str | None = None,
    lead_0_only: bool = False,
    control_only: bool = False,
    filters: Sequence[CodecConfig] | None = None,
    lead_time_overrides: tuple[tuple[pd.Timedelta, pd.Timedelta, str], ...] = (),
    flag_values: tuple[int, ...] | None = None,
    flag_meanings: str | None = None,
) -> EcmwfIfsEnsVirtualDataVar:
    chunks = (1, 1, 1, *grid) + (() if group is ROOT else (1,))
    if filters is None:
        filters = (
            [ScaleOffset(offset=-273.15, scale=1).to_dict()]
            if units == "degree_Celsius"
            else ()
        )
    comments = [comment] if comment else []
    if step_type == "accum":
        comments.append("Accumulated since forecast initialization.")
    if step_type in ("max", "min"):
        comments.append("Not available at lead time 0.")
    if control_only:
        comments.append("Available for control member 0 only; members 1-50 are NaN.")
    if lead_0_only:
        comments.append("Time-invariant field published at lead time 0 only.")
    return EcmwfIfsEnsVirtualDataVar(
        name=name,
        group=group,
        encoding=Encoding(
            dtype="float64",
            fill_value=np.nan,
            chunks=chunks,
            shards=None,
            compressors=(),
            filters=filters,
            serializer=GribberishCodec(
                var=param, adjust_longitude_range=True, north_up=True
            ).to_dict(),
        ),
        attrs=DataVarAttrs(
            short_name=short_name or param,
            long_name=long_name,
            units=units,
            standard_name=standard_name,
            step_type=step_type,
            comment=" ".join(comments) or None,
            flag_values=flag_values,
            flag_meanings=flag_meanings,
        ),
        internal_attrs=EcmwfIfsEnsVirtualInternalAttrs(
            grib_index_param=param,
            grib_index_level_type="pl"
            if group is not ROOT
            else "sol"
            if legacy_soil_param
            else "sfc",
            grib_index_level_value=level_value,
            grib_index_param_lead_time_overrides=lead_time_overrides,
            date_available=date_available,
            date_unavailable=date_unavailable,
            legacy_soil_param=legacy_soil_param,
            lead_0_only=lead_0_only,
            control_only=control_only,
            keep_mantissa_bits="no-rounding",
            grib_comment="",
            grib_element="",
            grib_description="",
        ),
    )


def _data_vars(grid: tuple[int, int]) -> list[EcmwfIfsEnsVirtualDataVar]:
    var = functools.partial(_var, grid=grid)
    return [
        var(
            "pressure_surface",
            param="sp",
            short_name="sp",
            long_name="Surface pressure",
            units="Pa",
            standard_name="surface_air_pressure",
        ),
        var(
            "pressure_reduced_to_mean_sea_level",
            param="msl",
            short_name="prmsl",
            long_name="Pressure reduced to MSL",
            units="Pa",
            standard_name="air_pressure_at_mean_sea_level",
        ),
        var(
            "temperature_2m",
            param="2t",
            short_name="2t",
            long_name="2 metre temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        var(
            "dew_point_temperature_2m",
            param="2d",
            short_name="2d",
            long_name="2 metre dewpoint temperature",
            units="degree_Celsius",
            standard_name="dew_point_temperature",
        ),
        var(
            "skin_temperature_surface",
            param="skt",
            short_name="skt",
            long_name="Skin temperature",
            units="degree_Celsius",
            standard_name="surface_temperature",
        ),
        var(
            "wind_u_10m",
            param="10u",
            short_name="10u",
            long_name="10 metre U wind component",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        var(
            "wind_v_10m",
            param="10v",
            short_name="10v",
            long_name="10 metre V wind component",
            units="m s-1",
            standard_name="northward_wind",
        ),
        var(
            "wind_u_100m",
            param="100u",
            short_name="100u",
            long_name="100 metre U wind component",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        var(
            "wind_v_100m",
            param="100v",
            short_name="100v",
            long_name="100 metre V wind component",
            units="m s-1",
            standard_name="northward_wind",
        ),
        var(
            "total_cloud_cover_atmosphere",
            param="tcc",
            short_name="tcc",
            long_name="Total cloud cover",
            units="percent",
            standard_name="cloud_area_fraction",
            date_available=pd.Timestamp("2025-11-20T12:00:00"),
            filters=[ScaleOffset(offset=0, scale=0.01).to_dict()],
        ),
        var(
            "low_cloud_cover",
            param="lcc",
            short_name="lcc",
            long_name="Low cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
            date_available=pd.Timestamp("2026-10-20T06:00:00"),
            filters=[ScaleOffset(offset=0, scale=0.01).to_dict()],
        ),
        var(
            "medium_cloud_cover",
            param="mcc",
            short_name="mcc",
            long_name="Medium cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
            date_available=pd.Timestamp("2026-10-20T06:00:00"),
            filters=[ScaleOffset(offset=0, scale=0.01).to_dict()],
        ),
        var(
            "high_cloud_cover",
            param="hcc",
            short_name="hcc",
            long_name="High cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
            date_available=pd.Timestamp("2026-10-20T06:00:00"),
            filters=[ScaleOffset(offset=0, scale=0.01).to_dict()],
        ),
        var(
            "total_column_water_atmosphere",
            param="tcw",
            short_name="tcw",
            long_name="Total column water",
            units="kg m-2",
            standard_name="atmosphere_mass_content_of_water",
            date_available=pd.Timestamp("2024-10-24T12:00:00"),
        ),
        var(
            "total_precipitation_run_total_surface",
            param="tp",
            short_name="tp",
            long_name="Total precipitation",
            units="kg m-2",
            standard_name="precipitation_amount",
            step_type="accum",
            filters=[ScaleOffset(offset=0, scale=0.001).to_dict()],
        ),
        var(
            "convective_precipitation_run_total_surface",
            param="cp",
            short_name="cp",
            long_name="Convective precipitation",
            units="kg m-2",
            standard_name="convective_precipitation_amount",
            step_type="accum",
            filters=[ScaleOffset(offset=0, scale=0.001).to_dict()],
            date_available=pd.Timestamp("2026-10-20T06:00:00"),
        ),
        var(
            "snowfall_water_equivalent_run_total_surface",
            param="sf",
            short_name="sf",
            long_name="Snowfall water equivalent",
            units="kg m-2",
            standard_name="snowfall_amount",
            step_type="accum",
            filters=[ScaleOffset(offset=0, scale=0.001).to_dict()],
            date_available=pd.Timestamp("2025-11-20T12:00:00"),
        ),
        var(
            "runoff_water_equivalent_run_total_surface",
            param="ro",
            short_name="rowe",
            long_name="Runoff water equivalent (surface plus subsurface)",
            units="kg m-2",
            standard_name="runoff_amount",
            step_type="accum",
            filters=[ScaleOffset(offset=0, scale=0.001).to_dict()],
        ),
        var(
            "downward_short_wave_radiation_run_total_surface",
            param="ssrd",
            short_name="ssrd",
            long_name="Surface short-wave (solar) radiation downwards",
            units="W s m-2",
            standard_name="integral_wrt_time_of_surface_downwelling_shortwave_flux_in_air",
            step_type="accum",
        ),
        var(
            "downward_long_wave_radiation_run_total_surface",
            param="strd",
            short_name="strd",
            long_name="Surface long-wave (thermal) radiation downwards",
            units="W s m-2",
            standard_name="integral_wrt_time_of_surface_downwelling_longwave_flux_in_air",
            step_type="accum",
        ),
        var(
            "soil_temperature_layer_1",
            param="sot",
            short_name="sot",
            long_name="Soil temperature",
            units="degree_Celsius",
            standard_name="soil_temperature",
            level_value=1,
            legacy_soil_param="st",
            comment="ECMWF soil level 1, the uppermost soil layer. Over water this is the sea surface temperature, not a soil temperature.",
        ),
        var(
            "soil_temperature_layer_2",
            param="sot",
            short_name="sot",
            long_name="Soil temperature",
            units="degree_Celsius",
            standard_name="soil_temperature",
            level_value=2,
            legacy_soil_param="stl2",
            comment="ECMWF soil level 2, the second soil layer from the surface. Over water this is the sea surface temperature, not a soil temperature.",
        ),
        var(
            "volumetric_soil_moisture_layer_1",
            param="vsw",
            short_name="vsw",
            long_name="Volumetric soil moisture",
            units="1",
            standard_name="volume_fraction_of_condensed_water_in_soil",
            level_value=1,
            legacy_soil_param="swvl1",
            comment="ECMWF soil level 1, the uppermost soil layer. Applies over land only; over water the values are near zero, so mask with land_sea_mask_surface.",
        ),
        var(
            "volumetric_soil_moisture_layer_2",
            param="vsw",
            short_name="vsw",
            long_name="Volumetric soil moisture",
            units="1",
            standard_name="volume_fraction_of_condensed_water_in_soil",
            level_value=2,
            legacy_soil_param="swvl2",
            comment="ECMWF soil level 2, the second soil layer from the surface. Applies over land only; over water the values are near zero, so mask with land_sea_mask_surface.",
        ),
        var(
            "land_sea_mask_surface",
            param="lsm",
            short_name="lsm",
            long_name="Land-sea mask",
            units="1",
            standard_name="land_area_fraction",
            comment="Fraction (0-1) of the grid box that is land.",
        ),
        var(
            "geopotential_height_surface",
            param="z",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            date_available=IFS_CYCLE_50R1_CUTOVER,
            control_only=True,
            lead_0_only=True,
            comment="Surface (orography) geopotential height.",
            filters=[ScaleOffset(offset=0, scale=9.80665).to_dict()],
        ),
        var(
            "standard_deviation_of_sub_gridscale_orography_surface",
            param="sdor",
            short_name="sdor",
            long_name="Standard deviation of sub-gridscale orography",
            units="m",
            date_available=IFS_CYCLE_50R1_CUTOVER,
            control_only=True,
            lead_0_only=True,
        ),
        var(
            "slope_of_sub_gridscale_orography_surface",
            param="slor",
            short_name="slor",
            long_name="Slope of sub-gridscale orography",
            units="1",
            date_available=IFS_CYCLE_50R1_CUTOVER,
            control_only=True,
            lead_0_only=True,
        ),
        var(
            "geopotential",
            param="z",
            short_name="z",
            long_name="Geopotential",
            units="m2 s-2",
            standard_name="geopotential",
            group="pressure_level",
            date_available=IFS_CYCLE_50R1_CUTOVER,
            control_only=True,
        ),
        var(
            "temperature",
            param="t",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
            group="pressure_level",
        ),
        var(
            "wind_u",
            param="u",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
            group="pressure_level",
        ),
        var(
            "wind_v",
            param="v",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
            group="pressure_level",
        ),
        var(
            "vertical_velocity",
            param="w",
            short_name="w",
            long_name="Vertical velocity",
            units="Pa s-1",
            standard_name="lagrangian_tendency_of_air_pressure",
            group="pressure_level",
        ),
        var(
            "specific_humidity",
            param="q",
            short_name="q",
            long_name="Specific humidity",
            units="1",
            standard_name="specific_humidity",
            group="pressure_level",
        ),
        var(
            "soil_temperature_layer_3",
            param="sot",
            long_name="Soil temperature",
            units="degree_Celsius",
            standard_name="soil_temperature",
            level_value=3,
            legacy_soil_param="stl3",
            comment="ECMWF soil level 3. Over water this is the sea surface temperature, not a soil temperature.",
        ),
        var(
            "volumetric_soil_moisture_layer_3",
            param="vsw",
            long_name="Volumetric soil moisture",
            units="1",
            standard_name="volume_fraction_of_condensed_water_in_soil",
            level_value=3,
            legacy_soil_param="swvl3",
            comment="ECMWF soil level 3. Applies over land only; over water the values are near zero, so mask with land_sea_mask_surface.",
        ),
        var(
            "soil_temperature_layer_4",
            param="sot",
            long_name="Soil temperature",
            units="degree_Celsius",
            standard_name="soil_temperature",
            level_value=4,
            legacy_soil_param="stl4",
            comment="ECMWF soil level 4. Over water this is the sea surface temperature, not a soil temperature.",
        ),
        var(
            "volumetric_soil_moisture_layer_4",
            param="vsw",
            long_name="Volumetric soil moisture",
            units="1",
            standard_name="volume_fraction_of_condensed_water_in_soil",
            level_value=4,
            legacy_soil_param="swvl4",
            comment="ECMWF soil level 4. Applies over land only; over water the values are near zero, so mask with land_sea_mask_surface.",
        ),
        var(
            "snow_albedo_surface",
            param="asn",
            long_name="Snow albedo",
            units="percent",
            comment=(
                "Applies only where snow is present (snow_water_equivalent_surface > 0); "
                "snow-free cells hold placeholder values that also occur over snow."
            ),
            filters=[ScaleOffset(offset=0, scale=0.01).to_dict()],
        ),
        var(
            "convective_available_potential_energy_surface",
            param="cape",
            long_name="Convective available potential energy",
            units="J kg-1",
            standard_name="atmosphere_convective_available_potential_energy",
            date_unavailable=IFS_CYCLE_49R1_CUTOVER,
        ),
        var(
            "most_unstable_convective_available_potential_energy_surface",
            param="mucape",
            standard_name="atmosphere_convective_available_potential_energy",
            long_name="Most-unstable CAPE",
            units="J kg-1",
            date_available=IFS_CYCLE_49R1_CUTOVER,
        ),
        var(
            "wind_gust_10m",
            param="10fg",
            long_name="Maximum 10 metre wind gust since previous post-processing",
            units="m s-1",
            standard_name="wind_speed_of_gust",
            step_type="max",
            date_available=IFS_CYCLE_49R1_CUTOVER,
            comment="Maximum over the preceding hour through lead 90 hours, then since the preceding forecast step.",
            lead_time_overrides=((pd.Timedelta("93h"), pd.Timedelta("144h"), "10fg3"),),
        ),
        var(
            "maximum_temperature_2m",
            param="mx2t3",
            short_name="tmax",
            long_name="Maximum temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
            step_type="max",
            date_available=IFS_CYCLE_49R1_CUTOVER,
            comment="Maximum since the preceding forecast step.",
            lead_time_overrides=(
                (pd.Timedelta("150h"), pd.Timedelta("360h"), "mx2t6"),
            ),
        ),
        var(
            "minimum_temperature_2m",
            param="mn2t3",
            short_name="tmin",
            long_name="Minimum temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
            step_type="min",
            date_available=IFS_CYCLE_49R1_CUTOVER,
            comment="Minimum since the preceding forecast step.",
            lead_time_overrides=(
                (pd.Timedelta("150h"), pd.Timedelta("360h"), "mn2t6"),
            ),
        ),
        var(
            "categorical_precipitation_type_surface",
            param="ptype",
            long_name="Precipitation type",
            units="1",
            date_available=IFS_CYCLE_49R1_CUTOVER,
            flag_values=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 255),
            flag_meanings="no_precipitation rain thunderstorm freezing_rain mixed_ice snow wet_snow mixture_of_rain_and_snow ice_pellets graupel hail drizzle freezing_drizzle hail_less_than_5mm hail_greater_than_or_equal_to_5mm missing",
        ),
        var(
            "eastward_turbulent_surface_stress_run_total_surface",
            param="ewss",
            standard_name="integral_wrt_time_of_surface_downward_eastward_stress",
            long_name="Time-integrated eastward turbulent surface stress",
            units="Pa s",
            step_type="accum",
            date_available=IFS_CYCLE_49R1_CUTOVER,
        ),
        var(
            "northward_turbulent_surface_stress_run_total_surface",
            param="nsss",
            standard_name="integral_wrt_time_of_surface_downward_northward_stress",
            long_name="Time-integrated northward turbulent surface stress",
            units="Pa s",
            step_type="accum",
            date_available=IFS_CYCLE_49R1_CUTOVER,
        ),
        var(
            "snow_density_surface",
            param="rsn",
            long_name="Snow density",
            units="kg m-3",
            standard_name="surface_snow_density",
            comment=(
                "Applies only where snow is present (snow_water_equivalent_surface > 0); "
                "snow-free cells hold placeholder values that also occur over snow."
            ),
            date_available=pd.Timestamp("2026-02-11T12:00:00"),
        ),
        var(
            "snow_water_equivalent_surface",
            param="sd",
            long_name="Snow depth water equivalent",
            units="m",
            standard_name="lwe_thickness_of_surface_snow_amount",
            date_available=pd.Timestamp("2025-11-20T12:00:00"),
            comment="Capped at 10 m, the value held over ice sheets and glaciers.",
        ),
        var(
            "sea_ice_thickness_surface",
            param="sithick",
            long_name="Sea ice thickness",
            units="m",
            standard_name="sea_ice_thickness",
            date_available=IFS_CYCLE_49R1_CUTOVER,
            comment="NaN where there is no ocean model value, including over land.",
        ),
        var(
            "sea_surface_height",
            param="zos",
            long_name="Sea surface height",
            units="m",
            date_available=IFS_CYCLE_49R1_CUTOVER,
            comment="NaN where there is no ocean model value, including over land.",
        ),
        var(
            "sea_water_velocity_u_surface",
            param="sve",
            long_name="Eastward surface sea water velocity",
            units="m s-1",
            standard_name="eastward_sea_water_velocity",
            date_available=IFS_CYCLE_49R1_CUTOVER,
            comment="Applies over the ocean only; over land the values are NaN or near zero, so mask with land_sea_mask_surface.",
        ),
        var(
            "sea_water_velocity_v_surface",
            param="svn",
            long_name="Northward surface sea water velocity",
            units="m s-1",
            standard_name="northward_sea_water_velocity",
            date_available=IFS_CYCLE_49R1_CUTOVER,
            comment="Applies over the ocean only; over land the values are NaN or near zero, so mask with land_sea_mask_surface.",
        ),
        var(
            "net_short_wave_radiation_run_total_surface",
            param="ssr",
            long_name="Surface net short-wave (solar) radiation",
            units="W s m-2",
            standard_name="integral_wrt_time_of_surface_net_downward_shortwave_flux",
            step_type="accum",
        ),
        var(
            "net_long_wave_radiation_run_total_surface",
            param="str",
            long_name="Surface net long-wave (thermal) radiation",
            units="W s m-2",
            standard_name="integral_wrt_time_of_surface_net_downward_longwave_flux",
            step_type="accum",
        ),
        var(
            "net_long_wave_radiation_run_total_atmosphere_top",
            param="ttr",
            comment="Positive downwards, the negative of outgoing long-wave radiation.",
            long_name="Top net long-wave (thermal) radiation",
            units="W s m-2",
            step_type="accum",
        ),
        var(
            "precipitable_water_atmosphere",
            param="tcwv",
            short_name="pwat",
            long_name="Precipitable water",
            units="kg m-2",
            standard_name="atmosphere_mass_content_of_water_vapor",
        ),
        var(
            "total_column_cloud_ice_water_atmosphere",
            param="tciw",
            long_name="Total column cloud ice water",
            units="kg m-2",
            standard_name="atmosphere_mass_content_of_cloud_ice",
            date_available=pd.Timestamp("2026-10-20T06:00:00"),
        ),
        var(
            "total_column_cloud_liquid_water_atmosphere",
            param="tclw",
            long_name="Total column cloud liquid water",
            units="kg m-2",
            standard_name="atmosphere_mass_content_of_cloud_liquid_water",
            date_available=pd.Timestamp("2026-10-20T06:00:00"),
        ),
        var(
            "total_column_rain_water_atmosphere",
            param="tcrw",
            long_name="Total column rain water",
            units="kg m-2",
            date_available=pd.Timestamp("2026-10-20T06:00:00"),
        ),
        var(
            "precipitation_rate_surface",
            param="tprate",
            long_name="Precipitation rate",
            units="kg m-2 s-1",
            standard_name="precipitation_flux",
            short_name="prate",
            date_available=IFS_CYCLE_49R1_CUTOVER,
        ),
        var(
            "geopotential_height",
            param="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            group="pressure_level",
        ),
        var(
            "relative_humidity",
            param="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
            group="pressure_level",
        ),
        var(
            "divergence",
            param="d",
            long_name="Divergence",
            units="s-1",
            standard_name="divergence_of_wind",
            group="pressure_level",
        ),
        var(
            "relative_vorticity",
            param="vo",
            long_name="Vorticity (relative)",
            units="s-1",
            standard_name="atmosphere_upward_relative_vorticity",
            group="pressure_level",
        ),
    ]
