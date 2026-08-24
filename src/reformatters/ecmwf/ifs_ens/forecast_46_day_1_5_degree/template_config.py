from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import (
    ROOT,
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
from reformatters.common.types import Dim, Dims, Timedelta
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_config_models import (
    EcmwfIfsEns46DayDataVar,
    EcmwfIfsEns46DayInternalAttrs,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_template_config import (
    EcmwfIfsEns46DayCommonTemplateConfig,
)

PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 300, 200, 100, 50, 10]


class EcmwfIfsEnsForecast46Day15DegreeTemplateConfig(
    EcmwfIfsEns46DayCommonTemplateConfig
):
    """The 24 hourly variables of the ECMWF IFS ENS sub-seasonal-range forecast."""

    dims: Dims = {
        ROOT: (
            "init_time",
            "lead_time",
            "ensemble_member",
            "latitude",
            "longitude",
        ),
        "pressure_level": (
            "init_time",
            "lead_time",
            "ensemble_member",
            "pressure_level",
            "latitude",
            "longitude",
        ),
    }

    lead_time_frequency: Timedelta = pd.Timedelta("24h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="ecmwf-ifs-ens-forecast-46-day-1-5-degree",
            dataset_version="0.1.0",
            name="ECMWF IFS ENS forecast, 46 day, 1.5 degree",
            description="Sub-seasonal-range ensemble weather forecasts from the ECMWF Integrated Forecasting System (IFS).",
            attribution="ECMWF IFS ENS sub-seasonal-range forecast data processed by dynamical.org from the ECMWF Data Store.",
            license="CC-BY-4.0",
            spatial_domain="Global",
            spatial_resolution="1.5 degrees (~165km)",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:.0f} hours",
            forecast_domain="Forecast lead time 0-1104 hours (0-46 days) ahead",
            forecast_resolution="Forecast step 24 hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        return super().dimension_coordinates() | {
            "pressure_level": np.array(PRESSURE_LEVELS, dtype=np.int64),
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        dim_coords = self.dimension_coordinates()
        return [
            *super().coords,
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
        ]

    @computed_field
    @property
    def data_vars(self) -> Sequence[EcmwfIfsEns46DayDataVar]:
        # Roughly ~10.0MB uncompressed
        root_chunks: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 47,  # All lead times
            "ensemble_member": 101,  # All ensemble members
            "latitude": 25,  # 5 chunks over 121 pixels
            "longitude": 24,  # 10 chunks over 240 pixels
        }
        # Roughly ~570MB uncompressed
        root_shards: dict[Dim, int] = {
            "init_time": root_chunks["init_time"],
            "lead_time": root_chunks["lead_time"],
            "ensemble_member": root_chunks["ensemble_member"],
            "latitude": root_chunks["latitude"] * 5,
            "longitude": root_chunks["longitude"] * 10,
        }
        pressure_level_chunks: dict[Dim, int] = root_chunks | {"pressure_level": 1}
        pressure_level_shards: dict[Dim, int] = root_shards | {"pressure_level": 1}

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(root_chunks[d] for d in self.dims[ROOT]),
            shards=tuple(root_shards[d] for d in self.dims[ROOT]),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )
        encoding_float32_pressure_level = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(pressure_level_chunks[d] for d in self.dims["pressure_level"]),
            shards=tuple(pressure_level_shards[d] for d in self.dims["pressure_level"]),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        return [
            EcmwfIfsEns46DayDataVar(
                name="pressure_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sp",
                    long_name="Surface pressure",
                    standard_name="surface_air_pressure",
                    units="Pa",
                    step_type="instant",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="surface_pressure",
                    grib_element="PRES",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[Pa]",
                    keep_mantissa_bits=11,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="pressure_reduced_to_mean_sea_level",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="prmsl",
                    long_name="Pressure reduced to MSL",
                    standard_name="air_pressure_at_mean_sea_level",
                    units="Pa",
                    step_type="instant",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="mean_sea_level_pressure",
                    grib_element="PRES",
                    grib_description='0[-] MSL="Mean sea level"',
                    grib_unit="[Pa]",
                    keep_mantissa_bits=11,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="precipitation_convective_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="cpr",
                    long_name="Convective precipitation rate",
                    standard_name="convective_precipitation_flux",
                    units="kg m-2 s-1",
                    step_type="avg",
                    comment="Average convective precipitation rate over the previous 24 hours. Units equivalent to mm/s.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="convective_precipitation",
                    grib_element="CPRAT",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[kg/(m^2*s)]",
                    keep_mantissa_bits=8,
                    deaccumulate_to_rate=True,
                    window_reset_frequency=pd.Timedelta.max,
                    hour_0_values_override=True,
                    deaccumulation_invalid_below_threshold_rate=PRECIPITATION_RATE_INVALID_BELOW_THRESHOLD,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="snowfall_water_equivalent_rate_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="tsrwe",
                    long_name="Total snowfall rate water equivalent",
                    standard_name="snowfall_flux",
                    units="kg m-2 s-1",
                    step_type="avg",
                    comment="Average snowfall water equivalent rate over the previous 24 hours. Units equivalent to mm/s.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="snow_fall_water_equivalent",
                    grib_element="TSRWE",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[kg/(m^2*s)]",
                    keep_mantissa_bits=8,
                    deaccumulate_to_rate=True,
                    window_reset_frequency=pd.Timedelta.max,
                    hour_0_values_override=True,
                    deaccumulation_invalid_below_threshold_rate=PRECIPITATION_RATE_INVALID_BELOW_THRESHOLD,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="runoff_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="surfror",
                    long_name="Surface runoff rate",
                    standard_name="surface_runoff_flux",
                    units="kg m-2 s-1",
                    step_type="avg",
                    comment="Average surface runoff rate over the previous 24 hours. Units equivalent to mm/s. Land points only; sea points are missing.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="surface_runoff",
                    grib_element="SFCWRO",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[kg/(m^2)]",
                    keep_mantissa_bits=7,
                    source_fill_value=9999.0,
                    deaccumulate_to_rate=True,
                    window_reset_frequency=pd.Timedelta.max,
                    hour_0_values_override=True,
                    deaccumulation_invalid_below_threshold_rate=PRECIPITATION_RATE_INVALID_BELOW_THRESHOLD,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="runoff_water_equivalent_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="rorwe",
                    long_name="Runoff rate water equivalent (surface plus subsurface)",
                    standard_name="runoff_flux",
                    units="kg m-2 s-1",
                    step_type="avg",
                    comment="Average runoff water equivalent rate (surface plus subsurface) over the previous 24 hours. Units equivalent to mm/s. Land points only; sea points are missing.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="water_runoff_and_drainage",
                    grib_element="WROD",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[kg/(m^2)]",
                    keep_mantissa_bits=7,
                    source_fill_value=9999.0,
                    deaccumulate_to_rate=True,
                    window_reset_frequency=pd.Timedelta.max,
                    hour_0_values_override=True,
                    deaccumulation_invalid_below_threshold_rate=PRECIPITATION_RATE_INVALID_BELOW_THRESHOLD,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="downward_short_wave_radiation_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sdswrf",
                    long_name="Surface downward short-wave radiation flux",
                    standard_name="surface_downwelling_shortwave_flux_in_air",
                    units="W m-2",
                    step_type="avg",
                    comment="Average surface downward short-wave radiation flux over the previous 24 hours.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="surface_solar_radiation_downwards",
                    grib_element="DSWRF",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[W/(m^2)]",
                    keep_mantissa_bits=7,
                    deaccumulate_to_rate=True,
                    window_reset_frequency=pd.Timedelta.max,
                    hour_0_values_override=True,
                    deaccumulation_invalid_below_threshold_rate=RADIATION_INVALID_BELOW_THRESHOLD,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="downward_long_wave_radiation_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sdlwrf",
                    long_name="Surface downward long-wave radiation flux",
                    standard_name="surface_downwelling_longwave_flux_in_air",
                    units="W m-2",
                    step_type="avg",
                    comment="Average surface downward long-wave radiation flux over the previous 24 hours.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="surface_thermal_radiation_downwards",
                    grib_element="DLWRF",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[W/(m^2)]",
                    keep_mantissa_bits=7,
                    deaccumulate_to_rate=True,
                    window_reset_frequency=pd.Timedelta.max,
                    hour_0_values_override=True,
                    deaccumulation_invalid_below_threshold_rate=RADIATION_INVALID_BELOW_THRESHOLD,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="net_short_wave_radiation_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="snswrf",
                    long_name="Surface net short-wave radiation flux",
                    standard_name="surface_net_downward_shortwave_flux",
                    units="W m-2",
                    step_type="avg",
                    comment="Average surface net short-wave radiation flux over the previous 24 hours.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="surface_net_solar_radiation",
                    grib_element="NSWRF",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[W/(m^2)]",
                    keep_mantissa_bits=7,
                    deaccumulate_to_rate=True,
                    window_reset_frequency=pd.Timedelta.max,
                    hour_0_values_override=True,
                    deaccumulation_invalid_below_threshold_rate=RADIATION_INVALID_BELOW_THRESHOLD,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="net_long_wave_radiation_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="snlwrf",
                    long_name="Surface net long-wave radiation flux",
                    standard_name="surface_net_downward_longwave_flux",
                    units="W m-2",
                    step_type="avg",
                    comment="Average surface net long-wave radiation flux over the previous 24 hours.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="surface_net_thermal_radiation",
                    grib_element="NLWRF",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[W/(m^2)]",
                    keep_mantissa_bits=7,
                    deaccumulate_to_rate=True,
                    deaccumulation_type="signed",
                    window_reset_frequency=pd.Timedelta.max,
                    hour_0_values_override=True,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="net_long_wave_radiation_flux_top_of_atmosphere",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="tnlwrf",
                    long_name="Top net long-wave radiation flux",
                    standard_name="toa_net_downward_longwave_flux",
                    units="W m-2",
                    step_type="avg",
                    comment="Average top net long-wave radiation flux over the previous 24 hours.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="top_net_thermal_radiation",
                    grib_element="NLWRF",
                    grib_description='0[-] NTAT="Nominal top of atmosphere"',
                    grib_unit="[W/(m^2)]",
                    keep_mantissa_bits=7,
                    deaccumulate_to_rate=True,
                    deaccumulation_type="signed",
                    window_reset_frequency=pd.Timedelta.max,
                    hour_0_values_override=True,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="downward_latent_heat_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="mslhfl",
                    long_name="Time-mean surface latent heat flux",
                    standard_name="surface_downward_latent_heat_flux",
                    units="W m-2",
                    step_type="avg",
                    comment="Average surface downward latent heat flux over the previous 24 hours.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="surface_latent_heat_flux",
                    grib_element="LHTFL",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[W/(m^2)]",
                    keep_mantissa_bits=7,
                    deaccumulate_to_rate=True,
                    deaccumulation_type="signed",
                    window_reset_frequency=pd.Timedelta.max,
                    hour_0_values_override=True,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="downward_sensible_heat_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="msshfl",
                    long_name="Time-mean surface sensible heat flux",
                    standard_name="surface_downward_sensible_heat_flux",
                    units="W m-2",
                    step_type="avg",
                    comment="Average surface downward sensible heat flux over the previous 24 hours.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="surface_sensible_heat_flux",
                    grib_element="SHTFL",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[W/(m^2)]",
                    keep_mantissa_bits=7,
                    deaccumulate_to_rate=True,
                    deaccumulation_type="signed",
                    window_reset_frequency=pd.Timedelta.max,
                    hour_0_values_override=True,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="eastward_turbulent_surface_stress",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="avg_iews",
                    long_name="Time-mean eastward turbulent surface stress",
                    standard_name="surface_downward_eastward_stress",
                    units="Pa",
                    step_type="avg",
                    comment="Average eastward turbulent surface stress over the previous 24 hours.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="eastward_turbulent_surface_stress",
                    grib_element="ETSS",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[1/(m^2)]",
                    keep_mantissa_bits=7,
                    deaccumulate_to_rate=True,
                    deaccumulation_type="signed",
                    window_reset_frequency=pd.Timedelta.max,
                    hour_0_values_override=True,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="northward_turbulent_surface_stress",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="avg_inss",
                    long_name="Time-mean northward turbulent surface stress",
                    standard_name="surface_downward_northward_stress",
                    units="Pa",
                    step_type="avg",
                    comment="Average northward turbulent surface stress over the previous 24 hours.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="northward_turbulent_surface_stress",
                    grib_element="NTSS",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[1/(m^2)]",
                    keep_mantissa_bits=7,
                    deaccumulate_to_rate=True,
                    deaccumulation_type="signed",
                    window_reset_frequency=pd.Timedelta.max,
                    hour_0_values_override=True,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="average_temperature_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="mean2t24",
                    long_name="Mean temperature at 2 metres in the last 24 hours",
                    standard_name="air_temperature",
                    units="degree_Celsius",
                    step_type="avg",
                    comment="Mean temperature at 2 metres over the previous 24 hours.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="2_m_temperature",
                    grib_element="TMP",
                    grib_description='2[m] HTGL="Specified height level above ground"',
                    grib_unit="[K]",
                    keep_mantissa_bits=7,
                    add_offset=-273.15,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="average_dew_point_temperature_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="mn2d24",
                    long_name="Mean 2 metre dewpoint temperature in the last 24 hours",
                    standard_name="dew_point_temperature",
                    units="degree_Celsius",
                    step_type="avg",
                    comment="Mean 2 metre dewpoint temperature over the previous 24 hours.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="2_m_dewpoint_temperature",
                    grib_element="DPT",
                    grib_description='2[m] HTGL="Specified height level above ground"',
                    grib_unit="[K]",
                    keep_mantissa_bits=7,
                    add_offset=-273.15,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="skin_temperature_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="skt",
                    long_name="Skin temperature",
                    standard_name="surface_temperature",
                    units="degree_Celsius",
                    step_type="avg",
                    comment="Mean skin temperature over the previous 24 hours.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="skin_temperature",
                    grib_element="SKINT",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[K]",
                    keep_mantissa_bits=7,
                    add_offset=-273.15,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="sea_surface_temperature",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sst",
                    long_name="Sea surface temperature",
                    standard_name="sea_surface_temperature",
                    units="degree_Celsius",
                    step_type="avg",
                    comment="Mean sea surface temperature over the previous 24 hours. Sea points only; land points are missing.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="sea_surface_temperature",
                    grib_element="WTMP",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[K]",
                    keep_mantissa_bits=7,
                    add_offset=-273.15,
                    source_fill_value=9999.0,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="sea_ice_area_fraction",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="ci",
                    long_name="Sea ice area fraction",
                    standard_name="sea_ice_area_fraction",
                    units="1",
                    step_type="avg",
                    comment="Mean sea ice area fraction over the previous 24 hours. Sea points only; land points are missing.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="sea_ice_area_fraction",
                    grib_element="ICEC",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[Proportion]",
                    keep_mantissa_bits=7,
                    source_fill_value=9999.0,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="soil_temperature_0_20cm",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="st20",
                    long_name="Soil temperature top 20 cm",
                    standard_name="soil_temperature",
                    units="degree_Celsius",
                    step_type="avg",
                    comment="Mean soil temperature 0-20 cm over the previous 24 hours. Land points only; sea points are missing.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="soil_temperature_top_20_cm",
                    grib_element="TSOIL",
                    grib_description='0-0.2[m] DBLL="Depth below land surface"',
                    grib_unit="[K]",
                    keep_mantissa_bits=7,
                    add_offset=-273.15,
                    source_fill_value=9999.0,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="soil_temperature_0_100cm",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="st100",
                    long_name="Soil temperature top 100 cm",
                    standard_name="soil_temperature",
                    units="degree_Celsius",
                    step_type="avg",
                    comment="Mean soil temperature 0-100 cm over the previous 24 hours. Land points only; sea points are missing.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="soil_temperature_top_100_cm",
                    grib_element="TSOIL",
                    grib_description='0-1[m] DBLL="Depth below land surface"',
                    grib_unit="[K]",
                    keep_mantissa_bits=7,
                    add_offset=-273.15,
                    source_fill_value=9999.0,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="soil_moisture_0_20cm",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sm20",
                    long_name="Soil moisture top 20 cm",
                    units="kg m-3",
                    step_type="avg",
                    comment="Mean soil moisture 0-20 cm over the previous 24 hours. Land points only; sea points are missing.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="soil_moisture_top_20_cm",
                    grib_element="SOILM",
                    grib_description='0-0.2[m] DBLL="Depth below land surface"',
                    grib_unit="[kg/m^3]",
                    keep_mantissa_bits=7,
                    source_fill_value=9999.0,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="soil_moisture_0_100cm",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sm100",
                    long_name="Soil moisture top 100 cm",
                    units="kg m-3",
                    step_type="avg",
                    comment="Mean soil moisture 0-100 cm over the previous 24 hours. Land points only; sea points are missing.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="soil_moisture_top_100_cm",
                    grib_element="SOILM",
                    grib_description='0-1[m] DBLL="Depth below land surface"',
                    grib_unit="[kg/m^3]",
                    keep_mantissa_bits=7,
                    source_fill_value=9999.0,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="snow_water_equivalent_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sd",
                    long_name="Snow depth water equivalent",
                    standard_name="lwe_thickness_of_surface_snow_amount",
                    units="m",
                    step_type="avg",
                    comment="Mean snow depth water equivalent over the previous 24 hours.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="snow_depth_water_equivalent",
                    grib_element="SDWE",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[kg/m^2]",
                    keep_mantissa_bits=7,
                    scale_factor=0.001,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="snow_density_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="rsn",
                    long_name="Snow density",
                    units="kg m-3",
                    step_type="avg",
                    comment="Mean snow density over the previous 24 hours. Land points only; sea points are missing.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="snow_density",
                    grib_element="SDEN",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[kg/m^3]",
                    keep_mantissa_bits=7,
                    source_fill_value=9999.0,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="snow_albedo_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="asn",
                    long_name="Snow albedo",
                    units="percent",
                    step_type="avg",
                    comment="Mean snow albedo over the previous 24 hours. Land points only; sea points are missing.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="snow_albedo",
                    grib_element="SALBD",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[%]",
                    keep_mantissa_bits=7,
                    source_fill_value=9999.0,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="total_cloud_cover_atmosphere",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="tcc",
                    long_name="Total cloud cover",
                    standard_name="cloud_area_fraction",
                    units="percent",
                    step_type="avg",
                    comment="Mean total cloud cover over the previous 24 hours.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="total_cloud_cover",
                    grib_element="TCDC",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[%]",
                    keep_mantissa_bits=7,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="total_column_water_atmosphere",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="tcw",
                    long_name="Total column water",
                    standard_name="atmosphere_mass_content_of_water",
                    units="kg m-2",
                    step_type="avg",
                    comment="Mean total column water over the previous 24 hours.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="total_column_water",
                    grib_element="TCWAT",
                    grib_description='0[-] SFC="Ground or water surface"',
                    grib_unit="[kg/m^2]",
                    keep_mantissa_bits=7,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="temperature",
                group="pressure_level",
                encoding=encoding_float32_pressure_level,
                attrs=DataVarAttrs(
                    short_name="t",
                    long_name="Temperature",
                    standard_name="air_temperature",
                    units="degree_Celsius",
                    step_type="instant",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="temperature",
                    grib_element="TMP",
                    grib_description="",
                    grib_unit="[K]",
                    keep_mantissa_bits=7,
                    add_offset=-273.15,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="specific_humidity",
                group="pressure_level",
                encoding=encoding_float32_pressure_level,
                attrs=DataVarAttrs(
                    short_name="q",
                    long_name="Specific humidity",
                    standard_name="specific_humidity",
                    units="1",
                    step_type="instant",
                    comment="The source provides no 10, 50 or 100 hPa levels for this variable; those levels are always NaN.",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="specific_humidity",
                    grib_element="SPFH",
                    grib_description="",
                    grib_unit="[kg/kg]",
                    keep_mantissa_bits=7,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="wind_u",
                group="pressure_level",
                encoding=encoding_float32_pressure_level,
                attrs=DataVarAttrs(
                    short_name="u",
                    long_name="U component of wind",
                    standard_name="eastward_wind",
                    units="m s-1",
                    step_type="instant",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="u_component_of_wind",
                    grib_element="UGRD",
                    grib_description="",
                    grib_unit="[m/s]",
                    keep_mantissa_bits=6,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="wind_v",
                group="pressure_level",
                encoding=encoding_float32_pressure_level,
                attrs=DataVarAttrs(
                    short_name="v",
                    long_name="V component of wind",
                    standard_name="northward_wind",
                    units="m s-1",
                    step_type="instant",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="v_component_of_wind",
                    grib_element="VGRD",
                    grib_description="",
                    grib_unit="[m/s]",
                    keep_mantissa_bits=6,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="geopotential_height",
                group="pressure_level",
                encoding=encoding_float32_pressure_level,
                attrs=DataVarAttrs(
                    short_name="gh",
                    long_name="Geopotential height",
                    standard_name="geopotential_height",
                    units="m",
                    step_type="instant",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="geopotential_height",
                    grib_element="HGT",
                    grib_description="",
                    grib_unit="[gpm]",
                    keep_mantissa_bits=11,
                ),
            ),
            EcmwfIfsEns46DayDataVar(
                name="vertical_velocity",
                group="pressure_level",
                encoding=encoding_float32_pressure_level,
                attrs=DataVarAttrs(
                    short_name="w",
                    long_name="Vertical velocity",
                    standard_name="lagrangian_tendency_of_air_pressure",
                    units="Pa s-1",
                    step_type="instant",
                ),
                internal_attrs=EcmwfIfsEns46DayInternalAttrs(
                    ecds_variable="vertical_velocity",
                    grib_element="VVEL",
                    grib_description="",
                    grib_unit="[Pa/s]",
                    keep_mantissa_bits=7,
                ),
            ),
        ]
