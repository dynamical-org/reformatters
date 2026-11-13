from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
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
from reformatters.common.types import CodecConfig, Dims, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar, EcmwfInternalAttrs
from reformatters.ecmwf.ecmwf_template_config import (
    Ecmwf025DegreeForecastTemplateConfig,
)

AIFS_ENS_UPGRADE_DATE = pd.Timestamp("2026-05-12T06:00")
PRESSURE_LEVELS = (10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
_KELVIN_TO_CELSIUS = ScaleOffset(offset=-273.15, scale=1.0).to_dict()
_GEOPOTENTIAL_TO_HEIGHT = ScaleOffset(offset=0.0, scale=9.80665).to_dict()
_STATIC_COMMENT = "Time-invariant field published at lead time 0 only."


def available_pressure_levels(init_time: Timestamp) -> tuple[int, ...]:
    return (
        PRESSURE_LEVELS if init_time >= AIFS_ENS_UPGRADE_DATE else PRESSURE_LEVELS[1:]
    )


class EcmwfAifsEnsVirtualInternalAttrs(EcmwfInternalAttrs):
    lead_0_only: bool = False


class EcmwfAifsEnsVirtualDataVar(EcmwfDataVar):
    internal_attrs: EcmwfAifsEnsVirtualInternalAttrs

    def available(self, init_time: Timestamp, lead_time: pd.Timedelta) -> bool:
        attrs = self.internal_attrs
        return (attrs.date_available is None or init_time >= attrs.date_available) and (
            lead_time == pd.Timedelta(0) or not attrs.lead_0_only
        )


class EcmwfAifsEnsForecastVirtualTemplateConfig(
    Ecmwf025DegreeForecastTemplateConfig[EcmwfAifsEnsVirtualDataVar]
):
    append_dim_start: Timestamp = pd.Timestamp("2025-07-01T06:00")
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

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="ecmwf-aifs-ens-forecast-virtual",
            dataset_version="0.1.0",
            name="ECMWF AIFS ENS forecast, virtual",
            description="Ensemble weather forecasts from the ECMWF Artificial Intelligence Forecasting System (AIFS) ENS model.",
            attribution="ECMWF AIFS ENS forecast data processed by dynamical.org from ECMWF Open Data.",
            license="CC-BY-4.0",
            spatial_domain="Global",
            spatial_resolution="0.25 degrees (~20km)",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:.0f} hours",
            forecast_domain="Forecast lead time 0-360 hours (0-15 days) ahead",
            forecast_resolution="6 hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        return super().dimension_coordinates() | {
            "lead_time": pd.timedelta_range("0h", "360h", freq="6h"),
            "ensemble_member": np.arange(51),
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
    def data_vars(self) -> Sequence[EcmwfAifsEnsVirtualDataVar]:
        return [*_root_data_vars(), *_pressure_data_vars()]


def _var(
    name: str,
    *,
    param: str,
    long_name: str,
    short_name: str,
    units: str,
    standard_name: str | None = None,
    group: Group = ROOT,
    level_type: Literal["sfc", "pl", "sol"] = "sfc",
    level_value: float = float("nan"),
    step_type: Literal["instant", "accum"] = "instant",
    comment: str | None = None,
    date_available: Timestamp | None = None,
    lead_0_only: bool = False,
    filters: Sequence[CodecConfig] | None = None,
) -> EcmwfAifsEnsVirtualDataVar:
    if filters is None:
        filters = [_KELVIN_TO_CELSIUS] if units == "degree_Celsius" else ()
    comments = [comment] if comment else []
    if step_type == "accum":
        comments.append("Accumulated since forecast initialization.")
    return EcmwfAifsEnsVirtualDataVar(
        name=name,
        group=group,
        encoding=Encoding(
            dtype="float64",
            fill_value=np.nan,
            chunks=(1, 1, 1, 721, 1440) + (() if group is ROOT else (1,)),
            shards=None,
            compressors=(),
            filters=filters,
            serializer=GribberishCodec(
                var=param, adjust_longitude_range=True, north_up=True
            ).to_dict(),
        ),
        attrs=DataVarAttrs(
            short_name=short_name,
            long_name=long_name,
            units=units,
            standard_name=standard_name,
            step_type=step_type,
            comment=" ".join(comments) or None,
        ),
        internal_attrs=EcmwfAifsEnsVirtualInternalAttrs(
            grib_index_param=param,
            grib_index_level_type="pl" if group == "pressure_level" else level_type,
            grib_index_level_value=level_value,
            date_available=date_available,
            lead_0_only=lead_0_only,
            keep_mantissa_bits="no-rounding",
            grib_comment="",
            grib_element="",
            grib_description="",
        ),
    )


def _root_data_vars() -> list[EcmwfAifsEnsVirtualDataVar]:
    return [
        _var(
            "pressure_surface",
            param="sp",
            short_name="sp",
            long_name="Surface pressure",
            units="Pa",
            standard_name="surface_air_pressure",
        ),
        _var(
            "pressure_reduced_to_mean_sea_level",
            param="msl",
            short_name="prmsl",
            long_name="Pressure reduced to MSL",
            units="Pa",
            standard_name="air_pressure_at_mean_sea_level",
        ),
        _var(
            "temperature_2m",
            param="2t",
            short_name="2t",
            long_name="2 metre temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        _var(
            "dew_point_temperature_2m",
            param="2d",
            short_name="2d",
            long_name="2 metre dewpoint temperature",
            units="degree_Celsius",
            standard_name="dew_point_temperature",
            comment="Dew point can exceed air temperature, so derived relative humidity can exceed 100%.",
        ),
        _var(
            "skin_temperature_surface",
            param="skt",
            short_name="skt",
            long_name="Skin temperature",
            units="degree_Celsius",
            standard_name="surface_temperature",
        ),
        _var(
            "wind_u_10m",
            param="10u",
            short_name="10u",
            long_name="10 metre U wind component",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        _var(
            "wind_v_10m",
            param="10v",
            short_name="10v",
            long_name="10 metre V wind component",
            units="m s-1",
            standard_name="northward_wind",
        ),
        _var(
            "wind_u_100m",
            param="100u",
            short_name="100u",
            long_name="100 metre U wind component",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        _var(
            "wind_v_100m",
            param="100v",
            short_name="100v",
            long_name="100 metre V wind component",
            units="m s-1",
            standard_name="northward_wind",
        ),
        _var(
            "total_cloud_cover_atmosphere",
            param="tcc",
            short_name="tcc",
            long_name="Total cloud cover",
            units="percent",
            standard_name="cloud_area_fraction",
        ),
        _var(
            "low_cloud_cover",
            param="lcc",
            short_name="lcc",
            long_name="Low cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
        ),
        _var(
            "medium_cloud_cover",
            param="mcc",
            short_name="mcc",
            long_name="Medium cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
        ),
        _var(
            "high_cloud_cover",
            param="hcc",
            short_name="hcc",
            long_name="High cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
        ),
        _var(
            "total_column_water_atmosphere",
            param="tcw",
            short_name="tcw",
            long_name="Total column water",
            units="kg m-2",
            standard_name="atmosphere_mass_content_of_water",
        ),
        _var(
            "snow_area_fraction_surface",
            param="fscov",
            short_name="snowc",
            long_name="Snow cover",
            units="1",
            standard_name="surface_snow_area_fraction",
            comment=(
                "Fraction (0-1) of the grid box covered by snow. Applies over land only; "
                "NaN over water, except at forecast hour 0, which has values over water: "
                "0 over open ocean and up to 1 in polar seas."
            ),
            date_available=AIFS_ENS_UPGRADE_DATE,
        ),
        _var(
            "total_precipitation_run_total_surface",
            param="tp",
            step_type="accum",
            short_name="tp",
            long_name="Total precipitation",
            units="kg m-2",
            standard_name="precipitation_amount",
        ),
        _var(
            "convective_precipitation_run_total_surface",
            param="cp",
            date_available=AIFS_ENS_UPGRADE_DATE,
            step_type="accum",
            short_name="cp",
            long_name="Convective precipitation",
            units="kg m-2",
            standard_name="convective_precipitation_amount",
        ),
        _var(
            "snowfall_water_equivalent_run_total_surface",
            param="sf",
            step_type="accum",
            short_name="sf",
            long_name="Snowfall water equivalent",
            units="kg m-2",
            standard_name="snowfall_amount",
        ),
        _var(
            "runoff_water_equivalent_run_total_surface",
            param="rowe",
            step_type="accum",
            short_name="rowe",
            long_name="Runoff water equivalent (surface plus subsurface)",
            units="kg m-2",
            standard_name="runoff_amount",
            comment="Applies over land only; over water values may be NaN or finite, so mask with land_sea_mask_surface.",
        ),
        _var(
            "downward_short_wave_radiation_run_total_surface",
            param="ssrd",
            step_type="accum",
            short_name="ssrd",
            long_name="Surface short-wave (solar) radiation downwards",
            units="W s m-2",
            standard_name="integral_wrt_time_of_surface_downwelling_shortwave_flux_in_air",
        ),
        _var(
            "downward_long_wave_radiation_run_total_surface",
            param="strd",
            step_type="accum",
            short_name="strd",
            long_name="Surface long-wave (thermal) radiation downwards",
            units="W s m-2",
            standard_name="integral_wrt_time_of_surface_downwelling_longwave_flux_in_air",
        ),
        _var(
            "soil_temperature_layer_1",
            param="sot",
            level_type="sol",
            level_value=1,
            short_name="sot",
            long_name="Soil temperature",
            units="degree_Celsius",
            standard_name="soil_temperature",
            comment="ECMWF soil level 1, the uppermost soil layer. Over water this is the sea surface temperature, not a soil temperature.",
        ),
        _var(
            "soil_temperature_layer_2",
            param="sot",
            level_type="sol",
            level_value=2,
            short_name="sot",
            long_name="Soil temperature",
            units="degree_Celsius",
            standard_name="soil_temperature",
            comment="ECMWF soil level 2, the second soil layer from the surface. Over water this is the sea surface temperature, not a soil temperature.",
        ),
        _var(
            "volumetric_soil_moisture_layer_1",
            param="vsw",
            date_available=AIFS_ENS_UPGRADE_DATE,
            level_type="sol",
            level_value=1,
            short_name="vsw",
            long_name="Volumetric soil moisture",
            units="1",
            standard_name="volume_fraction_of_condensed_water_in_soil",
            comment="ECMWF soil level 1, the uppermost soil layer. Applies over land only; NaN over water.",
        ),
        _var(
            "volumetric_soil_moisture_layer_2",
            param="vsw",
            date_available=AIFS_ENS_UPGRADE_DATE,
            level_type="sol",
            level_value=2,
            short_name="vsw",
            long_name="Volumetric soil moisture",
            units="1",
            standard_name="volume_fraction_of_condensed_water_in_soil",
            comment="ECMWF soil level 2, the second soil layer from the surface. Applies over land only; NaN over water.",
        ),
        _var(
            "land_sea_mask_surface",
            param="lsm",
            short_name="lsm",
            long_name="Land-sea mask",
            units="1",
            standard_name="land_area_fraction",
            comment=f"Fraction (0-1) of the grid box that is land. {_STATIC_COMMENT}",
            lead_0_only=True,
        ),
        _var(
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
        _var(
            "standard_deviation_of_sub_gridscale_orography_surface",
            param="sdor",
            short_name="sdor",
            long_name="Standard deviation of sub-gridscale orography",
            units="m",
            comment=_STATIC_COMMENT,
            lead_0_only=True,
        ),
        _var(
            "slope_of_sub_gridscale_orography_surface",
            param="slor",
            short_name="slor",
            long_name="Slope of sub-gridscale orography",
            units="1",
            comment=_STATIC_COMMENT,
            lead_0_only=True,
        ),
    ]


def _pressure_data_vars() -> list[EcmwfAifsEnsVirtualDataVar]:
    return [
        _var(
            "geopotential_height",
            param="z",
            group="pressure_level",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            filters=[_GEOPOTENTIAL_TO_HEIGHT],
        ),
        _var(
            "temperature",
            param="t",
            group="pressure_level",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        _var(
            "wind_u",
            param="u",
            group="pressure_level",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        _var(
            "wind_v",
            param="v",
            group="pressure_level",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        _var(
            "vertical_velocity",
            param="w",
            group="pressure_level",
            short_name="w",
            long_name="Vertical velocity",
            units="Pa s-1",
            standard_name="lagrangian_tendency_of_air_pressure",
        ),
        _var(
            "specific_humidity",
            param="q",
            group="pressure_level",
            short_name="q",
            long_name="Specific humidity",
            units="1",
            standard_name="specific_humidity",
            comment="The source provides no 10 hPa level for this variable; that level is always NaN.",
        ),
    ]
