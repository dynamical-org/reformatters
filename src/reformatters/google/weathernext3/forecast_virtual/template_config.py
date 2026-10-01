from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import computed_field
from zarr.codecs import BytesCodec, ScaleOffset, ZstdCodec

from reformatters.common.config_models import (
    ROOT,
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
from reformatters.common.types import AppendDim, Dims, Timedelta, Timestamp
from reformatters.common.zarr import BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE

ATTRIBUTION = "Google requires this attribution: © 2026 DeepMind Technologies Limited's machine learning models used to create the experimental data made available at https://developers.google.com/earth-engine/datasets/catalog/projects_gcp-public-data-weathernext_assets_weathernext_3_0_0_0p1deg under CC BY 4.0 licence terms. This data is intended for experimental modelling only and is not intended, validated, or approved for real world use. Use of the third-party materials referred to in the Acknowledgements section may be governed by separate terms and conditions or license provisions. Your use of the third-party materials is subject to any such terms and you should check that you can comply with any applicable restrictions or terms and conditions before use."

STATISTICS = ("mean", "p10", "p25", "p50", "p75", "p90")


class GoogleWeathernext3InternalAttrs(BaseInternalAttrs):
    source_name: str


class GoogleWeathernext3DataVar(DataVar[GoogleWeathernext3InternalAttrs]):
    pass


class GoogleWeathernext3ForecastVirtualTemplateConfig(
    TemplateConfig[GoogleWeathernext3DataVar]
):
    horizon_hours: ClassVar[int]
    grid_degrees: ClassVar[float]
    dataset_id_value: ClassVar[str]
    dataset_name_value: ClassVar[str]
    dims: Dims = {ROOT: ("init_time", "statistic", "lead_time", "y", "x")}
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = pd.Timestamp("2026-01-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        description = "Weather forecasts from the 64-member Google DeepMind WeatherNext 3 ensemble model. Forecast values are published hourly once their valid time is at least one hour in the past, so recent initialization times are intentionally partial."
        if self.grid_degrees == 0.05:
            description += " These variables are WeatherNext 3's station-trained head, defined everywhere."
        return DatasetAttributes(
            dataset_id=self.dataset_id_value,
            dataset_version="0.1.0",
            name=self.dataset_name_value,
            description=description,
            attribution=ATTRIBUTION,
            license="CC-BY-4.0",
            spatial_domain="Global",
            spatial_resolution="0.1 degrees (~10km)"
            if self.grid_degrees == 0.1
            else "0.05 degrees (~5km)",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:g} hours",
            forecast_domain=f"Forecast lead time 1-{self.horizon_hours} hours ahead",
            forecast_resolution="Hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            self.append_dim: self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "statistic": np.array(STATISTICS),
            "lead_time": pd.timedelta_range(
                "1h", periods=self.horizon_hours, freq="1h"
            ),
            "y": np.linspace(-90, 90, round(180 / self.grid_degrees) + 1),
            "x": np.arange(round(360 / self.grid_degrees)) * self.grid_degrees,
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
                        min=dim_coords[self.append_dim].min().isoformat(),
                        max="Present",
                    ),
                ),
            ),
            Coordinate(
                name="statistic",
                encoding=Encoding(dtype="str", fill_value="", chunks=6, shards=None),
                attrs=CoordinateAttrs(
                    long_name="Ensemble statistic",
                    units="1",
                    statistics_approximate=None,
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
                name="y",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["y"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Latitude",
                    standard_name="latitude",
                    units="degree_north",
                    axis="Y",
                    statistics_approximate=StatisticsApproximate(
                        min=float(dim_coords["y"].min()),
                        max=float(dim_coords["y"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="x",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["x"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Longitude",
                    standard_name="longitude",
                    units="degree_east",
                    axis="X",
                    statistics_approximate=StatisticsApproximate(
                        min=float(dim_coords["x"].min()),
                        max=float(dim_coords["x"].max()),
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
                        max=f"Present + {self.horizon_hours} hours",
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
                    grid_mapping_name="latitude_longitude",
                    comment=f"The source declares no coordinate reference system and uses a {self.grid_degrees:g} degree latitude-longitude grid.",
                ),
            ),
        ]

    @computed_field
    @property
    def data_vars(self) -> Sequence[GoogleWeathernext3DataVar]:
        specs = VARIABLE_SPECS
        if self.grid_degrees == 0.05:
            specs = [
                spec
                for spec in specs
                if spec[0] in ("temperature_2m", "dew_point_temperature_2m")
            ]
        result = []
        for name, source, scale, offset, attributes in specs:
            attrs = DataVarAttrs.model_validate(attributes)
            source_name = source
            if self.grid_degrees == 0.05:
                source_name = "station_head_" + source
                attrs = attrs.model_copy(
                    update={
                        "comment": "WeatherNext 3's station-trained head, defined everywhere."
                    }
                )
            result.append(
                GoogleWeathernext3DataVar(
                    name=name,
                    attrs=attrs,
                    encoding=Encoding(
                        dtype="float32",
                        fill_value=np.nan,
                        chunks=(
                            1,
                            1,
                            1,
                            round(180 / self.grid_degrees) + 1,
                            round(360 / self.grid_degrees),
                        ),
                        shards=None,
                        serializer=BytesCodec(endian="little").to_dict(),
                        compressors=[ZstdCodec(level=0, checksum=False).to_dict()],
                        filters=[ScaleOffset(scale=scale, offset=offset).to_dict()]
                        if (scale, offset) != (1, 0)
                        else [],
                    ),
                    internal_attrs=GoogleWeathernext3InternalAttrs(
                        source_name=source_name, keep_mantissa_bits="no-rounding"
                    ),
                )
            )
        return result


VARIABLE_SPECS = [
    (
        "temperature_2m",
        "temperature_2m",
        1,
        -273.15,
        {
            "long_name": "2 metre temperature",
            "short_name": "2t",
            "standard_name": "air_temperature",
            "units": "degree_Celsius",
            "step_type": "instant",
        },
    ),
    (
        "pressure_reduced_to_mean_sea_level",
        "mean_sea_level_pressure",
        1,
        0,
        {
            "long_name": "Pressure reduced to MSL",
            "short_name": "prmsl",
            "standard_name": "air_pressure_at_mean_sea_level",
            "units": "Pa",
            "step_type": "instant",
        },
    ),
    (
        "sea_surface_temperature",
        "sea_surface_temperature",
        1,
        -273.15,
        {
            "long_name": "Sea surface temperature",
            "short_name": "sst",
            "standard_name": "sea_surface_temperature",
            "units": "degree_Celsius",
            "comment": "NaN over land where sea surface temperature does not apply.",
            "step_type": "instant",
        },
    ),
    (
        "wind_u_10m",
        "u_component_of_wind_10m",
        1,
        0,
        {
            "long_name": "10 metre U wind component",
            "short_name": "10u",
            "standard_name": "eastward_wind",
            "units": "m s-1",
            "step_type": "instant",
        },
    ),
    (
        "wind_v_10m",
        "v_component_of_wind_10m",
        1,
        0,
        {
            "long_name": "10 metre V wind component",
            "short_name": "10v",
            "standard_name": "northward_wind",
            "units": "m s-1",
            "step_type": "instant",
        },
    ),
    (
        "wind_u_100m",
        "u_component_of_wind_100m",
        1,
        0,
        {
            "long_name": "100 metre U wind component",
            "short_name": "100u",
            "standard_name": "eastward_wind",
            "units": "m s-1",
            "step_type": "instant",
        },
    ),
    (
        "wind_v_100m",
        "v_component_of_wind_100m",
        1,
        0,
        {
            "long_name": "100 metre V wind component",
            "short_name": "100v",
            "standard_name": "northward_wind",
            "units": "m s-1",
            "step_type": "instant",
        },
    ),
    (
        "dew_point_temperature_2m",
        "dewpoint_temperature_2m",
        1,
        -273.15,
        {
            "long_name": "2 metre dewpoint temperature",
            "short_name": "2d",
            "standard_name": "dew_point_temperature",
            "units": "degree_Celsius",
            "step_type": "instant",
        },
    ),
    (
        "wind_speed_10m",
        "wind_speed_10m",
        1,
        0,
        {
            "long_name": "10 metre wind speed",
            "short_name": "10si",
            "standard_name": "wind_speed",
            "units": "m s-1",
            "step_type": "instant",
            "comment": "The statistic of ensemble wind speed, not derived from the u/v statistics.",
        },
    ),
    (
        "wind_speed_100m",
        "wind_speed_100m",
        1,
        0,
        {
            "long_name": "100 metre wind speed",
            "short_name": "100si",
            "standard_name": "wind_speed",
            "units": "m s-1",
            "step_type": "instant",
            "comment": "The statistic of ensemble wind speed, not derived from the u/v statistics.",
        },
    ),
    (
        "total_cloud_cover_atmosphere",
        "total_cloud_cover",
        0.01,
        0,
        {
            "long_name": "Total cloud cover",
            "short_name": "tcc",
            "standard_name": "cloud_area_fraction",
            "units": "percent",
            "step_type": "instant",
        },
    ),
    (
        "cloud_cover_low",
        "low_cloud_cover",
        0.01,
        0,
        {
            "long_name": "Low cloud cover",
            "short_name": "lcc",
            "standard_name": "cloud_area_fraction_in_atmosphere_layer",
            "units": "percent",
            "step_type": "instant",
        },
    ),
    (
        "cloud_cover_medium",
        "medium_cloud_cover",
        0.01,
        0,
        {
            "long_name": "Medium cloud cover",
            "short_name": "mcc",
            "standard_name": "cloud_area_fraction_in_atmosphere_layer",
            "units": "percent",
            "step_type": "instant",
        },
    ),
    (
        "cloud_cover_high",
        "high_cloud_cover",
        0.01,
        0,
        {
            "long_name": "High cloud cover",
            "short_name": "hcc",
            "standard_name": "cloud_area_fraction_in_atmosphere_layer",
            "units": "percent",
            "step_type": "instant",
        },
    ),
    (
        "precipitation_surface",
        "total_precipitation_1hr",
        3.6,
        0,
        {
            "long_name": "Precipitation rate",
            "short_name": "prate",
            "standard_name": "precipitation_flux",
            "units": "kg m-2 s-1",
            "comment": "Average precipitation rate since the previous forecast step. Units equivalent to mm/s. WeatherNext 3 model-native 1-hour precipitation.",
            "step_type": "avg",
        },
    ),
    (
        "precipitation_imerg_surface",
        "imerg_tp_1hr",
        3.6,
        0,
        {
            "long_name": "Precipitation rate",
            "short_name": "prate",
            "standard_name": "precipitation_flux",
            "units": "kg m-2 s-1",
            "comment": "Average precipitation rate since the previous forecast step. Units equivalent to mm/s. WeatherNext 3 forecast of 1-hour precipitation calibrated to IMERG satellite precipitation; a model forecast, not an observation.",
            "step_type": "avg",
        },
    ),
    (
        "precipitation_experimental_surface",
        "experimental_tp_1hr",
        3.6,
        0,
        {
            "long_name": "Precipitation rate",
            "short_name": "prate",
            "standard_name": "precipitation_flux",
            "units": "kg m-2 s-1",
            "comment": "Average precipitation rate since the previous forecast step. Units equivalent to mm/s. WeatherNext 3 forecast of 1-hour precipitation from its experimental satellite-radar precipitation head; a model forecast, not an observation.",
            "step_type": "avg",
        },
    ),
    (
        "downward_short_wave_radiation_flux_surface",
        "surface_solar_radiation_downwards_1hr",
        3600,
        0,
        {
            "long_name": "Surface downward short-wave radiation flux",
            "short_name": "sdswrf",
            "standard_name": "surface_downwelling_shortwave_flux_in_air",
            "units": "W m-2",
            "comment": "Average flux since the previous forecast step.",
            "step_type": "avg",
        },
    ),
    (
        "downward_direct_short_wave_radiation_flux_surface",
        "total_sky_direct_solar_radiation_at_surface_1hr",
        3600,
        0,
        {
            "long_name": "Surface direct short-wave radiation flux",
            "short_name": "aswdir_s",
            "standard_name": "surface_direct_downwelling_shortwave_flux_in_air",
            "units": "W m-2",
            "comment": "Average flux since the previous forecast step.",
            "step_type": "avg",
        },
    ),
]
