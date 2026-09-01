import functools
from collections.abc import Callable, Sequence
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
    DataVarAttrs,
    Encoding,
    Group,
    StatisticsApproximate,
)
from reformatters.common.types import CodecConfig, Timedelta
from reformatters.common.zarr import BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE
from reformatters.noaa.gfs.template_config import NoaaGfsCommonTemplateConfig
from reformatters.noaa.models import NoaaDataVar, NoaaInternalAttrs

# GFS 0.25 degree global grid, from NoaaGfsCommonTemplateConfig._latitude_longitude_coordinates.
# Asserted against those coordinates in data_vars so the two cannot drift.
_GRID_NLAT = 721
_GRID_NLON = 1440

# The union of the isobaric levels pgrb2 (41) and pgrb2b (16 more) publish, in hPa,
# descending like the GRIB order. Not every element is carried on all 57.
# The idx renders an isobaric level with %g formatting, so the float64 pressure_level
# coordinate reproduces every one of the 57 level strings exactly ("1000", "0.01").
PRESSURE_LEVEL_INDEX_FORMAT = "{level:g} mb"

PRESSURE_LEVELS = [
    1000.0,
    975.0,
    950.0,
    925.0,
    900.0,
    875.0,
    850.0,
    825.0,
    800.0,
    775.0,
    750.0,
    725.0,
    700.0,
    675.0,
    650.0,
    625.0,
    600.0,
    575.0,
    550.0,
    525.0,
    500.0,
    475.0,
    450.0,
    425.0,
    400.0,
    375.0,
    350.0,
    325.0,
    300.0,
    275.0,
    250.0,
    225.0,
    200.0,
    175.0,
    150.0,
    125.0,
    100.0,
    70.0,
    50.0,
    40.0,
    30.0,
    20.0,
    15.0,
    10.0,
    7.0,
    5.0,
    3.0,
    2.0,
    1.0,
    0.7,
    0.4,
    0.2,
    0.1,
    0.07,
    0.04,
    0.02,
    0.01,
]

# The eight "N m above mean sea level" heights, in metres, ascending. GRIB2 level type
# 102, specified altitude above mean sea level. pgrb2 publishes 1829/2743/3658 and
# pgrb2b the other five, so the family is dense only across both products; note 4572 m
# is the topmost yet comes from pgrb2b, so the split is not a high/low cut.
HEIGHT_LEVEL_INDEX_FORMAT = "{level:g} m above mean sea level"

HEIGHT_ABOVE_MEAN_SEA_LEVELS = [
    305.0,
    457.0,
    610.0,
    914.0,
    1829.0,
    2743.0,
    3658.0,
    4572.0,
]

# GribberishCodec decodes the raw kelvin message; this array->array filter subtracts
# 273.15 on read. GDAL relabels ten GFS elements kelvin -> Celsius but converts only six,
# so this set is derived from measured values, not from the GRIB unit label: it drops
# POT (kelvin by convention, matching the HRRR virtual potential_temperature_2m) and the
# lifted-index temperature differences, and adds TSOIL and ICETMP, which GDAL mislabels.
_KELVIN_TO_CELSIUS = ScaleOffset(offset=-273.15, scale=1.0).to_dict()
_CELSIUS_ELEMENTS = frozenset(
    {"TMP", "TMAX", "TMIN", "DPT", "APTMP", "TSOIL", "ICETMP"}
)

# ScaleOffset decodes on read as value / scale + offset (see zarr.codecs.ScaleOffset).
# WEASD decodes as kg m-2 of water and 1 kg m-2 = 0.001 m lwe, so scale=1000 yields the
# metres the identically-named HRRR virtual snow_water_equivalent_surface serves.
_WATER_KG_M2_TO_M_LWE = ScaleOffset(offset=0.0, scale=1000.0).to_dict()

type WindowKind = Literal["instant", "max", "min", "avg", "acc_6h", "acc_run"]

# Each windowed kind's (step_type, window_reset_frequency). acc_6h and the avg/max/min
# kinds are the bucket since the most recent multiple of 6 hours of lead time; acc_run is
# the accumulation since initialization. noaa_grib_index.grib_index_window_str renders the
# matching idx window string per lead from step_type + window_reset_frequency.
_WINDOW_ATTRS: dict[WindowKind, tuple[str, Timedelta | None]] = {
    "instant": ("instant", None),
    "max": ("max", pd.Timedelta("6h")),
    "min": ("min", pd.Timedelta("6h")),
    "avg": ("avg", pd.Timedelta("6h")),
    "acc_6h": ("accum", pd.Timedelta("6h")),
    "acc_run": ("accum", pd.Timedelta.max),
}


class NoaaGfsVirtualTemplateConfig(NoaaGfsCommonTemplateConfig):
    """Virtual GFS template: one chunk per GRIB message on the source 0.25 degree
    latitude/longitude grid, covering every pgrb2 and pgrb2b message as a root array or
    a pressure_level group array. A subclass declares dims and time structure.
    """

    def _vertical_dimension_coordinates(self) -> dict[str, Any]:
        return {
            "pressure_level": np.array(PRESSURE_LEVELS, dtype=np.float64),
            "height_above_mean_sea_level": np.array(
                HEIGHT_ABOVE_MEAN_SEA_LEVELS, dtype=np.float64
            ),
        }

    def _vertical_coords(self) -> list[Coordinate]:
        vertical = self._vertical_dimension_coordinates()
        pressure_levels = vertical["pressure_level"]
        heights = vertical["height_above_mean_sea_level"]
        return [
            Coordinate(
                name="pressure_level",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(pressure_levels),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Pressure level",
                    standard_name="air_pressure",
                    units="hPa",
                    axis="Z",
                    positive="down",
                    statistics_approximate=StatisticsApproximate(
                        min=float(pressure_levels.min()),
                        max=float(pressure_levels.max()),
                    ),
                ),
            ),
            Coordinate(
                name="height_above_mean_sea_level",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(heights),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    # CF distinguishes this from `altitude`, which is height above the
                    # geoid; GRIB2 level type 102 is height above mean sea level.
                    long_name="Height above mean sea level",
                    standard_name="height_above_mean_sea_level",
                    units="m",
                    axis="Z",
                    positive="up",
                    statistics_approximate=StatisticsApproximate(
                        min=float(heights.min()), max=float(heights.max())
                    ),
                ),
            ),
        ]

    @computed_field
    @property
    def data_vars(self) -> Sequence[NoaaDataVar]:
        lat_lon = self._latitude_longitude_coordinates()
        assert (len(lat_lon["latitude"]), len(lat_lon["longitude"])) == (
            _GRID_NLAT,
            _GRID_NLON,
        ), "grid shape drifted from _GRID_NLAT/_GRID_NLON used in the virtual encoding"
        return self._catalog_data_vars()

    def _catalog_data_vars(self) -> list[NoaaDataVar]:
        """The variable catalog, for the groups this config declares; a subclass may
        filter it further to serve a subset."""
        catalog: dict[Group, Callable[[tuple[int, ...]], list[NoaaDataVar]]] = {
            ROOT: _root_data_vars,
            "pressure_level": _pressure_data_vars,
            "height_above_mean_sea_level": _height_data_vars,
        }
        return [
            var
            for group, build_vars in catalog.items()
            if group in self.dims
            for var in build_vars(self._message_chunks(group))
        ]

    def _message_chunks(self, group: Group) -> tuple[int, ...]:
        """One chunk per GRIB message: the full grid, size 1 along every other dim."""
        dims = self.dims[group]
        assert {"latitude", "longitude"} <= set(dims), (
            f"{group} dims {dims} do not span the GFS latitude/longitude grid, so a "
            "chunk would not hold one whole GRIB message"
        )
        return tuple(
            {"latitude": _GRID_NLAT, "longitude": _GRID_NLON}.get(dim, 1)
            for dim in dims
        )


def _virtual_encoding(
    element: str,
    chunks: tuple[int, ...],
    filters: Sequence[CodecConfig],
    fill_value: float,
) -> Encoding:
    """No shards, no compressors; GribberishCodec decodes the raw message and any
    array->array filters (K->C, unit scaling) are chained on read."""
    return Encoding(
        # GribberishCodec decodes to float64 natively; declaring float64 avoids a cast.
        dtype="float64",
        fill_value=fill_value,
        chunks=chunks,
        shards=None,
        compressors=(),
        filters=filters,
        serializer=GribberishCodec(
            var=element, adjust_longitude_range=True, north_up=True
        ).to_dict(),
    )


def _data_var(
    name: str,
    *,
    chunks: tuple[int, ...],
    element: str,
    element_alternatives: tuple[str, ...] = (),
    grib_index_level: str,
    group: Group,
    window: WindowKind,
    short_name: str,
    long_name: str,
    units: str,
    standard_name: str | None,
    comment: str | None,
    hour_0: bool | None,
    fill_value: float = np.nan,
    filters: Sequence[CodecConfig] | None = None,
    flag_values: tuple[int, ...] | None = None,
    flag_meanings: str | None = None,
) -> NoaaDataVar:
    step_type, window_reset_frequency = _WINDOW_ATTRS[window]
    resolved_filters: Sequence[CodecConfig] = (
        filters
        if filters is not None
        else ([_KELVIN_TO_CELSIUS] if element in _CELSIUS_ELEMENTS else ())
    )
    return NoaaDataVar(
        name=name,
        group=group,
        encoding=_virtual_encoding(element, chunks, resolved_filters, fill_value),
        attrs=DataVarAttrs(
            short_name=short_name,
            long_name=long_name,
            units=units,
            standard_name=standard_name,
            step_type=step_type,  # ty: ignore[invalid-argument-type]
            comment=comment,
            flag_values=flag_values,
            flag_meanings=flag_meanings,
        ),
        internal_attrs=NoaaInternalAttrs(
            grib_element=element,
            grib_element_alternatives=element_alternatives,
            # Group vars carry a "{level:g} mb" format string the region job fills per
            # level; root vars carry the literal idx level string.
            grib_index_level=grib_index_level,
            window_reset_frequency=window_reset_frequency,
            hour_0_values_override=hour_0,
            # Virtual chunks are never rewritten, so no rounding and no rasterio band
            # description / index position (unused fields the base model requires).
            keep_mantissa_bits="no-rounding",
            grib_description="",
            index_position=0,
        ),
    )


def _root_var(
    name: str,
    *,
    chunks: tuple[int, ...],
    element: str,
    element_alternatives: tuple[str, ...] = (),
    level: str,
    window: WindowKind = "instant",
    short_name: str,
    long_name: str,
    units: str,
    standard_name: str | None = None,
    comment: str | None = None,
    hour_0: bool | None = None,
    fill_value: float = np.nan,
    filters: Sequence[CodecConfig] | None = None,
    flag_values: tuple[int, ...] | None = None,
    flag_meanings: str | None = None,
) -> NoaaDataVar:
    return _data_var(
        name,
        chunks=chunks,
        element=element,
        element_alternatives=element_alternatives,
        grib_index_level=level,
        group=ROOT,
        window=window,
        short_name=short_name,
        long_name=long_name,
        units=units,
        standard_name=standard_name,
        comment=comment,
        hour_0=hour_0,
        fill_value=fill_value,
        filters=filters,
        flag_values=flag_values,
        flag_meanings=flag_meanings,
    )


def _pressure_var(
    name: str,
    *,
    chunks: tuple[int, ...],
    element: str,
    element_alternatives: tuple[str, ...] = (),
    short_name: str,
    long_name: str,
    units: str,
    standard_name: str | None = None,
    comment: str | None = None,
) -> NoaaDataVar:
    return _data_var(
        name,
        chunks=chunks,
        element=element,
        element_alternatives=element_alternatives,
        grib_index_level=PRESSURE_LEVEL_INDEX_FORMAT,
        group="pressure_level",
        window="instant",
        short_name=short_name,
        long_name=long_name,
        units=units,
        standard_name=standard_name,
        comment=comment,
        hour_0=None,
    )


def _height_var(
    name: str,
    *,
    chunks: tuple[int, ...],
    element: str,
    short_name: str,
    long_name: str,
    units: str,
    standard_name: str | None = None,
    comment: str | None = None,
) -> NoaaDataVar:
    return _data_var(
        name,
        chunks=chunks,
        element=element,
        grib_index_level=HEIGHT_LEVEL_INDEX_FORMAT,
        group="height_above_mean_sea_level",
        window="instant",
        short_name=short_name,
        long_name=long_name,
        units=units,
        standard_name=standard_name,
        comment=comment,
        hour_0=None,
    )


def _height_data_vars(chunks: tuple[int, ...]) -> list[NoaaDataVar]:
    height_var = functools.partial(_height_var, chunks=chunks)
    return [
        height_var(
            "temperature",
            element="TMP",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
            comment="NaN where the terrain is above this level.",
        ),
        height_var(
            "wind_u",
            element="UGRD",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
            comment="NaN where the terrain is above this level.",
        ),
        height_var(
            "wind_v",
            element="VGRD",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
            comment="NaN where the terrain is above this level.",
        ),
    ]


def _root_data_vars(chunks: tuple[int, ...]) -> list[NoaaDataVar]:
    root_var = functools.partial(_root_var, chunks=chunks)
    return [
        root_var(
            "pressure_reduced_to_mean_sea_level",
            element="PRMSL",
            level="mean sea level",
            short_name="prmsl",
            long_name="Pressure reduced to MSL",
            units="Pa",
            standard_name="air_pressure_at_mean_sea_level",
        ),
        root_var(
            "composite_reflectivity",
            element="REFC",
            level="entire atmosphere",
            short_name="refc",
            long_name="Maximum/Composite radar reflectivity",
            units="dBZ",
            standard_name="equivalent_reflectivity_factor",
            comment=(
                "-20 dBZ is the source's no-echo floor: those cells mean no echo "
                "was detected, not a measured value."
            ),
        ),
        root_var(
            "visibility_surface",
            element="VIS",
            level="surface",
            short_name="vis",
            long_name="Visibility",
            units="m",
            standard_name="visibility_in_air",
            comment=(
                "Saturates at the model's maximum reported visibility, about 24 km. "
                "That ceiling means unlimited visibility rather than missing data."
            ),
        ),
        root_var(
            "wind_u_planetary_boundary_layer",
            element="UGRD",
            level="planetary boundary layer",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        root_var(
            "wind_v_planetary_boundary_layer",
            element="VGRD",
            level="planetary boundary layer",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        root_var(
            "ventilation_rate_planetary_boundary_layer",
            element="VRATE",
            level="planetary boundary layer",
            short_name="VRATE",
            long_name="Ventilation Rate",
            units="m2 s-1",
        ),
        root_var(
            "wind_gust_surface",
            element="GUST",
            level="surface",
            short_name="gust",
            long_name="Wind speed (gust)",
            units="m s-1",
            standard_name="wind_speed_of_gust",
        ),
        root_var(
            "haines_index_surface",
            element="HINDEX",
            level="surface",
            short_name="hindex",
            long_name="Haines Index",
            units="1",
            comment=(
                "Fire-weather index of lower-atmosphere stability and dryness, an "
                "ordinal value from 2 (very low potential) to 6 (high potential) for "
                "large plume-dominated fire growth. NaN on about 60% of the grid, "
                "chiefly over ocean, where the source does not compute the index."
            ),
        ),
        root_var(
            "pressure_reduced_to_mean_sea_level_eta_model",
            element="MSLET",
            level="mean sea level",
            short_name="mslet",
            long_name="MSLP (Eta model reduction)",
            units="Pa",
            standard_name="air_pressure_at_mean_sea_level",
            comment=(
                "Mean sea level pressure computed with the Eta model reduction, an "
                "alternative to pressure_reduced_to_mean_sea_level that differs mainly "
                "over high terrain."
            ),
        ),
        root_var(
            "derived_radar_reflectivity_4000m",
            element="REFD",
            level="4000 m above ground",
            short_name="refd",
            long_name="Derived radar reflectivity",
            units="dBZ",
            standard_name="equivalent_reflectivity_factor",
            comment=(
                "-20 dBZ is the source's no-echo floor: those cells mean no echo "
                "was detected, not a measured value."
            ),
        ),
        root_var(
            "derived_radar_reflectivity_1000m",
            element="REFD",
            level="1000 m above ground",
            short_name="refd",
            long_name="Derived radar reflectivity",
            units="dBZ",
            standard_name="equivalent_reflectivity_factor",
            comment=(
                "-20 dBZ is the source's no-echo floor: those cells mean no echo "
                "was detected, not a measured value."
            ),
        ),
        root_var(
            "pressure_surface",
            element="PRES",
            level="surface",
            short_name="sp",
            long_name="Surface pressure",
            units="Pa",
            standard_name="surface_air_pressure",
        ),
        root_var(
            "geopotential_height_surface",
            element="HGT",
            level="surface",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
        ),
        root_var(
            "temperature_surface",
            element="TMP",
            level="surface",
            short_name="skt",
            long_name="Skin temperature",
            units="degree_Celsius",
            standard_name="surface_temperature",
        ),
        root_var(
            "soil_temperature_0_10cm",
            element="TSOIL",
            level="0-0.1 m below ground",
            short_name="sot",
            long_name="Soil temperature",
            units="degree_Celsius",
            standard_name="soil_temperature",
            comment=(
                "NaN over open water; the source reports this quantity only over land "
                "and sea ice."
            ),
        ),
        root_var(
            "volumetric_soil_moisture_0_10cm",
            element="SOILW",
            level="0-0.1 m below ground",
            short_name="soilw",
            long_name="Volumetric soil moisture content",
            units="1",
            standard_name="volume_fraction_of_condensed_water_in_soil",
            comment="NaN over water, where this quantity does not apply.",
        ),
        root_var(
            "liquid_volumetric_soil_moisture_0_10cm",
            element="SOILL",
            level="0-0.1 m below ground",
            short_name="soill",
            long_name="Liquid volumetric soil moisture (non-frozen)",
            units="1",
            comment=(
                "Unfrozen fraction only; volumetric_soil_moisture_0_10cm carries frozen "
                "plus liquid water. NaN over water, where this quantity does not apply."
            ),
        ),
        root_var(
            "soil_temperature_10_40cm",
            element="TSOIL",
            level="0.1-0.4 m below ground",
            short_name="sot",
            long_name="Soil temperature",
            units="degree_Celsius",
            standard_name="soil_temperature",
            comment=(
                "NaN over open water; the source reports this quantity only over land "
                "and sea ice."
            ),
        ),
        root_var(
            "volumetric_soil_moisture_10_40cm",
            element="SOILW",
            level="0.1-0.4 m below ground",
            short_name="soilw",
            long_name="Volumetric soil moisture content",
            units="1",
            standard_name="volume_fraction_of_condensed_water_in_soil",
            comment="NaN over water, where this quantity does not apply.",
        ),
        root_var(
            "liquid_volumetric_soil_moisture_10_40cm",
            element="SOILL",
            level="0.1-0.4 m below ground",
            short_name="soill",
            long_name="Liquid volumetric soil moisture (non-frozen)",
            units="1",
            comment=(
                "Unfrozen fraction only; volumetric_soil_moisture_10_40cm carries frozen "
                "plus liquid water. NaN over water, where this quantity does not apply."
            ),
        ),
        root_var(
            "soil_temperature_40_100cm",
            element="TSOIL",
            level="0.4-1 m below ground",
            short_name="sot",
            long_name="Soil temperature",
            units="degree_Celsius",
            standard_name="soil_temperature",
            comment=(
                "NaN over open water; the source reports this quantity only over land "
                "and sea ice."
            ),
        ),
        root_var(
            "volumetric_soil_moisture_40_100cm",
            element="SOILW",
            level="0.4-1 m below ground",
            short_name="soilw",
            long_name="Volumetric soil moisture content",
            units="1",
            standard_name="volume_fraction_of_condensed_water_in_soil",
            comment="NaN over water, where this quantity does not apply.",
        ),
        root_var(
            "liquid_volumetric_soil_moisture_40_100cm",
            element="SOILL",
            level="0.4-1 m below ground",
            short_name="soill",
            long_name="Liquid volumetric soil moisture (non-frozen)",
            units="1",
            comment=(
                "Unfrozen fraction only; volumetric_soil_moisture_40_100cm carries "
                "frozen plus liquid water. NaN over water, where this quantity does not "
                "apply."
            ),
        ),
        root_var(
            "soil_temperature_100_200cm",
            element="TSOIL",
            level="1-2 m below ground",
            short_name="sot",
            long_name="Soil temperature",
            units="degree_Celsius",
            standard_name="soil_temperature",
            comment=(
                "NaN over open water; the source reports this quantity only over land "
                "and sea ice."
            ),
        ),
        root_var(
            "volumetric_soil_moisture_100_200cm",
            element="SOILW",
            level="1-2 m below ground",
            short_name="soilw",
            long_name="Volumetric soil moisture content",
            units="1",
            standard_name="volume_fraction_of_condensed_water_in_soil",
            comment="NaN over water, where this quantity does not apply.",
        ),
        root_var(
            "liquid_volumetric_soil_moisture_100_200cm",
            element="SOILL",
            level="1-2 m below ground",
            short_name="soill",
            long_name="Liquid volumetric soil moisture (non-frozen)",
            units="1",
            comment=(
                "Unfrozen fraction only; volumetric_soil_moisture_100_200cm carries "
                "frozen plus liquid water. NaN over water, where this quantity does not "
                "apply."
            ),
        ),
        root_var(
            "plant_canopy_surface_water_surface",
            element="CNWAT",
            level="surface",
            short_name="cnwat",
            long_name="Plant canopy surface water",
            units="kg m-2",
            standard_name="canopy_water_amount",
            comment="NaN over water, where this quantity does not apply.",
        ),
        root_var(
            "snow_water_equivalent_surface",
            element="WEASD",
            level="surface",
            short_name="sd",
            long_name="Snow depth water equivalent",
            units="m",
            standard_name="lwe_thickness_of_surface_snow_amount",
            comment=(
                "NaN over open water; the source reports this quantity only over land "
                "and sea ice."
            ),
            filters=[_WATER_KG_M2_TO_M_LWE],
        ),
        root_var(
            "snow_thickness_surface",
            element="SNOD",
            level="surface",
            short_name="sde",
            long_name="Snow depth",
            units="m",
            standard_name="surface_snow_thickness",
            comment=(
                "NaN over open water; the source reports this quantity only over land "
                "and sea ice."
            ),
        ),
        root_var(
            "potential_evaporation_rate_surface",
            element="PEVPR",
            level="surface",
            short_name="pevr",
            long_name="Potential evaporation rate",
            units="W m-2",
            comment=(
                "NaN over open water; the source reports this quantity only over land "
                "and sea ice."
            ),
            hour_0=False,
        ),
        root_var(
            "ice_thickness_surface",
            element="ICETK",
            level="surface",
            short_name="icetk",
            long_name="Ice thickness",
            units="m",
            comment="Thickness of ice on water, covering lake ice as well as sea ice.",
        ),
        root_var(
            "temperature_2m",
            element="TMP",
            level="2 m above ground",
            short_name="2t",
            long_name="2 metre temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        root_var(
            "specific_humidity_2m",
            element="SPFH",
            level="2 m above ground",
            short_name="q",
            long_name="Specific humidity",
            units="1",
            standard_name="specific_humidity",
        ),
        root_var(
            "dew_point_temperature_2m",
            element="DPT",
            level="2 m above ground",
            short_name="2d",
            long_name="2 metre dewpoint temperature",
            units="degree_Celsius",
            standard_name="dew_point_temperature",
        ),
        root_var(
            "relative_humidity_2m",
            element="RH",
            level="2 m above ground",
            short_name="2r",
            long_name="2 metre relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        root_var(
            "apparent_temperature_2m",
            element="APTMP",
            level="2 m above ground",
            short_name="aptmp",
            long_name="Apparent temperature",
            units="degree_Celsius",
        ),
        root_var(
            "maximum_temperature_2m",
            element="TMAX",
            level="2 m above ground",
            window="max",
            short_name="tmax",
            long_name="Maximum temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        root_var(
            "minimum_temperature_2m",
            element="TMIN",
            level="2 m above ground",
            window="min",
            short_name="tmin",
            long_name="Minimum temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        root_var(
            "wind_u_10m",
            element="UGRD",
            level="10 m above ground",
            short_name="10u",
            long_name="10 metre U wind component",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        root_var(
            "wind_v_10m",
            element="VGRD",
            level="10 m above ground",
            short_name="10v",
            long_name="10 metre V wind component",
            units="m s-1",
            standard_name="northward_wind",
        ),
        root_var(
            "ice_growth_rate_10m_amsl",
            element="ICEG",
            level="10 m above mean sea level",
            short_name="iceg",
            long_name="Ice growth rate",
            units="m s-1",
            comment="Rate of marine icing accretion on a vessel superstructure.",
        ),
        root_var(
            "percent_frozen_precipitation_surface",
            element="CPOFP",
            level="surface",
            short_name="cpofp",
            long_name="Percent frozen precipitation",
            units="percent",
            comment=(
                "Negative values mark no precipitation. Interpolation in the source "
                "mixes the no data value with real percentages, so unusable values span "
                "a range rather than one value and are not converted to NaN. Mask values "
                "< -0.1."
            ),
        ),
        root_var(
            "instantaneous_precipitation_convective_surface",
            element="CPRAT",
            level="surface",
            short_name="cpr",
            long_name="Convective precipitation rate",
            units="kg m-2 s-1",
            standard_name="convective_precipitation_flux",
            hour_0=False,
        ),
        root_var(
            "precipitation_rate_surface",
            element="PRATE",
            level="surface",
            short_name="prate",
            long_name="Precipitation rate",
            units="kg m-2 s-1",
            standard_name="precipitation_flux",
        ),
        root_var(
            "precipitation_convective_surface",
            element="CPRAT",
            level="surface",
            window="avg",
            short_name="cpr",
            long_name="Convective precipitation rate",
            units="kg m-2 s-1",
            standard_name="convective_precipitation_flux",
        ),
        root_var(
            "average_precipitation_rate_surface",
            element="PRATE",
            level="surface",
            window="avg",
            short_name="prate",
            long_name="Precipitation rate",
            units="kg m-2 s-1",
            standard_name="precipitation_flux",
        ),
        root_var(
            "total_precipitation_surface",
            element="APCP",
            level="surface",
            window="acc_6h",
            short_name="tp",
            long_name="Total precipitation",
            units="kg m-2",
            standard_name="precipitation_amount",
        ),
        root_var(
            "total_precipitation_run_total_surface",
            element="APCP",
            level="surface",
            window="acc_run",
            short_name="tp",
            long_name="Total precipitation",
            units="kg m-2",
            standard_name="precipitation_amount",
        ),
        root_var(
            "convective_precipitation_surface",
            element="ACPCP",
            level="surface",
            window="acc_6h",
            short_name="acpcp",
            long_name="Convective precipitation (water)",
            units="kg m-2",
            standard_name="convective_precipitation_amount",
        ),
        root_var(
            "convective_precipitation_run_total_surface",
            element="ACPCP",
            level="surface",
            window="acc_run",
            short_name="acpcp",
            long_name="Convective precipitation (water)",
            units="kg m-2",
            standard_name="convective_precipitation_amount",
        ),
        root_var(
            "water_runoff_surface",
            element="WATR",
            level="surface",
            window="acc_6h",
            short_name="watr",
            long_name="Water runoff",
            units="kg m-2",
            standard_name="runoff_amount",
            comment="NaN over water, where this quantity does not apply.",
        ),
        root_var(
            "instantaneous_categorical_snow_surface",
            element="CSNOW",
            level="surface",
            short_name="csnow",
            long_name="Categorical snow",
            units="1",
            comment="0=no; 1=yes.",
            flag_values=(0, 1),
            flag_meanings="no yes",
        ),
        root_var(
            "instantaneous_categorical_ice_pellets_surface",
            element="CICEP",
            level="surface",
            short_name="cicep",
            long_name="Categorical ice pellets",
            units="1",
            comment="0=no; 1=yes.",
            flag_values=(0, 1),
            flag_meanings="no yes",
        ),
        root_var(
            "instantaneous_categorical_freezing_rain_surface",
            element="CFRZR",
            level="surface",
            short_name="cfrzr",
            long_name="Categorical freezing rain",
            units="1",
            comment="0=no; 1=yes.",
            flag_values=(0, 1),
            flag_meanings="no yes",
        ),
        root_var(
            "instantaneous_categorical_rain_surface",
            element="CRAIN",
            level="surface",
            short_name="crain",
            long_name="Categorical rain",
            units="1",
            comment="0=no; 1=yes.",
            flag_values=(0, 1),
            flag_meanings="no yes",
        ),
        root_var(
            "categorical_snow_surface",
            element="CSNOW",
            level="surface",
            window="avg",
            short_name="csnow",
            long_name="Categorical snow",
            units="1",
            comment="0=no; 1=yes.",
            flag_values=(0, 1),
            flag_meanings="no yes",
        ),
        root_var(
            "categorical_ice_pellets_surface",
            element="CICEP",
            level="surface",
            window="avg",
            short_name="cicep",
            long_name="Categorical ice pellets",
            units="1",
            comment="0=no; 1=yes.",
            flag_values=(0, 1),
            flag_meanings="no yes",
        ),
        root_var(
            "categorical_freezing_rain_surface",
            element="CFRZR",
            level="surface",
            window="avg",
            short_name="cfrzr",
            long_name="Categorical freezing rain",
            units="1",
            comment="0=no; 1=yes.",
            flag_values=(0, 1),
            flag_meanings="no yes",
        ),
        root_var(
            "categorical_rain_surface",
            element="CRAIN",
            level="surface",
            window="avg",
            short_name="crain",
            long_name="Categorical rain",
            units="1",
            comment="0=no; 1=yes.",
            flag_values=(0, 1),
            flag_meanings="no yes",
        ),
        root_var(
            "latent_heat_flux_surface",
            element="LHTFL",
            level="surface",
            window="avg",
            short_name="lhf",
            long_name="Latent heat flux",
            units="W m-2",
            standard_name="surface_upward_latent_heat_flux",
        ),
        root_var(
            "sensible_heat_flux_surface",
            element="SHTFL",
            level="surface",
            window="avg",
            short_name="shf",
            long_name="Sensible heat flux",
            units="W m-2",
            standard_name="surface_upward_sensible_heat_flux",
        ),
        root_var(
            "ground_heat_flux_surface",
            element="GFLUX",
            level="surface",
            window="avg",
            short_name="gflux",
            long_name="Ground heat flux",
            units="W m-2",
            standard_name="upward_heat_flux_at_ground_level_in_soil",
            comment=(
                "NaN over open water; the source reports this quantity only over land "
                "and sea ice."
            ),
        ),
        root_var(
            "momentum_flux_u_component_surface",
            element="UFLX",
            level="surface",
            window="avg",
            short_name="uflx",
            long_name="Momentum flux, u-component",
            units="Pa",
            comment=(
                "Positive values are an upward flux of eastward momentum, from the "
                "surface into the atmosphere: the opposite sign convention to CF's "
                "surface_downward_eastward_stress."
            ),
        ),
        root_var(
            "momentum_flux_v_component_surface",
            element="VFLX",
            level="surface",
            window="avg",
            short_name="vflx",
            long_name="Momentum flux, v-component",
            units="Pa",
            comment=(
                "Positive values are an upward flux of northward momentum, from the "
                "surface into the atmosphere: the opposite sign convention to CF's "
                "surface_downward_northward_stress."
            ),
        ),
        root_var(
            "surface_roughness_surface",
            element="SFCR",
            level="surface",
            short_name="fsr",
            long_name="Forecast surface roughness",
            units="m",
            standard_name="surface_roughness_length",
        ),
        root_var(
            "friction_velocity_surface",
            element="FRICV",
            level="surface",
            short_name="zust",
            long_name="Friction velocity",
            units="m s-1",
            standard_name="magnitude_of_surface_friction_velocity_in_air",
        ),
        root_var(
            "eastward_gravity_wave_surface_stress",
            element="U-GWD",
            level="surface",
            window="avg",
            short_name="lgws",
            long_name="Eastward gravity wave surface stress",
            units="Pa",
            standard_name="atmosphere_eastward_stress_due_to_gravity_wave_drag",
        ),
        root_var(
            "northward_gravity_wave_surface_stress",
            element="V-GWD",
            level="surface",
            window="avg",
            short_name="mgws",
            long_name="Northward gravity wave surface stress",
            units="Pa",
            standard_name="atmosphere_northward_stress_due_to_gravity_wave_drag",
        ),
        root_var(
            "vegetation_surface",
            element="VEG",
            level="surface",
            short_name="veg",
            long_name="Vegetation fraction",
            units="percent",
            standard_name="vegetation_area_fraction",
            comment="NaN over water, where this quantity does not apply.",
        ),
        root_var(
            "soil_type_surface",
            element="SOTYP",
            level="surface",
            short_name="slt",
            long_name="Soil type",
            units="1",
            standard_name="soil_type",
            comment=(
                "Soil texture class from the 16 category STATSGO classification used by "
                "the GFS Noah land surface model, and 0 over water. Interpolation to "
                "this grid leaves values between the integer class codes, so a value is "
                "not exactly a class number."
            ),
        ),
        root_var(
            "wilting_point_surface",
            element="WILT",
            level="surface",
            short_name="wilt",
            long_name="Wilting point",
            units="1",
            standard_name="volume_fraction_of_condensed_water_in_soil_at_wilting_point",
            comment="NaN over water, where this quantity does not apply.",
        ),
        root_var(
            "field_capacity_surface",
            element="FLDCP",
            level="surface",
            short_name="fldcp",
            long_name="Field Capacity",
            units="1",
            standard_name=(
                "volume_fraction_of_condensed_water_in_soil_at_field_capacity"
            ),
            comment="NaN over water, where this quantity does not apply.",
        ),
        root_var(
            "sunshine_duration_surface",
            element="SUNSD",
            level="surface",
            short_name="SUNSD",
            long_name="Sunshine Duration",
            units="s",
            standard_name="duration_of_sunshine",
            comment=(
                "Sunshine accumulated within the 6 hour window containing this step, "
                "not an instantaneous value: the total restarts every 6 hours of "
                "forecast lead time and so reaches at most 21600 s. The source index "
                "labels it instantaneous, which is why it carries no window step type."
            ),
        ),
        root_var(
            "surface_lifted_index_surface",
            element="LFTX",
            level="surface",
            short_name="lftx",
            long_name="Surface lifted index",
            units="K",
            standard_name=(
                "temperature_difference_between_ambient_air_and_air_lifted_adiabatically_from_the_surface"
            ),
            comment=(
                "A temperature difference, so kelvin rather than the degree_Celsius the "
                "absolute temperature variables carry."
            ),
        ),
        root_var(
            "convective_available_potential_energy_surface",
            element="CAPE",
            level="surface",
            short_name="cape",
            long_name="Convective available potential energy",
            units="J kg-1",
            standard_name="atmosphere_convective_available_potential_energy",
        ),
        root_var(
            "convective_inhibition_surface",
            element="CIN",
            level="surface",
            short_name="cin",
            long_name="Convective inhibition",
            units="J kg-1",
            standard_name="atmosphere_convective_inhibition",
        ),
        root_var(
            "precipitable_water_atmosphere",
            element="PWAT",
            level="entire atmosphere (considered as a single layer)",
            short_name="pwat",
            long_name="Precipitable water",
            units="kg m-2",
            standard_name="atmosphere_mass_content_of_water_vapor",
        ),
        root_var(
            "cloud_water_atmosphere",
            element="CWAT",
            level="entire atmosphere (considered as a single layer)",
            short_name="cwat",
            long_name="Cloud water",
            units="kg m-2",
            standard_name="atmosphere_mass_content_of_cloud_condensed_water",
        ),
        root_var(
            "relative_humidity_atmosphere",
            element="RH",
            level="entire atmosphere (considered as a single layer)",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        root_var(
            "total_ozone_atmosphere",
            element="TOZNE",
            level="entire atmosphere (considered as a single layer)",
            short_name="tozne",
            long_name="Total ozone",
            units="DU",
            comment=(
                "Dobson units; 1 DU is a 10 um thick layer of pure ozone at standard "
                "temperature and pressure."
            ),
        ),
        root_var(
            "low_cloud_cover",
            element="LCDC",
            level="low cloud layer",
            short_name="lcc",
            long_name="Low cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
        ),
        root_var(
            "average_low_cloud_cover",
            element="LCDC",
            level="low cloud layer",
            window="avg",
            short_name="lcc",
            long_name="Low cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
        ),
        root_var(
            "medium_cloud_cover",
            element="MCDC",
            level="middle cloud layer",
            short_name="mcc",
            long_name="Medium cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
        ),
        root_var(
            "average_medium_cloud_cover",
            element="MCDC",
            level="middle cloud layer",
            window="avg",
            short_name="mcc",
            long_name="Medium cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
        ),
        root_var(
            "high_cloud_cover",
            element="HCDC",
            level="high cloud layer",
            short_name="hcc",
            long_name="High cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
        ),
        root_var(
            "average_high_cloud_cover",
            element="HCDC",
            level="high cloud layer",
            window="avg",
            short_name="hcc",
            long_name="High cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
        ),
        root_var(
            "instantaneous_total_cloud_cover_atmosphere",
            element="TCDC",
            level="entire atmosphere",
            short_name="tcc",
            long_name="Total cloud cover",
            units="percent",
            standard_name="cloud_area_fraction",
        ),
        root_var(
            "total_cloud_cover_atmosphere",
            element="TCDC",
            level="entire atmosphere",
            window="avg",
            short_name="tcc",
            long_name="Total cloud cover",
            units="percent",
            standard_name="cloud_area_fraction",
        ),
        root_var(
            "geopotential_height_cloud_ceiling",
            element="HGT",
            level="cloud ceiling",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            comment=(
                "Values near 20,000m mark no cloud ceiling. Interpolation in the source "
                "mixes the no data value with real ceiling heights, so unusable values "
                "span a range rather than one value and are not converted to NaN. Mask "
                "values above 19,000m."
            ),
        ),
        root_var(
            "pressure_convective_cloud_bottom",
            element="PRES",
            level="convective cloud bottom level",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure_at_convective_cloud_base",
            comment="NaN where the source reports no convective cloud in the column. Every cell "
            "where convective_cloud_cover is zero is NaN, as are some cells at the edge "
            "of a cloud field where it is small.",
            hour_0=False,
        ),
        root_var(
            "average_pressure_low_cloud_bottom",
            element="PRES",
            level="low cloud bottom level",
            window="avg",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure_at_cloud_base",
            comment="NaN where the source reports no low cloud in the column. Every cell "
            "where average_low_cloud_cover is zero is NaN, as are some cells at the edge "
            "of a cloud field where it is small.",
        ),
        root_var(
            "average_pressure_middle_cloud_bottom",
            element="PRES",
            level="middle cloud bottom level",
            window="avg",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure_at_cloud_base",
            comment="NaN where the source reports no middle cloud in the column. Every cell "
            "where average_medium_cloud_cover is zero is NaN, as are some cells at the edge "
            "of a cloud field where it is small.",
        ),
        root_var(
            "average_pressure_high_cloud_bottom",
            element="PRES",
            level="high cloud bottom level",
            window="avg",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure_at_cloud_base",
            comment="NaN where the source reports no high cloud in the column. Every cell "
            "where average_high_cloud_cover is zero is NaN, as are some cells at the edge "
            "of a cloud field where it is small.",
        ),
        root_var(
            "pressure_convective_cloud_top",
            element="PRES",
            level="convective cloud top level",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure_at_convective_cloud_top",
            comment="NaN where the source reports no convective cloud in the column. Every cell "
            "where convective_cloud_cover is zero is NaN, as are some cells at the edge "
            "of a cloud field where it is small.",
            hour_0=False,
        ),
        root_var(
            "average_pressure_low_cloud_top",
            element="PRES",
            level="low cloud top level",
            window="avg",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure_at_cloud_top",
            comment="NaN where the source reports no low cloud in the column. Every cell "
            "where average_low_cloud_cover is zero is NaN, as are some cells at the edge "
            "of a cloud field where it is small.",
        ),
        root_var(
            "average_pressure_middle_cloud_top",
            element="PRES",
            level="middle cloud top level",
            window="avg",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure_at_cloud_top",
            comment="NaN where the source reports no middle cloud in the column. Every cell "
            "where average_medium_cloud_cover is zero is NaN, as are some cells at the edge "
            "of a cloud field where it is small.",
        ),
        root_var(
            "average_pressure_high_cloud_top",
            element="PRES",
            level="high cloud top level",
            window="avg",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure_at_cloud_top",
            comment="NaN where the source reports no high cloud in the column. Every cell "
            "where average_high_cloud_cover is zero is NaN, as are some cells at the edge "
            "of a cloud field where it is small.",
        ),
        root_var(
            "average_temperature_low_cloud_top",
            element="TMP",
            level="low cloud top level",
            window="avg",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature_at_cloud_top",
            comment="NaN where the source reports no low cloud in the column. Every cell "
            "where average_low_cloud_cover is zero is NaN, as are some cells at the edge "
            "of a cloud field where it is small.",
        ),
        root_var(
            "average_temperature_middle_cloud_top",
            element="TMP",
            level="middle cloud top level",
            window="avg",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature_at_cloud_top",
            comment="NaN where the source reports no middle cloud in the column. Every cell "
            "where average_medium_cloud_cover is zero is NaN, as are some cells at the edge "
            "of a cloud field where it is small.",
        ),
        root_var(
            "average_temperature_high_cloud_top",
            element="TMP",
            level="high cloud top level",
            window="avg",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature_at_cloud_top",
            comment="NaN where the source reports no high cloud in the column. Every cell "
            "where average_high_cloud_cover is zero is NaN, as are some cells at the edge "
            "of a cloud field where it is small.",
        ),
        root_var(
            "convective_cloud_cover",
            element="TCDC",
            level="convective cloud layer",
            short_name="ccc",
            long_name="Convective cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
            hour_0=False,
        ),
        root_var(
            "average_total_cloud_cover_boundary_layer",
            element="TCDC",
            level="boundary layer cloud layer",
            window="avg",
            short_name="tcc",
            long_name="Boundary layer cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
        ),
        root_var(
            "cloud_work_function_atmosphere",
            element="CWORK",
            level="entire atmosphere (considered as a single layer)",
            window="avg",
            short_name="cwork",
            long_name="Cloud work function",
            units="J kg-1",
        ),
        root_var(
            "downward_short_wave_radiation_flux_surface",
            element="DSWRF",
            level="surface",
            window="avg",
            short_name="sdswrf",
            long_name="Surface downward short-wave radiation flux",
            units="W m-2",
            standard_name="surface_downwelling_shortwave_flux_in_air",
        ),
        root_var(
            "downward_long_wave_radiation_flux_surface",
            element="DLWRF",
            level="surface",
            window="avg",
            short_name="sdlwrf",
            long_name="Surface downward long-wave radiation flux",
            units="W m-2",
            standard_name="surface_downwelling_longwave_flux_in_air",
        ),
        root_var(
            "upward_short_wave_radiation_flux_surface",
            element="USWRF",
            level="surface",
            window="avg",
            short_name="suswrf",
            long_name="Surface upward short-wave radiation flux",
            units="W m-2",
            standard_name="surface_upwelling_shortwave_flux_in_air",
        ),
        root_var(
            "upward_long_wave_radiation_flux_surface",
            element="ULWRF",
            level="surface",
            window="avg",
            short_name="sulwrf",
            long_name="Surface upward long-wave radiation flux",
            units="W m-2",
            standard_name="surface_upwelling_longwave_flux_in_air",
        ),
        root_var(
            "upward_short_wave_radiation_flux_top_of_atmosphere",
            element="USWRF",
            level="top of atmosphere",
            window="avg",
            short_name="uswrf",
            long_name="Upward short-wave radiation flux",
            units="W m-2",
            standard_name="toa_outgoing_shortwave_flux",
        ),
        root_var(
            "upward_long_wave_radiation_flux_top_of_atmosphere",
            element="ULWRF",
            level="top of atmosphere",
            window="avg",
            short_name="ulwrf",
            long_name="Upward long-wave radiation flux",
            units="W m-2",
            standard_name="toa_outgoing_longwave_flux",
        ),
        root_var(
            "storm_relative_helicity_3000_0m",
            element="HLCY",
            level="3000-0 m above ground",
            short_name="hlcy",
            long_name="Storm relative helicity",
            units="m2 s-2",
        ),
        root_var(
            "u_component_storm_motion_6000_0m",
            element="USTM",
            level="6000-0 m above ground",
            short_name="ustm",
            long_name="U-component storm motion",
            units="m s-1",
        ),
        root_var(
            "v_component_storm_motion_6000_0m",
            element="VSTM",
            level="6000-0 m above ground",
            short_name="vstm",
            long_name="V-component storm motion",
            units="m s-1",
        ),
        root_var(
            "pressure_tropopause",
            element="PRES",
            level="tropopause",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="tropopause_air_pressure",
        ),
        root_var(
            "icao_standard_atmosphere_reference_height_tropopause",
            element="ICAHT",
            level="tropopause",
            short_name="icaht",
            long_name="ICAO Standard Atmosphere reference height",
            units="m",
            comment=(
                "Pressure altitude: the height at which the ICAO Standard Atmosphere "
                "reaches the pressure found here, not a geometric height."
            ),
        ),
        root_var(
            "geopotential_height_tropopause",
            element="HGT",
            level="tropopause",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
        ),
        root_var(
            "temperature_tropopause",
            element="TMP",
            level="tropopause",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        root_var(
            "wind_u_tropopause",
            element="UGRD",
            level="tropopause",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        root_var(
            "wind_v_tropopause",
            element="VGRD",
            level="tropopause",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        root_var(
            "vertical_speed_shear_tropopause",
            element="VWSH",
            level="tropopause",
            short_name="vwsh",
            long_name="Vertical speed shear",
            units="s-1",
            standard_name="wind_speed_shear",
        ),
        root_var(
            "pressure_max_wind",
            element="PRES",
            level="max wind",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure",
        ),
        root_var(
            "icao_standard_atmosphere_reference_height_max_wind",
            element="ICAHT",
            level="max wind",
            short_name="icaht",
            long_name="ICAO Standard Atmosphere reference height",
            units="m",
            comment=(
                "Pressure altitude: the height at which the ICAO Standard Atmosphere "
                "reaches the pressure found here, not a geometric height."
            ),
        ),
        root_var(
            "geopotential_height_max_wind",
            element="HGT",
            level="max wind",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
        ),
        root_var(
            "wind_u_max_wind",
            element="UGRD",
            level="max wind",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        root_var(
            "wind_v_max_wind",
            element="VGRD",
            level="max wind",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        root_var(
            "temperature_max_wind",
            element="TMP",
            level="max wind",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        root_var(
            "wind_u_20m",
            element="UGRD",
            level="20 m above ground",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        root_var(
            "wind_v_20m",
            element="VGRD",
            level="20 m above ground",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        root_var(
            "wind_u_30m",
            element="UGRD",
            level="30 m above ground",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        root_var(
            "wind_v_30m",
            element="VGRD",
            level="30 m above ground",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        root_var(
            "wind_u_40m",
            element="UGRD",
            level="40 m above ground",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        root_var(
            "wind_v_40m",
            element="VGRD",
            level="40 m above ground",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        root_var(
            "wind_u_50m",
            element="UGRD",
            level="50 m above ground",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        root_var(
            "wind_v_50m",
            element="VGRD",
            level="50 m above ground",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        root_var(
            "temperature_80m",
            element="TMP",
            level="80 m above ground",
            short_name="80t",
            long_name="80 metre temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        root_var(
            "specific_humidity_80m",
            element="SPFH",
            level="80 m above ground",
            short_name="q",
            long_name="Specific humidity",
            units="1",
            standard_name="specific_humidity",
        ),
        root_var(
            "pressure_80m",
            element="PRES",
            level="80 m above ground",
            short_name="80sp",
            long_name="80 metre pressure",
            units="Pa",
            standard_name="air_pressure",
        ),
        root_var(
            "wind_u_80m",
            element="UGRD",
            level="80 m above ground",
            short_name="80u",
            long_name="80 metre U wind component",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        root_var(
            "wind_v_80m",
            element="VGRD",
            level="80 m above ground",
            short_name="80v",
            long_name="80 metre V wind component",
            units="m s-1",
            standard_name="northward_wind",
        ),
        root_var(
            "temperature_100m",
            element="TMP",
            level="100 m above ground",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        root_var(
            "wind_u_100m",
            element="UGRD",
            level="100 m above ground",
            short_name="100u",
            long_name="100 metre U wind component",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        root_var(
            "wind_v_100m",
            element="VGRD",
            level="100 m above ground",
            short_name="100v",
            long_name="100 metre V wind component",
            units="m s-1",
            standard_name="northward_wind",
        ),
        root_var(
            "geopotential_height_0c_isotherm",
            element="HGT",
            level="0C isotherm",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            comment=(
                "Zero marks a freezing level at or below the surface rather than "
                "missing data; over water, where the surface is at zero height, it "
                "is a genuine value."
            ),
        ),
        root_var(
            "relative_humidity_0c_isotherm",
            element="RH",
            level="0C isotherm",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        root_var(
            "geopotential_height_highest_tropospheric_freezing_level",
            element="HGT",
            level="highest tropospheric freezing level",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            comment=(
                "Zero marks a freezing level at or below the surface rather than "
                "missing data; over water, where the surface is at zero height, it "
                "is a genuine value."
            ),
        ),
        root_var(
            "relative_humidity_highest_tropospheric_freezing_level",
            element="RH",
            level="highest tropospheric freezing level",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        root_var(
            "temperature_30_0mb",
            element="TMP",
            level="30-0 mb above ground",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        root_var(
            "relative_humidity_30_0mb",
            element="RH",
            level="30-0 mb above ground",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        root_var(
            "specific_humidity_30_0mb",
            element="SPFH",
            level="30-0 mb above ground",
            short_name="q",
            long_name="Specific humidity",
            units="1",
            standard_name="specific_humidity",
        ),
        root_var(
            "wind_u_30_0mb",
            element="UGRD",
            level="30-0 mb above ground",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        root_var(
            "wind_v_30_0mb",
            element="VGRD",
            level="30-0 mb above ground",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        root_var(
            "best_4_layer_lifted_index_surface",
            element="4LFTX",
            level="surface",
            short_name="4lftx",
            long_name="Best (4-layer) lifted index",
            units="K",
            standard_name=(
                "temperature_difference_between_ambient_air_and_air_lifted_adiabatically"
            ),
            comment=(
                "A temperature difference, so kelvin rather than the degree_Celsius the "
                "absolute temperature variables carry."
            ),
        ),
        root_var(
            "convective_available_potential_energy_180_0mb",
            element="CAPE",
            level="180-0 mb above ground",
            short_name="cape",
            long_name="Convective available potential energy",
            units="J kg-1",
            standard_name="atmosphere_convective_available_potential_energy",
        ),
        root_var(
            "convective_inhibition_180_0mb",
            element="CIN",
            level="180-0 mb above ground",
            short_name="cin",
            long_name="Convective inhibition",
            units="J kg-1",
            standard_name="atmosphere_convective_inhibition",
        ),
        root_var(
            "planetary_boundary_layer_height_surface",
            element="HPBL",
            level="surface",
            short_name="blh",
            long_name="Boundary layer height",
            units="m",
            standard_name="atmosphere_boundary_layer_thickness",
        ),
        root_var(
            "relative_humidity_0p33_1_sigma",
            element="RH",
            level="0.33-1 sigma layer",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        root_var(
            "relative_humidity_0p44_1_sigma",
            element="RH",
            level="0.44-1 sigma layer",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        root_var(
            "relative_humidity_0p72_0p94_sigma",
            element="RH",
            level="0.72-0.94 sigma layer",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        root_var(
            "relative_humidity_0p44_0p72_sigma",
            element="RH",
            level="0.44-0.72 sigma layer",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        root_var(
            "temperature_0p995_sigma",
            element="TMP",
            level="0.995 sigma level",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        root_var(
            "potential_temperature_0p995_sigma",
            element="POT",
            level="0.995 sigma level",
            short_name="pt",
            long_name="Potential temperature",
            units="K",
            standard_name="air_potential_temperature",
            comment=(
                "Potential temperature is conventionally reported in kelvin, so this "
                "variable is not converted to Celsius as the absolute temperatures are."
            ),
        ),
        root_var(
            "relative_humidity_0p995_sigma",
            element="RH",
            level="0.995 sigma level",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        root_var(
            "wind_u_0p995_sigma",
            element="UGRD",
            level="0.995 sigma level",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        root_var(
            "wind_v_0p995_sigma",
            element="VGRD",
            level="0.995 sigma level",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        root_var(
            "vertical_velocity_0p995_sigma",
            element="VVEL",
            level="0.995 sigma level",
            short_name="w",
            long_name="Vertical velocity",
            units="Pa s-1",
            standard_name="lagrangian_tendency_of_air_pressure",
        ),
        root_var(
            "convective_available_potential_energy_90_0mb",
            element="CAPE",
            level="90-0 mb above ground",
            short_name="cape",
            long_name="Convective available potential energy",
            units="J kg-1",
            standard_name="atmosphere_convective_available_potential_energy",
        ),
        root_var(
            "convective_inhibition_90_0mb",
            element="CIN",
            level="90-0 mb above ground",
            short_name="cin",
            long_name="Convective inhibition",
            units="J kg-1",
            standard_name="atmosphere_convective_inhibition",
        ),
        root_var(
            "convective_available_potential_energy_255_0mb",
            element="CAPE",
            level="255-0 mb above ground",
            short_name="cape",
            long_name="Convective available potential energy",
            units="J kg-1",
            standard_name="atmosphere_convective_available_potential_energy",
        ),
        root_var(
            "convective_inhibition_255_0mb",
            element="CIN",
            level="255-0 mb above ground",
            short_name="cin",
            long_name="Convective inhibition",
            units="J kg-1",
            standard_name="atmosphere_convective_inhibition",
        ),
        root_var(
            "pressure_of_lifted_parcel_level_255_0mb",
            element="PLPL",
            level="255-0 mb above ground",
            short_name="plpl",
            long_name="Pressure of level from which parcel was lifted",
            units="Pa",
            standard_name="original_air_pressure_of_lifted_parcel",
        ),
        root_var(
            "land_sea_mask_surface",
            element="LAND",
            level="surface",
            short_name="lsm",
            long_name="Land-sea mask",
            units="1",
            standard_name="land_binary_mask",
            flag_values=(0, 1),
            flag_meanings="sea land",
        ),
        root_var(
            "ice_cover_surface",
            element="ICEC",
            level="surface",
            short_name="icec",
            long_name="Ice cover (1=ice, 0=no ice)",
            units="1",
            standard_name="floating_ice_area_fraction",
            comment=(
                "The fraction of the cell covered by floating ice, taking any value "
                "between 0 and 1 rather than only those two. Covers lake ice as well "
                "as sea ice."
            ),
        ),
        root_var(
            "albedo_surface",
            element="ALBDO",
            level="surface",
            window="avg",
            short_name="fal",
            long_name="Forecast albedo",
            units="percent",
            standard_name="surface_albedo",
        ),
        root_var(
            "ice_temperature_surface",
            element="ICETMP",
            level="surface",
            short_name="sist",
            long_name="Sea ice surface temperature",
            units="degree_Celsius",
            comment=(
                "Temperature of the ice surface, covering lake ice as well as sea ice "
                "despite the source parameter name. NaN where there is no ice."
            ),
        ),
        root_var(
            "wind_u_2pvu",
            element="UGRD",
            level="PV=2e-06 (Km^2/kg/s) surface",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "wind_v_2pvu",
            element="VGRD",
            level="PV=2e-06 (Km^2/kg/s) surface",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "temperature_2pvu",
            element="TMP",
            level="PV=2e-06 (Km^2/kg/s) surface",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "geopotential_height_2pvu",
            element="HGT",
            level="PV=2e-06 (Km^2/kg/s) surface",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "pressure_2pvu",
            element="PRES",
            level="PV=2e-06 (Km^2/kg/s) surface",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "vertical_speed_shear_2pvu",
            element="VWSH",
            level="PV=2e-06 (Km^2/kg/s) surface",
            short_name="vwsh",
            long_name="Vertical speed shear",
            units="s-1",
            standard_name="wind_speed_shear",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "wind_u_minus2pvu",
            element="UGRD",
            level="PV=-2e-06 (Km^2/kg/s) surface",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "wind_v_minus2pvu",
            element="VGRD",
            level="PV=-2e-06 (Km^2/kg/s) surface",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "temperature_minus2pvu",
            element="TMP",
            level="PV=-2e-06 (Km^2/kg/s) surface",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "geopotential_height_minus2pvu",
            element="HGT",
            level="PV=-2e-06 (Km^2/kg/s) surface",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "pressure_minus2pvu",
            element="PRES",
            level="PV=-2e-06 (Km^2/kg/s) surface",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "vertical_speed_shear_minus2pvu",
            element="VWSH",
            level="PV=-2e-06 (Km^2/kg/s) surface",
            short_name="vwsh",
            long_name="Vertical speed shear",
            units="s-1",
            standard_name="wind_speed_shear",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "uv_b_downward_solar_flux_surface",
            element="DUVB",
            level="surface",
            window="avg",
            short_name="duvb",
            long_name="UV-B downward solar flux",
            units="W m-2",
            comment="Downward solar flux in the UV-B band (280-315 nm) at the surface.",
        ),
        root_var(
            "clear_sky_uv_b_downward_solar_flux_surface",
            element="CDUVB",
            level="surface",
            window="avg",
            short_name="cduvb",
            long_name="Clear sky UV-B downward solar flux",
            units="W m-2",
            comment=(
                "Downward solar flux in the UV-B band (280-315 nm) at the surface "
                "computed with clouds removed."
            ),
        ),
        root_var(
            "temperature_60_30mb",
            element="TMP",
            level="60-30 mb above ground",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        root_var(
            "relative_humidity_60_30mb",
            element="RH",
            level="60-30 mb above ground",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        root_var(
            "specific_humidity_60_30mb",
            element="SPFH",
            level="60-30 mb above ground",
            short_name="q",
            long_name="Specific humidity",
            units="1",
            standard_name="specific_humidity",
        ),
        root_var(
            "wind_u_60_30mb",
            element="UGRD",
            level="60-30 mb above ground",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        root_var(
            "wind_v_60_30mb",
            element="VGRD",
            level="60-30 mb above ground",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        root_var(
            "temperature_90_60mb",
            element="TMP",
            level="90-60 mb above ground",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        root_var(
            "relative_humidity_90_60mb",
            element="RH",
            level="90-60 mb above ground",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        root_var(
            "specific_humidity_90_60mb",
            element="SPFH",
            level="90-60 mb above ground",
            short_name="q",
            long_name="Specific humidity",
            units="1",
            standard_name="specific_humidity",
        ),
        root_var(
            "wind_u_90_60mb",
            element="UGRD",
            level="90-60 mb above ground",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        root_var(
            "wind_v_90_60mb",
            element="VGRD",
            level="90-60 mb above ground",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        root_var(
            "temperature_120_90mb",
            element="TMP",
            level="120-90 mb above ground",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        root_var(
            "relative_humidity_120_90mb",
            element="RH",
            level="120-90 mb above ground",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        root_var(
            "specific_humidity_120_90mb",
            element="SPFH",
            level="120-90 mb above ground",
            short_name="q",
            long_name="Specific humidity",
            units="1",
            standard_name="specific_humidity",
        ),
        root_var(
            "wind_u_120_90mb",
            element="UGRD",
            level="120-90 mb above ground",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        root_var(
            "wind_v_120_90mb",
            element="VGRD",
            level="120-90 mb above ground",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        root_var(
            "temperature_150_120mb",
            element="TMP",
            level="150-120 mb above ground",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        root_var(
            "relative_humidity_150_120mb",
            element="RH",
            level="150-120 mb above ground",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        root_var(
            "specific_humidity_150_120mb",
            element="SPFH",
            level="150-120 mb above ground",
            short_name="q",
            long_name="Specific humidity",
            units="1",
            standard_name="specific_humidity",
        ),
        root_var(
            "wind_u_150_120mb",
            element="UGRD",
            level="150-120 mb above ground",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        root_var(
            "wind_v_150_120mb",
            element="VGRD",
            level="150-120 mb above ground",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        root_var(
            "temperature_180_150mb",
            element="TMP",
            level="180-150 mb above ground",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        root_var(
            "relative_humidity_180_150mb",
            element="RH",
            level="180-150 mb above ground",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        root_var(
            "specific_humidity_180_150mb",
            element="SPFH",
            level="180-150 mb above ground",
            short_name="q",
            long_name="Specific humidity",
            units="1",
            standard_name="specific_humidity",
        ),
        root_var(
            "wind_u_180_150mb",
            element="UGRD",
            level="180-150 mb above ground",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        root_var(
            "wind_v_180_150mb",
            element="VGRD",
            level="180-150 mb above ground",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        root_var(
            "wind_u_0p5pvu",
            element="UGRD",
            level="PV=5e-07 (Km^2/kg/s) surface",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "wind_v_0p5pvu",
            element="VGRD",
            level="PV=5e-07 (Km^2/kg/s) surface",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "temperature_0p5pvu",
            element="TMP",
            level="PV=5e-07 (Km^2/kg/s) surface",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "geopotential_height_0p5pvu",
            element="HGT",
            level="PV=5e-07 (Km^2/kg/s) surface",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "pressure_0p5pvu",
            element="PRES",
            level="PV=5e-07 (Km^2/kg/s) surface",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "vertical_speed_shear_0p5pvu",
            element="VWSH",
            level="PV=5e-07 (Km^2/kg/s) surface",
            short_name="vwsh",
            long_name="Vertical speed shear",
            units="s-1",
            standard_name="wind_speed_shear",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "wind_u_minus0p5pvu",
            element="UGRD",
            level="PV=-5e-07 (Km^2/kg/s) surface",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "wind_v_minus0p5pvu",
            element="VGRD",
            level="PV=-5e-07 (Km^2/kg/s) surface",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "temperature_minus0p5pvu",
            element="TMP",
            level="PV=-5e-07 (Km^2/kg/s) surface",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "geopotential_height_minus0p5pvu",
            element="HGT",
            level="PV=-5e-07 (Km^2/kg/s) surface",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "pressure_minus0p5pvu",
            element="PRES",
            level="PV=-5e-07 (Km^2/kg/s) surface",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "vertical_speed_shear_minus0p5pvu",
            element="VWSH",
            level="PV=-5e-07 (Km^2/kg/s) surface",
            short_name="vwsh",
            long_name="Vertical speed shear",
            units="s-1",
            standard_name="wind_speed_shear",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "wind_u_1pvu",
            element="UGRD",
            level="PV=1e-06 (Km^2/kg/s) surface",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "wind_v_1pvu",
            element="VGRD",
            level="PV=1e-06 (Km^2/kg/s) surface",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "temperature_1pvu",
            element="TMP",
            level="PV=1e-06 (Km^2/kg/s) surface",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "geopotential_height_1pvu",
            element="HGT",
            level="PV=1e-06 (Km^2/kg/s) surface",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "pressure_1pvu",
            element="PRES",
            level="PV=1e-06 (Km^2/kg/s) surface",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "vertical_speed_shear_1pvu",
            element="VWSH",
            level="PV=1e-06 (Km^2/kg/s) surface",
            short_name="vwsh",
            long_name="Vertical speed shear",
            units="s-1",
            standard_name="wind_speed_shear",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "wind_u_minus1pvu",
            element="UGRD",
            level="PV=-1e-06 (Km^2/kg/s) surface",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "wind_v_minus1pvu",
            element="VGRD",
            level="PV=-1e-06 (Km^2/kg/s) surface",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "temperature_minus1pvu",
            element="TMP",
            level="PV=-1e-06 (Km^2/kg/s) surface",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "geopotential_height_minus1pvu",
            element="HGT",
            level="PV=-1e-06 (Km^2/kg/s) surface",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "pressure_minus1pvu",
            element="PRES",
            level="PV=-1e-06 (Km^2/kg/s) surface",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "vertical_speed_shear_minus1pvu",
            element="VWSH",
            level="PV=-1e-06 (Km^2/kg/s) surface",
            short_name="vwsh",
            long_name="Vertical speed shear",
            units="s-1",
            standard_name="wind_speed_shear",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "wind_u_1p5pvu",
            element="UGRD",
            level="PV=1.5e-06 (Km^2/kg/s) surface",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "wind_v_1p5pvu",
            element="VGRD",
            level="PV=1.5e-06 (Km^2/kg/s) surface",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "temperature_1p5pvu",
            element="TMP",
            level="PV=1.5e-06 (Km^2/kg/s) surface",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "geopotential_height_1p5pvu",
            element="HGT",
            level="PV=1.5e-06 (Km^2/kg/s) surface",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "pressure_1p5pvu",
            element="PRES",
            level="PV=1.5e-06 (Km^2/kg/s) surface",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "vertical_speed_shear_1p5pvu",
            element="VWSH",
            level="PV=1.5e-06 (Km^2/kg/s) surface",
            short_name="vwsh",
            long_name="Vertical speed shear",
            units="s-1",
            standard_name="wind_speed_shear",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "wind_u_minus1p5pvu",
            element="UGRD",
            level="PV=-1.5e-06 (Km^2/kg/s) surface",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "wind_v_minus1p5pvu",
            element="VGRD",
            level="PV=-1.5e-06 (Km^2/kg/s) surface",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "temperature_minus1p5pvu",
            element="TMP",
            level="PV=-1.5e-06 (Km^2/kg/s) surface",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "geopotential_height_minus1p5pvu",
            element="HGT",
            level="PV=-1.5e-06 (Km^2/kg/s) surface",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "pressure_minus1p5pvu",
            element="PRES",
            level="PV=-1.5e-06 (Km^2/kg/s) surface",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "vertical_speed_shear_minus1p5pvu",
            element="VWSH",
            level="PV=-1.5e-06 (Km^2/kg/s) surface",
            short_name="vwsh",
            long_name="Vertical speed shear",
            units="s-1",
            standard_name="wind_speed_shear",
            comment=(
                "NaN where this potential vorticity surface does not exist in the "
                "column."
            ),
        ),
        root_var(
            "cloud_mixing_ratio_model_level_1",
            element="CLMR",
            element_alternatives=("CLWMR",),
            level="1 hybrid level",
            short_name="clwmr",
            long_name="Cloud mixing ratio",
            units="kg kg-1",
            standard_name="cloud_liquid_water_mixing_ratio",
            comment=(
                "GFS model level 1, the lowest native hybrid sigma-pressure layer, "
                "immediately above the ground."
            ),
        ),
        root_var(
            "ice_water_mixing_ratio_model_level_1",
            element="ICMR",
            level="1 hybrid level",
            short_name="icmr",
            long_name="Ice water mixing ratio",
            units="kg kg-1",
            standard_name="cloud_ice_mixing_ratio",
            comment=(
                "GFS model level 1, the lowest native hybrid sigma-pressure layer, "
                "immediately above the ground."
            ),
        ),
        root_var(
            "rain_mixing_ratio_model_level_1",
            element="RWMR",
            level="1 hybrid level",
            short_name="rwmr",
            long_name="Rain mixing ratio",
            units="kg kg-1",
            comment=(
                "GFS model level 1, the lowest native hybrid sigma-pressure layer, "
                "immediately above the ground."
            ),
        ),
        root_var(
            "snow_mixing_ratio_model_level_1",
            element="SNMR",
            level="1 hybrid level",
            short_name="snmr",
            long_name="Snow mixing ratio",
            units="kg kg-1",
            comment=(
                "GFS model level 1, the lowest native hybrid sigma-pressure layer, "
                "immediately above the ground."
            ),
        ),
        root_var(
            "graupel_model_level_1",
            element="GRLE",
            level="1 hybrid level",
            short_name="grle",
            long_name="Graupel (snow pellets)",
            units="kg kg-1",
            comment=(
                "GFS model level 1, the lowest native hybrid sigma-pressure layer, "
                "immediately above the ground."
            ),
        ),
        root_var(
            "derived_radar_reflectivity_model_level_1",
            element="REFD",
            level="1 hybrid level",
            short_name="refd",
            long_name="Derived radar reflectivity",
            units="dBZ",
            standard_name="equivalent_reflectivity_factor",
            comment=(
                "GFS model level 1, the lowest native hybrid sigma-pressure layer, immediately above the ground. "
                "-20 dBZ is the source's no-echo floor: those cells mean no echo "
                "was detected, not a measured value."
            ),
        ),
        root_var(
            "derived_radar_reflectivity_model_level_2",
            element="REFD",
            level="2 hybrid level",
            short_name="refd",
            long_name="Derived radar reflectivity",
            units="dBZ",
            standard_name="equivalent_reflectivity_factor",
            comment=(
                "GFS model level 2, the second native hybrid sigma-pressure layer above the ground. "
                "-20 dBZ is the source's no-echo floor: those cells mean no echo "
                "was detected, not a measured value."
            ),
        ),
    ]


def _pressure_data_vars(chunks: tuple[int, ...]) -> list[NoaaDataVar]:
    pressure_var = functools.partial(_pressure_var, chunks=chunks)
    return [
        pressure_var(
            "geopotential_height",
            element="HGT",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
        ),
        pressure_var(
            "temperature",
            element="TMP",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        pressure_var(
            "relative_humidity",
            element="RH",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        pressure_var(
            "specific_humidity",
            element="SPFH",
            short_name="q",
            long_name="Specific humidity",
            units="1",
            standard_name="specific_humidity",
            comment=(
                "Not published at the 16 levels 125, 175, 225, 275, 325, 375, 425, 475, "
                "525, 575, 625, 675, 725, 775, 825 and 875 hPa, which are NaN at every "
                "step."
            ),
        ),
        pressure_var(
            "vertical_velocity",
            element="VVEL",
            short_name="w",
            long_name="Vertical velocity",
            units="Pa s-1",
            standard_name="lagrangian_tendency_of_air_pressure",
        ),
        pressure_var(
            "vertical_velocity_geometric",
            element="DZDT",
            short_name="dzdt",
            long_name="Vertical velocity (geometric)",
            units="m s-1",
            standard_name="upward_air_velocity",
        ),
        pressure_var(
            "wind_u",
            element="UGRD",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        pressure_var(
            "wind_v",
            element="VGRD",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        pressure_var(
            "absolute_vorticity",
            element="ABSV",
            short_name="absv",
            long_name="Absolute vorticity",
            units="s-1",
            standard_name="atmosphere_upward_absolute_vorticity",
        ),
        pressure_var(
            "ozone_mixing_ratio",
            element="O3MR",
            short_name="o3mr",
            long_name="Ozone mixing ratio",
            units="kg kg-1",
            standard_name="mass_fraction_of_ozone_in_air",
            comment=(
                "Not published at the 16 levels 125, 175, 225, 275, 325, 375, 425, 475, "
                "525, 575, 625, 675, 725, 775, 825 and 875 hPa, which are NaN at every "
                "step."
            ),
        ),
        pressure_var(
            "cloud_cover",
            element="TCDC",
            short_name="cc",
            long_name="Fraction of cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
            comment=(
                "Published only from 1000 to 50 hPa. The 18 levels above 50 hPa (40, 30, "
                "20, 15, 10, 7, 5, 3, 2, 1, 0.7, 0.4, 0.2, 0.1, 0.07, 0.04, 0.02 and "
                "0.01 hPa) are NaN at every step."
            ),
        ),
        pressure_var(
            "cloud_mixing_ratio",
            element="CLMR",
            element_alternatives=("CLWMR",),
            short_name="clwmr",
            long_name="Cloud mixing ratio",
            units="kg kg-1",
            standard_name="cloud_liquid_water_mixing_ratio",
            comment=(
                "Published only from 1000 to 50 hPa. The 18 levels above 50 hPa (40, 30, "
                "20, 15, 10, 7, 5, 3, 2, 1, 0.7, 0.4, 0.2, 0.1, 0.07, 0.04, 0.02 and "
                "0.01 hPa) are NaN at every step."
            ),
        ),
        pressure_var(
            "ice_water_mixing_ratio",
            element="ICMR",
            short_name="icmr",
            long_name="Ice water mixing ratio",
            units="kg kg-1",
            standard_name="cloud_ice_mixing_ratio",
            comment=(
                "Published only from 1000 to 50 hPa. The 18 levels above 50 hPa (40, 30, "
                "20, 15, 10, 7, 5, 3, 2, 1, 0.7, 0.4, 0.2, 0.1, 0.07, 0.04, 0.02 and "
                "0.01 hPa) are NaN at every step."
            ),
        ),
        pressure_var(
            "rain_mixing_ratio",
            element="RWMR",
            short_name="rwmr",
            long_name="Rain mixing ratio",
            units="kg kg-1",
            comment=(
                "Published only from 1000 to 50 hPa. The 18 levels above 50 hPa (40, 30, "
                "20, 15, 10, 7, 5, 3, 2, 1, 0.7, 0.4, 0.2, 0.1, 0.07, 0.04, 0.02 and "
                "0.01 hPa) are NaN at every step."
            ),
        ),
        pressure_var(
            "snow_mixing_ratio",
            element="SNMR",
            short_name="snmr",
            long_name="Snow mixing ratio",
            units="kg kg-1",
            comment=(
                "Published only from 1000 to 50 hPa. The 18 levels above 50 hPa (40, 30, "
                "20, 15, 10, 7, 5, 3, 2, 1, 0.7, 0.4, 0.2, 0.1, 0.07, 0.04, 0.02 and "
                "0.01 hPa) are NaN at every step."
            ),
        ),
        pressure_var(
            "graupel",
            element="GRLE",
            short_name="grle",
            long_name="Graupel (snow pellets)",
            units="kg kg-1",
            comment=(
                "Published only from 1000 to 50 hPa. The 18 levels above 50 hPa (40, 30, "
                "20, 15, 10, 7, 5, 3, 2, 1, 0.7, 0.4, 0.2, 0.1, 0.07, 0.04, 0.02 and "
                "0.01 hPa) are NaN at every step."
            ),
        ),
    ]
