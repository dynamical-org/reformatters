from collections.abc import Sequence

import numpy as np
import pandas as pd
from gribberish.zarr import GribberishCodec

from reformatters.common.config_models import ROOT, DataVarAttrs, Encoding
from reformatters.common.pydantic import FrozenBaseModel, replace
from reformatters.common.types import CodecConfig, Dims
from reformatters.noaa.rrfs.variables import VARIABLE_DEFINITIONS

from .models import (
    NoaaRefsDataVar,
    NoaaRefsInternalAttrs,
    ProductFamily,
    supported_leads,
)


class ProductField(FrozenBaseModel):
    name: str
    element: str
    level: str
    attrs: DataVarAttrs
    filters: tuple[CodecConfig, ...] = ()


_RRFS_FIELDS = {
    d.name if d.group is ROOT else f"{d.group}/{d.name}": d
    for d in VARIABLE_DEFINITIONS
}


def _rrfs_field(path: str, *, name: str, level: str) -> ProductField:
    definition = _RRFS_FIELDS[path]
    comment = None
    if definition.element in ("CAPE", "CIN") and level in (
        "180-0 mb above ground",
        "90-0 mb above ground",
    ):
        comment = f"The {level.removesuffix(' above ground')} in the name are pressure differences from the surface, not isobaric levels."
    elif definition.element == "UGRD":
        comment = "Velocity along the model grid's x dimension, not eastward velocity."
    elif definition.element == "VGRD":
        comment = "Velocity along the model grid's y dimension, not northward velocity."
    attrs = replace(definition.attrs, comment=comment, step_type="instant")
    filters = definition.filters
    return ProductField(
        name=name,
        element=definition.element,
        level=level,
        attrs=attrs,
        filters=filters,
    )


_FIELDS = {
    ("APCP", "surface"): _rrfs_field(
        "total_precipitation_surface",
        name="total_precipitation_surface",
        level="surface",
    ),
    ("ASNOW", "surface"): ProductField(
        name="total_snowfall_surface",
        element="ASNOW",
        level="surface",
        attrs=DataVarAttrs(
            short_name="asnow",
            long_name="Total snowfall",
            units="m",
            standard_name="thickness_of_snowfall_amount",
            step_type="instant",
        ),
        filters=(),
    ),
    ("CAPE", "180-0 mb above ground"): _rrfs_field(
        "convective_available_potential_energy_180_0mb",
        name="convective_available_potential_energy_180_0mb",
        level="180-0 mb above ground",
    ),
    ("CAPE", "90-0 mb above ground"): _rrfs_field(
        "convective_available_potential_energy_90_0mb",
        name="convective_available_potential_energy_90_0mb",
        level="90-0 mb above ground",
    ),
    ("CAPE", "surface"): _rrfs_field(
        "convective_available_potential_energy_surface",
        name="convective_available_potential_energy_surface",
        level="surface",
    ),
    ("CAT", "175 mb"): ProductField(
        name="clear_air_turbulence_175hpa",
        element="CAT",
        level="175 mb",
        attrs=DataVarAttrs(
            short_name="cat",
            long_name="Clear air turbulence",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("CAT", "200 mb"): ProductField(
        name="clear_air_turbulence_200hpa",
        element="CAT",
        level="200 mb",
        attrs=DataVarAttrs(
            short_name="cat",
            long_name="Clear air turbulence",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("CAT", "225 mb"): ProductField(
        name="clear_air_turbulence_225hpa",
        element="CAT",
        level="225 mb",
        attrs=DataVarAttrs(
            short_name="cat",
            long_name="Clear air turbulence",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("CAT", "275 mb"): ProductField(
        name="clear_air_turbulence_275hpa",
        element="CAT",
        level="275 mb",
        attrs=DataVarAttrs(
            short_name="cat",
            long_name="Clear air turbulence",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("CAT", "300 mb"): ProductField(
        name="clear_air_turbulence_300hpa",
        element="CAT",
        level="300 mb",
        attrs=DataVarAttrs(
            short_name="cat",
            long_name="Clear air turbulence",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("CAT", "350 mb"): ProductField(
        name="clear_air_turbulence_350hpa",
        element="CAT",
        level="350 mb",
        attrs=DataVarAttrs(
            short_name="cat",
            long_name="Clear air turbulence",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("CAT", "425 mb"): ProductField(
        name="clear_air_turbulence_425hpa",
        element="CAT",
        level="425 mb",
        attrs=DataVarAttrs(
            short_name="cat",
            long_name="Clear air turbulence",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("CAT", "450 mb"): ProductField(
        name="clear_air_turbulence_450hpa",
        element="CAT",
        level="450 mb",
        attrs=DataVarAttrs(
            short_name="cat",
            long_name="Clear air turbulence",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("CAT", "525 mb"): ProductField(
        name="clear_air_turbulence_525hpa",
        element="CAT",
        level="525 mb",
        attrs=DataVarAttrs(
            short_name="cat",
            long_name="Clear air turbulence",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("CFRZR", "surface"): ProductField(
        name="freezing_rain_fraction_surface",
        element="CFRZR",
        level="surface",
        attrs=DataVarAttrs(
            short_name="cfrzr_fraction",
            long_name="Freezing rain occurrence fraction",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("CICEP", "surface"): ProductField(
        name="ice_pellets_fraction_surface",
        element="CICEP",
        level="surface",
        attrs=DataVarAttrs(
            short_name="cicep_fraction",
            long_name="Ice pellets occurrence fraction",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("CIN", "180-0 mb above ground"): _rrfs_field(
        "convective_inhibition_180_0mb",
        name="convective_inhibition_180_0mb",
        level="180-0 mb above ground",
    ),
    ("CIN", "90-0 mb above ground"): _rrfs_field(
        "convective_inhibition_90_0mb",
        name="convective_inhibition_90_0mb",
        level="90-0 mb above ground",
    ),
    ("CIN", "surface"): _rrfs_field(
        "convective_inhibition_surface",
        name="convective_inhibition_surface",
        level="surface",
    ),
    ("CRAIN", "surface"): ProductField(
        name="rain_fraction_surface",
        element="CRAIN",
        level="surface",
        attrs=DataVarAttrs(
            short_name="crain_fraction",
            long_name="Rain occurrence fraction",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("CSNOW", "surface"): ProductField(
        name="snow_fraction_surface",
        element="CSNOW",
        level="surface",
        attrs=DataVarAttrs(
            short_name="csnow_fraction",
            long_name="Snow occurrence fraction",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("DPT", "2 m above ground"): _rrfs_field(
        "dew_point_temperature_2m",
        name="dew_point_temperature_2m",
        level="2 m above ground",
    ),
    ("DPT", "500 mb"): _rrfs_field(
        "pressure_level/dew_point_temperature",
        name="dew_point_temperature_500hpa",
        level="500 mb",
    ),
    ("DPT", "700 mb"): _rrfs_field(
        "pressure_level/dew_point_temperature",
        name="dew_point_temperature_700hpa",
        level="700 mb",
    ),
    ("DPT", "850 mb"): _rrfs_field(
        "pressure_level/dew_point_temperature",
        name="dew_point_temperature_850hpa",
        level="850 mb",
    ),
    ("DPT", "925 mb"): _rrfs_field(
        "pressure_level/dew_point_temperature",
        name="dew_point_temperature_925hpa",
        level="925 mb",
    ),
    ("DZDT", "700 mb"): _rrfs_field(
        "pressure_level/vertical_velocity_geometric",
        name="vertical_velocity_geometric_700hpa",
        level="700 mb",
    ),
    ("DZDT", "700-500 mb above ground"): _rrfs_field(
        "pressure_level/vertical_velocity_geometric",
        name="vertical_velocity_geometric_700_500mb",
        level="700-500 mb above ground",
    ),
    ("FLGHT", "surface"): ProductField(
        name="flight_category_surface",
        element="FLGHT",
        level="surface",
        attrs=DataVarAttrs(
            short_name="flight_category",
            long_name="Flight category",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("FRZR", "surface"): ProductField(
        name="freezing_rain_surface",
        element="FRZR",
        level="surface",
        attrs=DataVarAttrs(
            short_name="fzra",
            long_name="Freezing rain",
            units="kg m-2",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("HCDC", "high cloud layer"): _rrfs_field(
        "high_cloud_cover", name="high_cloud_cover", level="high cloud layer"
    ),
    ("HGT", "0C isotherm"): _rrfs_field(
        "geopotential_height_0c_isotherm",
        name="geopotential_height_0c_isotherm",
        level="0C isotherm",
    ),
    ("HGT", "250 mb"): _rrfs_field(
        "pressure_level/geopotential_height",
        name="geopotential_height_250hpa",
        level="250 mb",
    ),
    ("HGT", "500 mb"): _rrfs_field(
        "pressure_level/geopotential_height",
        name="geopotential_height_500hpa",
        level="500 mb",
    ),
    ("HGT", "700 mb"): _rrfs_field(
        "pressure_level/geopotential_height",
        name="geopotential_height_700hpa",
        level="700 mb",
    ),
    ("HGT", "850 mb"): _rrfs_field(
        "pressure_level/geopotential_height",
        name="geopotential_height_850hpa",
        level="850 mb",
    ),
    ("HGT", "925 mb"): _rrfs_field(
        "pressure_level/geopotential_height",
        name="geopotential_height_925hpa",
        level="925 mb",
    ),
    ("HGT", "cloud base"): _rrfs_field(
        "geopotential_height_cloud_base",
        name="geopotential_height_cloud_base",
        level="cloud base",
    ),
    ("HGT", "cloud ceiling"): _rrfs_field(
        "geopotential_height_cloud_ceiling",
        name="geopotential_height_cloud_ceiling",
        level="cloud ceiling",
    ),
    ("HGT", "surface"): _rrfs_field(
        "geopotential_height_surface",
        name="geopotential_height_surface",
        level="surface",
    ),
    ("HLCY", "3000-0 m above ground"): _rrfs_field(
        "storm_relative_helicity_3000_0m",
        name="storm_relative_helicity_3000_0m",
        level="3000-0 m above ground",
    ),
    ("ICI", "1000 mb"): ProductField(
        name="icing_1000hpa",
        element="ICI",
        level="1000 mb",
        attrs=DataVarAttrs(
            short_name="ici",
            long_name="Icing",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("ICI", "400 mb"): ProductField(
        name="icing_400hpa",
        element="ICI",
        level="400 mb",
        attrs=DataVarAttrs(
            short_name="ici",
            long_name="Icing",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("ICI", "500 mb"): ProductField(
        name="icing_500hpa",
        element="ICI",
        level="500 mb",
        attrs=DataVarAttrs(
            short_name="ici",
            long_name="Icing",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("ICI", "575 mb"): ProductField(
        name="icing_575hpa",
        element="ICI",
        level="575 mb",
        attrs=DataVarAttrs(
            short_name="ici",
            long_name="Icing",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("ICI", "650 mb"): ProductField(
        name="icing_650hpa",
        element="ICI",
        level="650 mb",
        attrs=DataVarAttrs(
            short_name="ici",
            long_name="Icing",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("ICI", "725 mb"): ProductField(
        name="icing_725hpa",
        element="ICI",
        level="725 mb",
        attrs=DataVarAttrs(
            short_name="ici",
            long_name="Icing",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("ICI", "800 mb"): ProductField(
        name="icing_800hpa",
        element="ICI",
        level="800 mb",
        attrs=DataVarAttrs(
            short_name="ici",
            long_name="Icing",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("ICI", "900 mb"): ProductField(
        name="icing_900hpa",
        element="ICI",
        level="900 mb",
        attrs=DataVarAttrs(
            short_name="ici",
            long_name="Icing",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("JFWPRB", "10 m above ground"): ProductField(
        name="joint_fire_weather_probability",
        element="JFWPRB",
        level="10 m above ground",
        attrs=DataVarAttrs(
            short_name="jfwprb",
            long_name="Joint Fire Weather Probability",
            units="percent",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("LCDC", "low cloud layer"): _rrfs_field(
        "low_cloud_cover", name="low_cloud_cover", level="low cloud layer"
    ),
    ("LTNG", "entire atmosphere"): ProductField(
        name="lightning_atmosphere",
        element="LTNG",
        level="entire atmosphere",
        attrs=DataVarAttrs(
            short_name="ltng",
            long_name="Lightning",
            units="1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("MAXREF", "1000 m above ground"): _rrfs_field(
        "hourly_maximum_radar_reflectivity_1000m",
        name="hourly_maximum_radar_reflectivity_1000m",
        level="1000 m above ground",
    ),
    ("MAXUVV", "100-1000 mb"): _rrfs_field(
        "maximum_upward_vertical_velocity_100_1000mb",
        name="maximum_upward_vertical_velocity_100_1000mb",
        level="100-1000 mb",
    ),
    ("MCDC", "middle cloud layer"): _rrfs_field(
        "medium_cloud_cover", name="medium_cloud_cover", level="middle cloud layer"
    ),
    ("MSLET", "mean sea level"): _rrfs_field(
        "pressure_reduced_to_mean_sea_level",
        name="pressure_reduced_to_mean_sea_level",
        level="mean sea level",
    ),
    ("MXUPHL", "5000-2000 m above ground"): _rrfs_field(
        "maximum_updraft_helicity_5000_2000m",
        name="maximum_updraft_helicity_5000_2000m",
        level="5000-2000 m above ground",
    ),
    ("PPFFG", "surface"): ProductField(
        name="precipitation_exceeding_flash_flood_guidance_surface",
        element="PPFFG",
        level="surface",
        attrs=DataVarAttrs(
            short_name="ppffg",
            long_name="Precipitation exceeding flash flood guidance",
            units="percent",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("PWAT", "entire atmosphere (considered as a single layer)"): _rrfs_field(
        "precipitable_water_atmosphere",
        name="precipitable_water_atmosphere",
        level="entire atmosphere (considered as a single layer)",
    ),
    ("REFC", "entire atmosphere (considered as a single layer)"): _rrfs_field(
        "composite_reflectivity",
        name="composite_reflectivity",
        level="entire atmosphere (considered as a single layer)",
    ),
    ("REFD", "1000 m above ground"): _rrfs_field(
        "derived_radar_reflectivity_1000m",
        name="derived_radar_reflectivity_1000m",
        level="1000 m above ground",
    ),
    ("RETOP", "entire atmosphere (considered as a single layer)"): _rrfs_field(
        "echo_top",
        name="echo_top",
        level="entire atmosphere (considered as a single layer)",
    ),
    ("RH", "700 mb"): _rrfs_field(
        "pressure_level/relative_humidity",
        name="relative_humidity_700hpa",
        level="700 mb",
    ),
    ("SOILW", "0-0 m below ground"): _rrfs_field(
        "depth_below_ground/volumetric_soil_moisture",
        name="volumetric_soil_moisture_0m",
        level="0-0 m below ground",
    ),
    ("TCDC", "entire atmosphere (considered as a single layer)"): _rrfs_field(
        "instantaneous_total_cloud_cover_atmosphere",
        name="total_cloud_cover_atmosphere",
        level="entire atmosphere (considered as a single layer)",
    ),
    ("TMP", "2 m above ground"): _rrfs_field(
        "temperature_2m", name="temperature_2m", level="2 m above ground"
    ),
    ("TMP", "250 mb"): _rrfs_field(
        "pressure_level/temperature", name="temperature_250hpa", level="250 mb"
    ),
    ("TMP", "500 mb"): _rrfs_field(
        "pressure_level/temperature", name="temperature_500hpa", level="500 mb"
    ),
    ("TMP", "700 mb"): _rrfs_field(
        "pressure_level/temperature", name="temperature_700hpa", level="700 mb"
    ),
    ("TMP", "850 mb"): _rrfs_field(
        "pressure_level/temperature", name="temperature_850hpa", level="850 mb"
    ),
    ("TMP", "925 mb"): _rrfs_field(
        "pressure_level/temperature", name="temperature_925hpa", level="925 mb"
    ),
    ("TMP", "surface"): _rrfs_field(
        "temperature_surface", name="temperature_surface", level="surface"
    ),
    ("TSOIL", "0-0 m below ground"): _rrfs_field(
        "depth_below_ground/soil_temperature",
        name="soil_temperature_0m",
        level="0-0 m below ground",
    ),
    ("UGRD", "10 m above ground"): _rrfs_field(
        "wind_u_10m", name="wind_u_10m", level="10 m above ground"
    ),
    ("UGRD", "250 mb"): _rrfs_field(
        "pressure_level/wind_u", name="wind_u_250hpa", level="250 mb"
    ),
    ("UGRD", "500 mb"): _rrfs_field(
        "pressure_level/wind_u", name="wind_u_500hpa", level="500 mb"
    ),
    ("UGRD", "550 mb"): _rrfs_field(
        "pressure_level/wind_u", name="wind_u_550hpa", level="550 mb"
    ),
    ("UGRD", "700 mb"): _rrfs_field(
        "pressure_level/wind_u", name="wind_u_700hpa", level="700 mb"
    ),
    ("UGRD", "850 mb"): _rrfs_field(
        "pressure_level/wind_u", name="wind_u_850hpa", level="850 mb"
    ),
    ("UGRD", "925 mb"): _rrfs_field(
        "pressure_level/wind_u", name="wind_u_925hpa", level="925 mb"
    ),
    ("VGRD", "10 m above ground"): _rrfs_field(
        "wind_v_10m", name="wind_v_10m", level="10 m above ground"
    ),
    ("VGRD", "250 mb"): _rrfs_field(
        "pressure_level/wind_v", name="wind_v_250hpa", level="250 mb"
    ),
    ("VGRD", "500 mb"): _rrfs_field(
        "pressure_level/wind_v", name="wind_v_500hpa", level="500 mb"
    ),
    ("VGRD", "550 mb"): _rrfs_field(
        "pressure_level/wind_v", name="wind_v_550hpa", level="550 mb"
    ),
    ("VGRD", "700 mb"): _rrfs_field(
        "pressure_level/wind_v", name="wind_v_700hpa", level="700 mb"
    ),
    ("VGRD", "850 mb"): _rrfs_field(
        "pressure_level/wind_v", name="wind_v_850hpa", level="850 mb"
    ),
    ("VGRD", "925 mb"): _rrfs_field(
        "pressure_level/wind_v", name="wind_v_925hpa", level="925 mb"
    ),
    ("VIS", "surface"): _rrfs_field(
        "visibility_surface", name="visibility_surface", level="surface"
    ),
    ("VWSH", "0-6000 m above ground"): ProductField(
        name="vertical_wind_shear_0_6000m",
        element="VWSH",
        level="0-6000 m above ground",
        attrs=DataVarAttrs(
            short_name="vws",
            long_name="Vertical wind shear",
            units="m s-1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("VWSH", "0-609 m above ground"): ProductField(
        name="vertical_wind_shear_0_609m",
        element="VWSH",
        level="0-609 m above ground",
        attrs=DataVarAttrs(
            short_name="vws",
            long_name="Vertical wind shear",
            units="m s-1",
            standard_name=None,
            step_type="instant",
        ),
        filters=(),
    ),
    ("WEASD", "surface"): ProductField(
        name="snowfall_water_equivalent_surface",
        element="WEASD",
        level="surface",
        attrs=DataVarAttrs(
            short_name="sf",
            long_name="Snowfall water equivalent",
            units="kg m-2",
            standard_name="snowfall_amount",
            step_type="instant",
        ),
        filters=(),
    ),
    ("WIND", "10 m above ground"): ProductField(
        name="wind_speed_10m",
        element="WIND",
        level="10 m above ground",
        attrs=DataVarAttrs(
            short_name="10si",
            long_name="10 metre wind speed",
            units="m s-1",
            standard_name="wind_speed",
            step_type="instant",
        ),
    ),
    ("WIND", "250 mb"): ProductField(
        name="wind_speed_250hpa",
        element="WIND",
        level="250 mb",
        attrs=DataVarAttrs(
            short_name="ws",
            long_name="Wind speed",
            units="m s-1",
            standard_name="wind_speed",
            step_type="instant",
        ),
    ),
    ("WIND", "550 mb"): ProductField(
        name="wind_speed_550hpa",
        element="WIND",
        level="550 mb",
        attrs=DataVarAttrs(
            short_name="ws",
            long_name="Wind speed",
            units="m s-1",
            standard_name="wind_speed",
            step_type="instant",
        ),
    ),
    ("WIND", "80 m above ground"): ProductField(
        name="wind_speed_80m",
        element="WIND",
        level="80 m above ground",
        attrs=DataVarAttrs(
            short_name="80si",
            long_name="80 metre wind speed",
            units="m s-1",
            standard_name="wind_speed",
            step_type="instant",
        ),
    ),
    ("WIND", "850 mb"): ProductField(
        name="wind_speed_850hpa",
        element="WIND",
        level="850 mb",
        attrs=DataVarAttrs(
            short_name="ws",
            long_name="Wind speed",
            units="m s-1",
            standard_name="wind_speed",
            step_type="instant",
        ),
    ),
    ("WIND", "850-300 mb"): ProductField(
        name="wind_speed_850_300mb",
        element="WIND",
        level="850-300 mb",
        attrs=DataVarAttrs(
            short_name="ws",
            long_name="Wind speed",
            units="m s-1",
            standard_name="wind_speed",
            step_type="instant",
        ),
        filters=(),
    ),
    ("WIND", "925 mb"): ProductField(
        name="wind_speed_925hpa",
        element="WIND",
        level="925 mb",
        attrs=DataVarAttrs(
            short_name="ws",
            long_name="Wind speed",
            units="m s-1",
            standard_name="wind_speed",
            step_type="instant",
        ),
    ),
    (
        "var discipline=0 center=7 local_table=1 parmcat=6 parm=202",
        "0 m above ground",
    ): ProductField(
        name="fog_liquid_water_content_0m",
        element="var discipline=0 center=7 local_table=1 parmcat=6 parm=202",
        level="0 m above ground",
        attrs=DataVarAttrs(
            short_name="fog_lwc",
            long_name="Fog liquid water content",
            units="kg kg-1",
            standard_name=None,
            step_type="instant",
        ),
        filters=({"name": "scale_offset", "configuration": {"scale": 1000.0}},),
    ),
}

_FIELD_COMMENTS = {
    (
        "VIS",
        "surface",
    ): "Mean and spread exclude members capped at the no-event visibility limit. A large mean can represent the cap when no uncapped member remains; the corresponding zero spread does not imply agreement.",
    (
        "HGT",
        "cloud ceiling",
    ): "Mean and spread exclude capped members. A mean of 20000 m denotes no reported cloud ceiling; zero spread can indicate no uncapped members as well as agreement.",
    (
        "HGT",
        "cloud base",
    ): "Mean and spread use only members with a cloud base. When none contribute, the mean is the source's no-cloud height of about 20000 m and the spread is zero, which does not indicate measured certainty. In some source messages, mean heights beyond the packing range of about 8192 m above the minimum are reduced by multiples of 8192 m. The no-cloud height then appears near 3616 m; wrapped values cannot be distinguished from true heights.",
    (
        "TSOIL",
        "0-0 m below ground",
    ): "Over water, absent member soil values enter the statistics as raw zero before Celsius conversion, while some members may contribute nonzero values. The Celsius mean can approach -273.15 degree_Celsius. Means and spreads there are not soil temperature quantities. These cells are not marked; use a land mask.",
    (
        "SOILW",
        "0-0 m below ground",
    ): "Over water, absent member soil values enter the statistics as zero, while some members may contribute nonzero values. Means and spreads there are not soil moisture quantities. These cells are not marked; use a land mask.",
    (
        "DZDT",
        "700-500 mb above ground",
    ): "Vertical velocity in a layer defined by pressure differences above ground, rather than an isobaric layer.",
    (
        "var discipline=0 center=7 local_table=1 parmcat=6 parm=202",
        "0 m above ground",
    ): "Fog liquid water is derived by the ensemble generator. Mean is weighted over members with positive fog liquid water. Spread divides the weighted sum of squared deviations from that mean over all nonmissing members, including zero-fog members, by the positive-member weight sum before taking the square root. Both are zero when no member has fog.",
}
_FIELDS = {
    key: replace(
        field,
        attrs=replace(
            field.attrs, comment=_FIELD_COMMENTS.get(key, field.attrs.comment)
        ),
    )
    for key, field in _FIELDS.items()
}
_FIELDS = {
    key: replace(
        field,
        attrs=replace(
            field.attrs,
            step_type="max",
            comment="Maximum over the hour ending at the valid time.",
        ),
    )
    if key[0] in ("MXUPHL", "MAXREF", "MAXUVV")
    else field
    for key, field in _FIELDS.items()
}

_STATISTICS: tuple[tuple[str, str, int, tuple[str, ...]], ...] = (
    ("APCP", "surface", 1, ()),
    ("APCP", "surface", 3, ()),
    ("APCP", "surface", 6, ()),
    ("ASNOW", "surface", 1, ()),
    ("ASNOW", "surface", 3, ()),
    ("ASNOW", "surface", 6, ()),
    ("ASNOW", "surface", 12, ()),
    ("ASNOW", "surface", 24, ()),
    ("CAPE", "180-0 mb above ground", 0, ()),
    ("CAPE", "90-0 mb above ground", 0, ()),
    ("CAPE", "surface", 0, ()),
    ("CFRZR", "surface", 0, ()),
    ("CICEP", "surface", 0, ()),
    ("CIN", "180-0 mb above ground", 0, ()),
    ("CIN", "90-0 mb above ground", 0, ()),
    ("CIN", "surface", 0, ()),
    ("CRAIN", "surface", 0, ()),
    ("CSNOW", "surface", 0, ()),
    ("DPT", "2 m above ground", 0, ()),
    ("DPT", "500 mb", 0, ()),
    ("DPT", "700 mb", 0, ()),
    ("DPT", "850 mb", 0, ()),
    ("DPT", "925 mb", 0, ()),
    ("DZDT", "700 mb", 0, ()),
    ("DZDT", "700-500 mb above ground", 0, ()),
    ("HCDC", "high cloud layer", 0, ()),
    ("HGT", "0C isotherm", 0, ()),
    ("HGT", "250 mb", 0, ()),
    ("HGT", "500 mb", 0, ()),
    ("HGT", "700 mb", 0, ()),
    ("HGT", "850 mb", 0, ()),
    ("HGT", "925 mb", 0, ()),
    ("HGT", "cloud base", 0, ()),
    ("HGT", "cloud ceiling", 0, ()),
    ("HGT", "surface", 0, ()),
    ("HLCY", "3000-0 m above ground", 0, ()),
    ("LCDC", "low cloud layer", 0, ()),
    ("MAXREF", "1000 m above ground", 0, ("process=193",)),
    ("MCDC", "middle cloud layer", 0, ()),
    ("MSLET", "mean sea level", 0, ()),
    ("MXUPHL", "5000-2000 m above ground", 0, ("process=193",)),
    ("PWAT", "entire atmosphere (considered as a single layer)", 0, ()),
    ("REFC", "entire atmosphere (considered as a single layer)", 0, ("process=193",)),
    ("REFD", "1000 m above ground", 0, ("process=193",)),
    ("RETOP", "entire atmosphere (considered as a single layer)", 0, ("process=193",)),
    ("RH", "700 mb", 0, ()),
    ("SOILW", "0-0 m below ground", 0, ()),
    ("TCDC", "entire atmosphere (considered as a single layer)", 0, ()),
    ("TMP", "2 m above ground", 0, ()),
    ("TMP", "250 mb", 0, ()),
    ("TMP", "500 mb", 0, ()),
    ("TMP", "700 mb", 0, ()),
    ("TMP", "850 mb", 0, ()),
    ("TMP", "925 mb", 0, ()),
    ("TMP", "surface", 0, ()),
    ("TSOIL", "0-0 m below ground", 0, ()),
    ("UGRD", "10 m above ground", 0, ()),
    ("UGRD", "250 mb", 0, ()),
    ("UGRD", "500 mb", 0, ()),
    ("UGRD", "550 mb", 0, ()),
    ("UGRD", "700 mb", 0, ()),
    ("UGRD", "850 mb", 0, ()),
    ("UGRD", "925 mb", 0, ()),
    ("VGRD", "10 m above ground", 0, ()),
    ("VGRD", "250 mb", 0, ()),
    ("VGRD", "500 mb", 0, ()),
    ("VGRD", "550 mb", 0, ()),
    ("VGRD", "700 mb", 0, ()),
    ("VGRD", "850 mb", 0, ()),
    ("VGRD", "925 mb", 0, ()),
    ("VIS", "surface", 0, ()),
    ("VWSH", "0-6000 m above ground", 0, ()),
    ("VWSH", "0-609 m above ground", 0, ()),
    ("WEASD", "surface", 1, ()),
    ("WEASD", "surface", 3, ()),
    ("WIND", "10 m above ground", 0, ()),
    ("WIND", "250 mb", 0, ()),
    ("WIND", "550 mb", 0, ()),
    ("WIND", "80 m above ground", 0, ()),
    ("WIND", "850 mb", 0, ()),
    ("WIND", "925 mb", 0, ()),
    (
        "var discipline=0 center=7 local_table=1 parmcat=6 parm=202",
        "0 m above ground",
        0,
        (),
    ),
)
_SPARSE: tuple[tuple[ProductFamily, str, str, int, tuple[str, ...]], ...] = (
    ("avrg", "APCP", "surface", 1, ("wt ens mean", "process=193")),
    ("avrg", "APCP", "surface", 3, ("wt ens mean", "process=193")),
    ("lpmm", "APCP", "surface", 1, ("wt ens mean", "process=200")),
    ("lpmm", "APCP", "surface", 3, ("wt ens mean", "process=200")),
    ("lpmm", "APCP", "surface", 6, ("wt ens mean", "process=200")),
    ("pmmn", "APCP", "surface", 1, ("wt ens mean", "process=193")),
    ("pmmn", "APCP", "surface", 3, ("wt ens mean", "process=193")),
    ("pmmn", "HGT", "surface", 0, ("wt ens mean", "process=193")),
    ("pmmn", "MAXREF", "1000 m above ground", 0, ("wt ens mean", "process=193")),
    ("pmmn", "MXUPHL", "5000-2000 m above ground", 0, ("wt ens mean", "process=193")),
    (
        "pmmn",
        "REFC",
        "entire atmosphere (considered as a single layer)",
        0,
        ("wt ens mean", "process=193"),
    ),
    ("pmmn", "REFD", "1000 m above ground", 0, ("wt ens mean", "process=193")),
    (
        "pmmn",
        "RETOP",
        "entire atmosphere (considered as a single layer)",
        0,
        ("wt ens mean", "process=193"),
    ),
)
_PROBABILITIES: tuple[
    tuple[ProductFamily, str, str, int, tuple[str, ...], tuple[str, ...]], ...
] = (
    (
        "eas",
        "APCP",
        "surface",
        1,
        ("prob >0.254", "prob >12.7", "prob >25.4", "prob >6.35"),
        ("process=197",),
    ),
    (
        "eas",
        "APCP",
        "surface",
        3,
        ("prob >0.254", "prob >12.7", "prob >25.4", "prob >50.8", "prob >6.35"),
        ("process=197",),
    ),
    (
        "eas",
        "APCP",
        "surface",
        6,
        (
            "prob >0.254",
            "prob >12.7",
            "prob >25.4",
            "prob >50.8",
            "prob >6.35",
            "prob >76.2",
        ),
        ("process=197",),
    ),
    (
        "eas",
        "APCP",
        "surface",
        12,
        (
            "prob >12.7",
            "prob >127",
            "prob >2.54",
            "prob >25.4",
            "prob >50.8",
            "prob >6.35",
            "prob >76.2",
        ),
        ("process=197",),
    ),
    (
        "eas",
        "APCP",
        "surface",
        24,
        (
            "prob >12.7",
            "prob >127",
            "prob >2.54",
            "prob >25.4",
            "prob >50.8",
            "prob >6.35",
            "prob >76.2",
        ),
        ("process=197",),
    ),
    ("eas", "ASNOW", "surface", 1, ("prob >0.025", "prob >0.076"), ("process=197",)),
    ("eas", "ASNOW", "surface", 3, ("prob >0.025", "prob >0.076"), ("process=197",)),
    (
        "eas",
        "ASNOW",
        "surface",
        6,
        ("prob >0.025", "prob >0.076", "prob >0.152"),
        ("process=197",),
    ),
    (
        "ffri",
        "APCP",
        "surface",
        6,
        (
            "prob >1",
            "prob >10",
            "prob >100",
            "prob >2",
            "prob >25",
            "prob >5",
            "prob >50",
        ),
        ("Neighborhood Probability",),
    ),
    (
        "ffri",
        "APCP",
        "surface",
        24,
        (
            "prob >1",
            "prob >10",
            "prob >100",
            "prob >2",
            "prob >25",
            "prob >5",
            "prob >50",
        ),
        ("Neighborhood Probability",),
    ),
    ("ffri", "PPFFG", "surface", 1, ("prob >1",), ("Neighborhood Probability",)),
    ("ffri", "PPFFG", "surface", 3, ("prob >3",), ("Neighborhood Probability",)),
    ("ffri", "PPFFG", "surface", 6, ("prob >6",), ("Neighborhood Probability",)),
    (
        "prob",
        "APCP",
        "surface",
        1,
        ("prob >12.7", "prob >25.4", "prob >50.8", "prob >76.2"),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "APCP",
        "surface",
        3,
        ("prob >12.7", "prob >127", "prob >25.4", "prob >50.8", "prob >76.2"),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "APCP",
        "surface",
        6,
        ("prob >12.7", "prob >127", "prob >25.4", "prob >50.8", "prob >76.2"),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "APCP",
        "surface",
        12,
        (
            "prob >12.7",
            "prob >127",
            "prob >203.2",
            "prob >25.4",
            "prob >50.8",
            "prob >76.2",
        ),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "APCP",
        "surface",
        24,
        (
            "prob >12.7",
            "prob >127",
            "prob >203.2",
            "prob >25.4",
            "prob >50.8",
            "prob >76.2",
        ),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "ASNOW",
        "surface",
        1,
        ("prob >0.025", "prob >0.076"),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "ASNOW",
        "surface",
        3,
        ("prob >0.025", "prob >0.076", "prob >0.152"),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "ASNOW",
        "surface",
        6,
        ("prob >0.025", "prob >0.076", "prob >0.152", "prob >0.304"),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "ASNOW",
        "surface",
        12,
        ("prob >0.025", "prob >0.076", "prob >0.152", "prob >0.304", "prob >0.457"),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "ASNOW",
        "surface",
        24,
        ("prob >0.025", "prob >0.076", "prob >0.152", "prob >0.304", "prob >0.457"),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "CAPE",
        "90-0 mb above ground",
        0,
        ("prob >1000", "prob >1500", "prob >2000", "prob >3000", "prob >500"),
        (),
    ),
    ("prob", "CAT", "175 mb", 0, ("prob >=1 <2", "prob >=2 <3", "prob >=3 <0"), ()),
    ("prob", "CAT", "200 mb", 0, ("prob >=1 <2", "prob >=2 <3", "prob >=3 <0"), ()),
    ("prob", "CAT", "225 mb", 0, ("prob >=1 <2", "prob >=2 <3", "prob >=3 <0"), ()),
    ("prob", "CAT", "275 mb", 0, ("prob >=1 <2", "prob >=2 <3", "prob >=3 <0"), ()),
    ("prob", "CAT", "300 mb", 0, ("prob >=1 <2", "prob >=2 <3", "prob >=3 <0"), ()),
    ("prob", "CAT", "350 mb", 0, ("prob >=1 <2", "prob >=2 <3", "prob >=3 <0"), ()),
    ("prob", "CAT", "425 mb", 0, ("prob >=1 <2", "prob >=2 <3", "prob >=3 <0"), ()),
    ("prob", "CAT", "450 mb", 0, ("prob >=1 <2", "prob >=2 <3", "prob >=3 <0"), ()),
    ("prob", "CAT", "525 mb", 0, ("prob >=1 <2", "prob >=2 <3", "prob >=3 <0"), ()),
    ("prob", "CFRZR", "surface", 0, ("prob >=1 <0",), ()),
    ("prob", "CICEP", "surface", 0, ("prob >=1 <0",), ()),
    (
        "prob",
        "CIN",
        "90-0 mb above ground",
        0,
        ("prob <-100", "prob <-400", "prob <-50", "prob <0"),
        (),
    ),
    ("prob", "CRAIN", "surface", 0, ("prob >=1 <0",), ()),
    ("prob", "CSNOW", "surface", 0, ("prob >=1 <0",), ()),
    (
        "prob",
        "DPT",
        "2 m above ground",
        0,
        (
            "prob >283.15",
            "prob >285.93",
            "prob >288.71",
            "prob >291.48",
            "prob >294.26",
        ),
        (),
    ),
    (
        "prob",
        "FLGHT",
        "surface",
        0,
        ("prob >=1 <2", "prob >=2 <3", "prob >=3 <4", "prob >=4 <0"),
        (),
    ),
    (
        "prob",
        "FRZR",
        "surface",
        1,
        ("prob >0.254", "prob >2.54", "prob >6.35"),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "FRZR",
        "surface",
        3,
        ("prob >0.254", "prob >2.54", "prob >6.35"),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "FRZR",
        "surface",
        6,
        ("prob >0.254", "prob >12.7", "prob >2.54", "prob >6.35"),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "FRZR",
        "surface",
        12,
        ("prob >0.254", "prob >12.7", "prob >2.54", "prob >6.35"),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "FRZR",
        "surface",
        24,
        ("prob >0.254", "prob >12.7", "prob >2.54", "prob >25.4", "prob >6.35"),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "HGT",
        "cloud ceiling",
        0,
        (
            "prob <1372",
            "prob <152.4",
            "prob <1830",
            "prob <305",
            "prob <3050",
            "prob <610",
            "prob <915",
        ),
        (),
    ),
    (
        "prob",
        "HLCY",
        "3000-0 m above ground",
        0,
        ("prob >100", "prob >200", "prob >400"),
        (),
    ),
    ("prob", "ICI", "1000 mb", 0, ("prob >=1 <0",), ()),
    ("prob", "ICI", "400 mb", 0, ("prob >=1 <0",), ()),
    ("prob", "ICI", "500 mb", 0, ("prob >=1 <0",), ()),
    ("prob", "ICI", "575 mb", 0, ("prob >=1 <0",), ()),
    ("prob", "ICI", "650 mb", 0, ("prob >=1 <0",), ()),
    ("prob", "ICI", "725 mb", 0, ("prob >=1 <0",), ()),
    ("prob", "ICI", "800 mb", 0, ("prob >=1 <0",), ()),
    ("prob", "ICI", "900 mb", 0, ("prob >=1 <0",), ()),
    ("prob", "JFWPRB", "10 m above ground", 0, ("prob >=9 <20",), ()),
    ("prob", "LTNG", "entire atmosphere", 0, ("prob >0.08",), ()),
    (
        "prob",
        "MAXREF",
        "1000 m above ground",
        0,
        ("prob >40", "prob >50"),
        ("Neighborhood Probability",),
    ),
    ("prob", "MAXUVV", "100-1000 mb", 0, ("prob >1", "prob >10", "prob >20"), ()),
    (
        "prob",
        "MXUPHL",
        "5000-2000 m above ground",
        0,
        ("prob >150", "prob >25", "prob >75"),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "PWAT",
        "entire atmosphere (considered as a single layer)",
        0,
        ("prob >25", "prob >37.5", "prob >50"),
        (),
    ),
    (
        "prob",
        "REFC",
        "entire atmosphere (considered as a single layer)",
        0,
        ("prob >10", "prob >20", "prob >30", "prob >40", "prob >50"),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "REFD",
        "1000 m above ground",
        0,
        ("prob >30", "prob >40", "prob >50"),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "RETOP",
        "entire atmosphere (considered as a single layer)",
        0,
        ("prob >10668", "prob >12192", "prob >15240", "prob >6096", "prob >9144"),
        ("Neighborhood Probability",),
    ),
    ("prob", "TMP", "2 m above ground", 0, ("prob <273.15",), ()),
    (
        "prob",
        "VIS",
        "surface",
        0,
        (
            "prob <1600",
            "prob <3200",
            "prob <400",
            "prob <4829",
            "prob <6400",
            "prob <800",
            "prob <8049",
        ),
        (),
    ),
    (
        "prob",
        "VWSH",
        "0-6000 m above ground",
        0,
        ("prob >10.3", "prob >15.4", "prob >20.6", "prob >25.7"),
        (),
    ),
    ("prob", "VWSH", "0-609 m above ground", 0, ("prob >10.3", "prob >20"), ()),
    (
        "prob",
        "WIND",
        "10 m above ground",
        0,
        (
            "prob >10.3",
            "prob >15.4",
            "prob >18.01",
            "prob >20.6",
            "prob >25.72",
            "prob >30.9",
        ),
        ("Neighborhood Probability",),
    ),
    (
        "prob",
        "WIND",
        "250 mb",
        0,
        ("prob >10.3", "prob >20.6", "prob >30.9", "prob >41.2", "prob >51.5"),
        (),
    ),
    (
        "prob",
        "WIND",
        "550 mb",
        0,
        ("prob >10.3", "prob >20.6", "prob >30.9", "prob >41.2", "prob >51.5"),
        (),
    ),
    (
        "prob",
        "WIND",
        "80 m above ground",
        0,
        ("prob >10.3", "prob >15.4", "prob >20.6"),
        (),
    ),
    (
        "prob",
        "WIND",
        "850 mb",
        0,
        ("prob >10.3", "prob >20.6", "prob >30.9", "prob >41.2", "prob >51.5"),
        (),
    ),
    ("prob", "WIND", "850-300 mb", 0, ("prob <5",), ()),
    (
        "prob",
        "var discipline=0 center=7 local_table=1 parmcat=6 parm=202",
        "0 m above ground",
        0,
        ("prob >0.016", "prob >0.036", "prob >0.103"),
        (),
    ),
)

_FFRI_COMMENT = (
    "Probabilities compare precipitation with a spatially varying guidance or recurrence-interval depth. "
    "Negative values are unavailable thresholds blended into neighboring cells by smoothing; "
    "positive values near coverage boundaries can also be biased low. Mask values < -0.1."
)
_WIND_COMMENT = (
    "Mean is the weighted mean of member speeds. Spread is sqrt(var(u) + var(v)), "
    "the vector spread including variation in direction, rather than the standard deviation of speed."
)
_RADAR_SPREAD_COMMENT = (
    "Spread includes members' no-echo markers. Differences in echo presence or in the markers "
    "contribute to spread, so it does not measure only uncertainty in echo magnitude or height."
)
_SPARSE_RADAR_COMMENTS = {
    "REFC": "The probability-matched mean retains member no-echo floors at -20, -10 or 0 dBZ. A value of 0 dBZ can also be a valid reflectivity; other negative dBZ values can be valid.",
    "REFD": "The probability-matched mean retains member no-echo floors at -20, -10 or 0 dBZ. A value of 0 dBZ can also be a valid reflectivity; other negative dBZ values can be valid.",
    "MAXREF": "A value of 0 dBZ can mean no echo or a valid reflectivity at the source's lowest reported value.",
    "RETOP": "Negative values indicate no echo. Mask values < 0.",
}
_PROBABILITY_THRESHOLD_UNITS = {
    "APCP": "kg m-2",
    "FRZR": "kg m-2",
    "TMP": "K",
    "DPT": "K",
    "var discipline=0 center=7 local_table=1 parmcat=6 parm=202": "g kg-1",
}
_OCCURRENCE_CATEGORIES = {
    "CFRZR": "freezing rain",
    "CICEP": "ice pellets",
    "CRAIN": "rain",
    "CSNOW": "snow",
}
_PROBABILITY_LEVEL_LABELS = {
    "entire atmosphere (considered as a single layer)": "entire atmosphere",
}
_PROBABILITY_CATEGORY_COMMENTS = {
    "CAT": "The generator defines categories from its Ellrod-based turbulence index: 0 at most 4; 1 above 4 through 8; 2 above 8 through 12; 3 above 12. Higher categories indicate stronger diagnosed turbulence.",
    "FLGHT": "Flight categories are 1 = LIFR (low instrument flight rules), 2 = IFR (instrument flight rules), 3 = MVFR (marginal visual flight rules), and 4 = VFR (visual flight rules). Lower categories indicate greater restriction.",
}


def _probability_condition(condition: str, unit: str) -> str:
    condition = condition.removesuffix(" <0")
    bounds = []
    for token in condition.split():
        for operator, description in (
            (">=", "at least"),
            ("<=", "at most"),
            (">", "above"),
            ("<", "below"),
        ):
            if token.startswith(operator):
                value = token.removeprefix(operator)
                bounds.append(
                    f"{description} {value}" + (f" {unit}" if unit != "1" else "")
                )
                break
        else:
            raise AssertionError(f"Unsupported probability threshold: {token!r}")
    return " and ".join(bounds)


def _window_name(name: str, duration: int) -> str:
    if not duration:
        return name
    assert name.endswith("_surface")
    return f"{name.removesuffix('_surface')}_{duration}_hour_surface"


def _variable(
    dims: Dims,
    field: ProductField,
    *,
    name: str,
    attrs: DataVarAttrs,
    families: tuple[ProductFamily, ...],
    duration: int,
    selectors: tuple[str, ...],
    filters: tuple[CodecConfig, ...] | None = None,
    has_statistic: bool = False,
) -> NoaaRefsDataVar:
    var_dims = tuple(dim for dim in dims[ROOT] if has_statistic or dim != "statistic")
    if duration:
        window_comment = (
            f"Threshold applies to the preceding {duration} hour{'s' if duration != 1 else ''} of accumulation."
            if families[0] in ("prob", "eas", "ffri")
            else f"Accumulated over the preceding {duration} hour{'s' if duration != 1 else ''}."
        )
        if duration >= 3:
            window_comment += (
                f" Published every 3 forecast hours from lead {duration} hours."
            )
        attrs = replace(
            attrs,
            step_type="accum",
            comment=" ".join(
                filter(
                    None,
                    (
                        window_comment,
                        attrs.comment,
                    ),
                )
            ),
        )
    return NoaaRefsDataVar(
        name=name,
        has_statistic=has_statistic,
        attrs=attrs,
        encoding=Encoding(
            dtype="float64",
            fill_value=np.nan,
            chunks=tuple(
                1059 if dim == "y" else 1799 if dim == "x" else 1 for dim in var_dims
            ),
            shards=None,
            compressors=(),
            filters=field.filters if filters is None else filters,
            serializer=GribberishCodec(
                var=field.element, adjust_longitude_range=True, north_up=True
            ).to_dict(),
        ),
        internal_attrs=NoaaRefsInternalAttrs(
            source_families=families,
            supported_lead_times=supported_leads(duration),
            grib_element=field.element,
            grib_description="",
            index_position=0,
            grib_index_level=field.level,
            grib_index_selectors=selectors,
            keep_mantissa_bits="no-rounding",
            hour_0_values_override=False,
            grib_index_step_type="instant" if field.attrs.step_type == "max" else None,
            window_duration=pd.Timedelta(hours=duration) if duration else None,
        ),
    )


def _statistic_attrs(field: ProductField, *, spread_only: bool) -> DataVarAttrs:
    element, level = field.element, field.level
    attrs = field.attrs
    if spread_only:
        attrs = replace(
            attrs,
            comment=" ".join(
                filter(
                    None,
                    (
                        attrs.comment,
                        "The mean slice is always missing; the probability-matched mean is provided as a separate variable.",
                    ),
                )
            ),
        )
    elif element in ("WIND", "VWSH"):
        attrs = replace(attrs, comment=_WIND_COMMENT)
    elif element == "CIN":
        attrs = replace(
            attrs,
            comment=" ".join(
                filter(
                    None,
                    (
                        attrs.comment,
                        "The mean is negative where there is inhibition and at or near zero where there is none.",
                    ),
                )
            ),
        )
    elif element in _OCCURRENCE_CATEGORIES:
        category = _OCCURRENCE_CATEGORIES[element]
        attrs = replace(
            attrs,
            comment=f"Mean divides the sum of weights of members diagnosing {category} by the total weight of contributing members. Spread is the weighted population standard deviation of the same 0/1 occurrence indicator.",
        )
    elif (element, level) == ("HGT", "surface"):
        attrs = replace(
            attrs,
            comment="Spread reflects differences in terrain height between member grids.",
        )
    if element in _SPARSE_RADAR_COMMENTS:
        attrs = replace(
            attrs,
            comment=" ".join(filter(None, (attrs.comment, _RADAR_SPREAD_COMMENT))),
        )
    return attrs


def _probability_attrs(
    field: ProductField,
    *,
    name: str,
    prefix: str,
    family: ProductFamily,
    duration: int,
    condition: str,
) -> DataVarAttrs:
    element, level = field.element, field.level
    probability_label = prefix.replace("_", " ").capitalize()
    if family == "ffri":
        number = condition.removeprefix(">")
        reference = (
            f"{number} hour flash flood guidance depth"
            if element == "PPFFG"
            else f"{number} year average recurrence interval depth"
        )
        description = f"{duration} hour precipitation above the {reference}"
        comment = f"{probability_label} {description}. {_FFRI_COMMENT}"
    elif element == "JFWPRB":
        description = "10 m wind speed at least 9 m s-1 and 2 m relative humidity at most 20 percent"
        comment = f"{probability_label} {description}."
    elif element in _OCCURRENCE_CATEGORIES:
        category = _OCCURRENCE_CATEGORIES[element]
        description = f"categorical {category} ({level})"
        comment = f"Percentage of members diagnosing {category}."
    else:
        quantity = field.attrs.long_name.lower()
        if duration:
            quantity = f"{duration} hour {quantity}"
        reader_condition = _probability_condition(
            condition,
            _PROBABILITY_THRESHOLD_UNITS.get(element, field.attrs.units),
        )
        level_label = _PROBABILITY_LEVEL_LABELS.get(level, level)
        description = f"{quantity} ({level_label}) {reader_condition}"
        comment = f"{probability_label} {description}."
        if element in ("CAPE", "CIN") and field.attrs.comment:
            comment += f" {field.attrs.comment}"
        if element == "CIN":
            comment += " Convective inhibition is negative where there is inhibition and at or near zero where there is none."
        if element in _PROBABILITY_CATEGORY_COMMENTS:
            comment += f" {_PROBABILITY_CATEGORY_COMMENTS[element]}"
    if field.attrs.step_type == "max":
        comment += " Maximum over the hour ending at the valid time."
    return DataVarAttrs(
        short_name=name,
        long_name=f"{probability_label} {description}",
        units="percent",
        step_type="max" if field.attrs.step_type == "max" else "instant",
        comment=comment,
    )


def data_vars(dims: Dims) -> Sequence[NoaaRefsDataVar]:
    result = []
    for element, level, duration, selectors in _STATISTICS:
        field = _FIELDS[element, level]
        families: tuple[ProductFamily, ...] = (
            ("sprd",) if selectors == ("process=193",) else ("mean", "sprd")
        )
        attrs = _statistic_attrs(field, spread_only=families == ("sprd",))
        name = _window_name(field.name, duration)
        has_offset = any(
            codec.get("name") == "scale_offset"
            and codec.get("configuration", {}).get("offset", 0) != 0
            for codec in field.filters
        )
        if has_offset:
            assert families == ("mean", "sprd")
            attrs = replace(
                attrs,
                comment=" ".join(
                    filter(
                        None,
                        (
                            attrs.comment,
                            f"The standard_deviation slice is deliberately empty (NaN). The mean slice duplicates {name}_mean; standard deviation is provided by {name}_standard_deviation without the statistic dimension.",
                        ),
                    )
                ),
            )
        result.append(
            _variable(
                dims,
                field,
                name=name,
                attrs=attrs,
                families=("mean",) if has_offset else families,
                duration=duration,
                selectors=selectors,
                has_statistic=True,
            )
        )
        if has_offset:
            for label, family in (("mean", "mean"), ("standard_deviation", "sprd")):
                alias_name = f"{name}_{label}"
                alias_attrs = replace(
                    field.attrs,
                    comment=" ".join(
                        filter(
                            None,
                            (
                                field.attrs.comment,
                                f"Ensemble mean in degree_Celsius, identical to the mean slice of {name}."
                                if label == "mean"
                                else "NOAA spread: member-weighted population standard deviation with weights normalized over contributing members. This is a temperature difference in K, numerically equal to the difference in degree_Celsius; no temperature offset is applied.",
                            ),
                        )
                    ),
                )
                if label == "standard_deviation":
                    alias_attrs = replace(
                        alias_attrs,
                        short_name=alias_name,
                        long_name=f"Standard deviation of {field.attrs.long_name.lower()}",
                        standard_name=None,
                        units="K",
                    )
                result.append(
                    _variable(
                        dims,
                        field,
                        name=alias_name,
                        attrs=alias_attrs,
                        families=(family,),
                        duration=duration,
                        selectors=selectors,
                        filters=tuple(
                            codec
                            for codec in field.filters
                            if codec.get("name") != "scale_offset"
                            or codec.get("configuration", {}).get("offset", 0) == 0
                        )
                        if label == "standard_deviation"
                        else None,
                    )
                )
    for family, element, level, duration, selectors in _SPARSE:
        field = _FIELDS[element, level]
        prefix, description = {
            "pmmn": ("probability_matched_mean", "Probability-matched mean"),
            "lpmm": (
                "localized_probability_matched_mean",
                "Localized probability-matched mean",
            ),
            "avrg": (
                "mean_probability_matched_mean_average",
                "Average of the ensemble mean and probability-matched mean",
            ),
        }[family]
        name = f"{prefix}_{_window_name(field.name, duration)}"
        attrs = replace(
            field.attrs,
            short_name=name,
            long_name=f"{description} of {field.attrs.long_name.lower()}",
            comment=" ".join(
                filter(
                    None,
                    (field.attrs.comment, _SPARSE_RADAR_COMMENTS.get(element)),
                )
            )
            or None,
        )
        result.append(
            _variable(
                dims,
                field,
                name=name,
                attrs=attrs,
                families=(family,),
                duration=duration,
                selectors=selectors,
            )
        )
    for family, element, level, duration, thresholds, extra in _PROBABILITIES:
        field = _FIELDS[element, level]
        for threshold in thresholds:
            condition = threshold.removeprefix("prob ")
            bound = (
                condition.replace(">=", "at_least_")
                .replace(">", "above_")
                .replace("<", "below_")
                .replace(" ", "_")
                .replace("-", "minus_")
                .replace(".", "p")
            )
            if bound.endswith("_below_0"):
                bound = bound.removesuffix("_below_0")
            base = _window_name(field.name, duration)
            if family == "ffri":
                number = condition.removeprefix(">")
                base = (
                    f"precipitation_exceeding_{number}_hour_flash_flood_guidance_{duration}_hour_surface"
                    if element == "PPFFG"
                    else f"precipitation_exceeding_{number}_year_average_recurrence_interval_{duration}_hour_surface"
                )
                bound = ""
            if element == "JFWPRB":
                base = "wind_speed_10m_at_least_9_and_relative_humidity_2m_at_most_20"
                bound = ""
            elif element in _OCCURRENCE_CATEGORIES:
                category = _OCCURRENCE_CATEGORIES[element].replace(" ", "_")
                base = f"categorical_{category}_surface"
                bound = ""
            prefix = (
                "ensemble_agreement_scale_probability_of"
                if family == "eas"
                else "neighborhood_probability_of"
                if extra == ("Neighborhood Probability",)
                else "probability_of"
            )
            name = "_".join(filter(None, (prefix, base, bound)))
            attrs = _probability_attrs(
                field,
                name=name,
                prefix=prefix,
                family=family,
                duration=duration,
                condition=condition,
            )
            result.append(
                _variable(
                    dims,
                    field,
                    name=name,
                    attrs=attrs,
                    families=(family,),
                    duration=duration,
                    selectors=(threshold, *extra),
                    filters=(),
                )
            )
    return result
