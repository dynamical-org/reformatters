"""Which ECDS variable each GRIB2 message in a staged blob answers.

ECDS names a variable in a request but the returned blob names nothing: the only
identity a message carries is its GRIB2 discipline, parameter and fixed surfaces.
`FIELD_KEYS` records that mapping for every ECMWF-origin S2S variable, decoded from
real messages, so a blob's inventory can be checked by name rather than by count.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

from .request_shards import ISENTROPIC_LEVELS, PRESSURE_LEVEL_VARIABLES

GRIB_SECTION_0_BYTES: Final[int] = 16
DISCIPLINE_OFFSET: Final[int] = 6
PRODUCT_DEFINITION_SECTION: Final[int] = 4
# Octets 23-34 of a section 4 product definition template hold the two fixed
# surfaces; the templates S2S uses (4.1 and 4.11) place them identically.
FIRST_SURFACE_OFFSET: Final[int] = 22
SECOND_SURFACE_OFFSET: Final[int] = 28
MISSING_SURFACE_TYPE: Final[int] = 255
MISSING_SCALED_VALUE: Final[int] = 0xFFFFFFFF

type Surface = tuple[int, float | None]

GROUND_OR_WATER_SURFACE: Final[Surface] = (1, None)
TOP_OF_ATMOSPHERE: Final[Surface] = (8, None)
MEAN_SEA_LEVEL: Final[Surface] = (101, None)
NO_SURFACE: Final[Surface] = (MISSING_SURFACE_TYPE, None)


def height_above_ground(metres: float) -> Surface:
    return 103, metres


def depth_below_land(metres: float) -> Surface:
    return 106, metres


def isobaric(hectopascals: float) -> Surface:
    return 100, hectopascals * 100


def isentropic(kelvin: float) -> Surface:
    return 107, kelvin


@dataclass(frozen=True, order=True)
class FieldKey:
    """The GRIB2 identity of one field, independent of member and lead time."""

    discipline: int
    parameter_category: int
    parameter_number: int
    first_surface: Surface
    second_surface: Surface = NO_SURFACE
    statistical_process: str | None = None


def _pressure_level_field_keys() -> dict[tuple[str, str], FieldKey]:
    parameters = {
        "geopotential_height": (0, 3, 5),
        "specific_humidity": (0, 1, 0),
        "temperature": (0, 0, 0),
        "u_component_of_wind": (0, 2, 2),
        "v_component_of_wind": (0, 2, 3),
        "vertical_velocity": (0, 2, 8),
    }
    return {
        (variable, level): FieldKey(
            *parameters[variable], isobaric(float(level.removesuffix("_hpa")))
        )
        for variable, levels in PRESSURE_LEVEL_VARIABLES.items()
        for level in levels
    }


# Keyed by (ECDS variable, level value); the level is "" for a single-level variable.
FIELD_KEYS: Final[dict[tuple[str, str], FieldKey]] = {
    ("10_m_u_component_of_wind", ""): FieldKey(0, 2, 2, height_above_ground(10)),
    ("10_m_v_component_of_wind", ""): FieldKey(0, 2, 3, height_above_ground(10)),
    ("2_m_dewpoint_temperature", ""): FieldKey(
        0, 0, 6, height_above_ground(2), statistical_process="average"
    ),
    ("2_m_temperature", ""): FieldKey(
        0, 0, 0, height_above_ground(2), statistical_process="average"
    ),
    ("convective_available_potential_energy", ""): FieldKey(
        0, 7, 6, GROUND_OR_WATER_SURFACE, TOP_OF_ATMOSPHERE, "average"
    ),
    ("convective_precipitation", ""): FieldKey(
        0, 1, 37, GROUND_OR_WATER_SURFACE, statistical_process="accumulation"
    ),
    ("eastward_turbulent_surface_stress", ""): FieldKey(
        0, 2, 38, GROUND_OR_WATER_SURFACE, statistical_process="accumulation"
    ),
    ("land_sea_mask", ""): FieldKey(2, 0, 0, GROUND_OR_WATER_SURFACE),
    ("maximum_2_m_temperature_in_the_last_6_hours", ""): FieldKey(
        0, 0, 0, height_above_ground(2), statistical_process="maximum"
    ),
    ("mean_sea_level_pressure", ""): FieldKey(0, 3, 0, MEAN_SEA_LEVEL),
    ("minimum_2_m_temperature_in_the_last_6_hours", ""): FieldKey(
        0, 0, 0, height_above_ground(2), statistical_process="minimum"
    ),
    ("northward_turbulent_surface_stress", ""): FieldKey(
        0, 2, 37, GROUND_OR_WATER_SURFACE, statistical_process="accumulation"
    ),
    ("orography", ""): FieldKey(0, 3, 5, GROUND_OR_WATER_SURFACE),
    ("sea_ice_area_fraction", ""): FieldKey(
        10, 2, 0, GROUND_OR_WATER_SURFACE, statistical_process="average"
    ),
    ("sea_surface_temperature", ""): FieldKey(
        10, 3, 0, GROUND_OR_WATER_SURFACE, statistical_process="average"
    ),
    ("skin_temperature", ""): FieldKey(
        0, 0, 17, GROUND_OR_WATER_SURFACE, statistical_process="average"
    ),
    ("snow_albedo", ""): FieldKey(
        0, 19, 19, GROUND_OR_WATER_SURFACE, statistical_process="average"
    ),
    ("snow_density", ""): FieldKey(
        0, 1, 61, GROUND_OR_WATER_SURFACE, statistical_process="average"
    ),
    ("snow_depth_water_equivalent", ""): FieldKey(
        0, 1, 60, GROUND_OR_WATER_SURFACE, statistical_process="average"
    ),
    ("snow_fall_water_equivalent", ""): FieldKey(
        0, 1, 53, GROUND_OR_WATER_SURFACE, statistical_process="accumulation"
    ),
    ("soil_moisture_top_100_cm", ""): FieldKey(
        2, 0, 22, depth_below_land(0), depth_below_land(1), "average"
    ),
    ("soil_moisture_top_20_cm", ""): FieldKey(
        2, 0, 22, depth_below_land(0), depth_below_land(0.2), "average"
    ),
    ("soil_temperature_top_100_cm", ""): FieldKey(
        2, 0, 2, depth_below_land(0), depth_below_land(1), "average"
    ),
    ("soil_temperature_top_20_cm", ""): FieldKey(
        2, 0, 2, depth_below_land(0), depth_below_land(0.2), "average"
    ),
    ("soil_type", ""): FieldKey(2, 3, 0, GROUND_OR_WATER_SURFACE),
    ("surface_latent_heat_flux", ""): FieldKey(
        0, 0, 10, GROUND_OR_WATER_SURFACE, statistical_process="accumulation"
    ),
    ("surface_net_solar_radiation", ""): FieldKey(
        0, 4, 9, GROUND_OR_WATER_SURFACE, statistical_process="accumulation"
    ),
    ("surface_net_thermal_radiation", ""): FieldKey(
        0, 5, 5, GROUND_OR_WATER_SURFACE, statistical_process="accumulation"
    ),
    ("surface_pressure", ""): FieldKey(0, 3, 0, GROUND_OR_WATER_SURFACE),
    ("surface_runoff", ""): FieldKey(
        2, 0, 34, GROUND_OR_WATER_SURFACE, statistical_process="accumulation"
    ),
    ("surface_sensible_heat_flux", ""): FieldKey(
        0, 0, 11, GROUND_OR_WATER_SURFACE, statistical_process="accumulation"
    ),
    ("surface_solar_radiation_downwards", ""): FieldKey(
        0, 4, 7, GROUND_OR_WATER_SURFACE, statistical_process="accumulation"
    ),
    ("surface_thermal_radiation_downwards", ""): FieldKey(
        0, 5, 3, GROUND_OR_WATER_SURFACE, statistical_process="accumulation"
    ),
    ("top_net_thermal_radiation", ""): FieldKey(
        0, 5, 5, TOP_OF_ATMOSPHERE, statistical_process="accumulation"
    ),
    ("total_cloud_cover", ""): FieldKey(
        0, 6, 1, GROUND_OR_WATER_SURFACE, TOP_OF_ATMOSPHERE, "average"
    ),
    ("total_column_water", ""): FieldKey(
        0, 1, 51, GROUND_OR_WATER_SURFACE, TOP_OF_ATMOSPHERE, "average"
    ),
    ("total_precipitation", ""): FieldKey(
        0, 1, 52, GROUND_OR_WATER_SURFACE, statistical_process="accumulation"
    ),
    ("water_runoff_and_drainage", ""): FieldKey(
        2, 0, 33, GROUND_OR_WATER_SURFACE, statistical_process="accumulation"
    ),
    **{
        ("potential_vorticity", level): FieldKey(
            0, 2, 14, isentropic(float(level.removesuffix("_k")))
        )
        for level in ISENTROPIC_LEVELS
    },
    **_pressure_level_field_keys(),
}

VARIABLES_BY_FIELD_KEY: Final[dict[FieldKey, tuple[str, str]]] = {
    field_key: variable_and_level
    for variable_and_level, field_key in FIELD_KEYS.items()
}
assert len(VARIABLES_BY_FIELD_KEY) == len(FIELD_KEYS), "Two variables share a FieldKey"


def field_key(message: bytes, statistical_process: str | None) -> FieldKey:
    """The GRIB2 identity of `message`, read from its discipline and section 4."""
    product_definition = _section(message, PRODUCT_DEFINITION_SECTION)
    return FieldKey(
        discipline=message[DISCIPLINE_OFFSET],
        parameter_category=product_definition[9],
        parameter_number=product_definition[10],
        first_surface=_surface(product_definition, FIRST_SURFACE_OFFSET),
        second_surface=_surface(product_definition, SECOND_SURFACE_OFFSET),
        statistical_process=statistical_process,
    )


def _sections(message: bytes) -> Iterator[tuple[int, bytes]]:
    offset = GRIB_SECTION_0_BYTES
    while offset < len(message) - len(b"7777"):
        length = int.from_bytes(message[offset : offset + 4], "big")
        assert length > 0, "GRIB2 section with no length"
        yield message[offset + 4], message[offset : offset + length]
        offset += length


def _section(message: bytes, number: int) -> bytes:
    for section_number, body in _sections(message):
        if section_number == number:
            return body
    raise AssertionError(f"GRIB2 message has no section {number}")


def _surface(product_definition: bytes, offset: int) -> Surface:
    surface_type = product_definition[offset]
    scale_factor = product_definition[offset + 1]
    scaled_value = int.from_bytes(product_definition[offset + 2 : offset + 6], "big")
    if surface_type == MISSING_SURFACE_TYPE or scaled_value == MISSING_SCALED_VALUE:
        return surface_type, None
    return surface_type, scaled_value / 10.0**scale_factor
