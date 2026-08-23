"""The ECMWF-origin S2S selection manifest, and how it is split into ECDS requests.

An ECDS request is a hyper-rectangle over
`variable x level_value x leadtime_hour x date`, so variables can only share a
request when they share a level set and a lead time set. The tables below record
which variables share which, for `origin=ecmwf`.
"""

from collections.abc import Sequence
from typing import Any, Final, Literal

import pandas as pd

from reformatters.common.iterating import digest
from reformatters.common.pydantic import FrozenBaseModel

type LevelType = Literal["single_level", "pressure", "isentropic"]
type ForecastType = Literal["control_forecast", "perturbed_forecast"]
type SelectionGroup = tuple[LevelType, tuple[str, ...], tuple[str, ...], bool]

ECMWF_ORIGIN: Final[str] = "ecmwf"

# ECDS charges a request as if it returned all 101 members, even a control request.
ENSEMBLE_SIZE: Final[int] = 101
CONTROL_MEMBER: Final[int] = 0
PERTURBED_MEMBERS: Final[tuple[int, ...]] = tuple(range(1, ENSEMBLE_SIZE))

MAXIMUM_LEAD_HOUR: Final[int] = 1104

SIX_HOURLY_LEAD_TIMES: Final[tuple[str, ...]] = tuple(
    str(hour) for hour in range(0, MAXIMUM_LEAD_HOUR + 1, 6)
)
SIX_HOURLY_LEAD_TIMES_FROM_6H: Final[tuple[str, ...]] = SIX_HOURLY_LEAD_TIMES[1:]
DAILY_LEAD_TIMES: Final[tuple[str, ...]] = tuple(
    str(hour) for hour in range(0, MAXIMUM_LEAD_HOUR + 1, 24)
)
DAILY_MEAN_LEAD_TIMES: Final[tuple[str, ...]] = tuple(
    f"{hour}_{hour + 24}" for hour in range(0, MAXIMUM_LEAD_HOUR, 24)
)

PRESSURE_LEVELS: Final[tuple[str, ...]] = (
    "10_hpa",
    "50_hpa",
    "100_hpa",
    "200_hpa",
    "300_hpa",
    "500_hpa",
    "700_hpa",
    "850_hpa",
    "925_hpa",
    "1000_hpa",
)
SPECIFIC_HUMIDITY_PRESSURE_LEVELS: Final[tuple[str, ...]] = PRESSURE_LEVELS[3:]

SINGLE_LEVEL_LEAD_TIMES: Final[dict[str, tuple[str, ...]]] = {
    variable: lead_times
    for lead_times, variables in (
        (
            SIX_HOURLY_LEAD_TIMES,
            (
                "10_m_u_component_of_wind",
                "10_m_v_component_of_wind",
                "total_precipitation",
            ),
        ),
        (
            SIX_HOURLY_LEAD_TIMES_FROM_6H,
            (
                "maximum_2_m_temperature_in_the_last_6_hours",
                "minimum_2_m_temperature_in_the_last_6_hours",
            ),
        ),
        (
            DAILY_LEAD_TIMES,
            (
                "convective_precipitation",
                "eastward_turbulent_surface_stress",
                "land_sea_mask",
                "mean_sea_level_pressure",
                "northward_turbulent_surface_stress",
                "orography",
                "snow_fall_water_equivalent",
                "soil_type",
                "surface_latent_heat_flux",
                "surface_net_solar_radiation",
                "surface_net_thermal_radiation",
                "surface_pressure",
                "surface_runoff",
                "surface_sensible_heat_flux",
                "surface_solar_radiation_downwards",
                "surface_thermal_radiation_downwards",
                "top_net_thermal_radiation",
                "water_runoff_and_drainage",
            ),
        ),
        (
            DAILY_MEAN_LEAD_TIMES,
            (
                "2_m_dewpoint_temperature",
                "2_m_temperature",
                "convective_available_potential_energy",
                "sea_ice_area_fraction",
                "sea_surface_temperature",
                "skin_temperature",
                "snow_albedo",
                "snow_density",
                "snow_depth_water_equivalent",
                "soil_moisture_top_100_cm",
                "soil_moisture_top_20_cm",
                "soil_temperature_top_100_cm",
                "soil_temperature_top_20_cm",
                "total_cloud_cover",
                "total_column_water",
            ),
        ),
    )
    for variable in variables
}

# Static fields published only as a control forecast.
CONTROL_ONLY_VARIABLES: Final[frozenset[str]] = frozenset(
    {"land_sea_mask", "orography", "soil_type"}
)

PRESSURE_LEVEL_VARIABLES: Final[dict[str, tuple[str, ...]]] = {
    "geopotential_height": PRESSURE_LEVELS,
    "specific_humidity": SPECIFIC_HUMIDITY_PRESSURE_LEVELS,
    "temperature": PRESSURE_LEVELS,
    "u_component_of_wind": PRESSURE_LEVELS,
    "v_component_of_wind": PRESSURE_LEVELS,
    "vertical_velocity": PRESSURE_LEVELS,
}

ISENTROPIC_LEVELS: Final[tuple[str, ...]] = ("320_k",)
ISENTROPIC_VARIABLES: Final[tuple[str, ...]] = ("potential_vorticity",)

# Mean bytes per 121 x 240 simple-packed message, measured across a full initialization.
MEAN_MESSAGE_BYTES: Final[int] = 55_691
# Sized to keep a single blob within a worker's ephemeral disk and within the
# window in which an ECDS result stays downloadable, not against the request cost cap.
DEFAULT_MAXIMUM_SHARD_BYTES: Final[int] = 4_000_000_000


class EcdsSelection(FrozenBaseModel):
    """One ECDS request's hyper-rectangle, excluding the initialization date."""

    level_type: LevelType
    forecast_type: ForecastType
    variables: tuple[str, ...]
    level_values: tuple[str, ...]
    lead_time_labels: tuple[str, ...]

    @property
    def field_count(self) -> int:
        return len(self.variables) * max(len(self.level_values), 1)

    @property
    def ensemble_members(self) -> tuple[int, ...]:
        if self.forecast_type == "control_forecast":
            return (CONTROL_MEMBER,)
        return PERTURBED_MEMBERS

    @property
    def message_count(self) -> int:
        return (
            self.field_count * len(self.ensemble_members) * len(self.lead_time_labels)
        )

    @property
    def estimated_bytes(self) -> int:
        return self.message_count * MEAN_MESSAGE_BYTES

    @property
    def cost(self) -> int:
        """The ECDS `size` cost of retrieving this selection for one date."""
        return ENSEMBLE_SIZE * self.field_count * len(self.lead_time_labels)

    @property
    def file_name(self) -> str:
        name = "-".join(
            (self.level_type, self.forecast_type, self.variables[0], self._digest)
        )
        return f"{name}.grib2"

    @property
    def _digest(self) -> str:
        return digest(
            (*self.variables, *self.level_values, *self.lead_time_labels),
        )

    def inputs(self, init_time: pd.Timestamp) -> dict[str, Any]:
        inputs: dict[str, Any] = {
            "origin": ECMWF_ORIGIN,
            "forecast_type": self.forecast_type,
            "level_type": self.level_type,
            "variable": list(self.variables),
            "year": [init_time.strftime("%Y")],
            "month": [init_time.strftime("%m")],
            "day": [init_time.strftime("%d")],
            "time": [init_time.strftime("%H:%M")],
            "leadtime_hour": list(self.lead_time_labels),
            "data_format": "grib",
        }
        if self.level_values:
            inputs["level_value"] = list(self.level_values)
        return inputs


def initialization_selections(
    variables: Sequence[str],
    maximum_shard_bytes: int = DEFAULT_MAXIMUM_SHARD_BYTES,
) -> list[EcdsSelection]:
    """Split `variables` into the ECDS requests that archive one whole initialization."""
    assert len(variables) > 0
    groups: dict[SelectionGroup, list[str]] = {}
    for variable in sorted(set(variables)):
        groups.setdefault(_selection_group(variable), []).append(variable)

    selections = []
    for (
        level_type,
        level_values,
        lead_time_labels,
        control_only,
    ), group_variables in sorted(groups.items()):
        forecast_types: tuple[ForecastType, ...] = (
            ("control_forecast",)
            if control_only
            else ("control_forecast", "perturbed_forecast")
        )
        for forecast_type in forecast_types:
            selections.extend(
                split_by_estimated_size(
                    EcdsSelection(
                        level_type=level_type,
                        forecast_type=forecast_type,
                        variables=tuple(group_variables),
                        level_values=level_values,
                        lead_time_labels=lead_time_labels,
                    ),
                    maximum_shard_bytes,
                )
            )
    return selections


def split_by_estimated_size(
    selection: EcdsSelection, maximum_shard_bytes: int
) -> list[EcdsSelection]:
    """Split a selection along its variable axis until each shard's blob is small enough."""
    bytes_per_variable = selection.estimated_bytes // len(selection.variables)
    variables_per_shard = max(1, maximum_shard_bytes // bytes_per_variable)
    return [
        selection.model_copy(
            update={
                "variables": selection.variables[start : start + variables_per_shard]
            }
        )
        for start in range(0, len(selection.variables), variables_per_shard)
    ]


def _selection_group(variable: str) -> SelectionGroup:
    control_only = variable in CONTROL_ONLY_VARIABLES
    if variable in SINGLE_LEVEL_LEAD_TIMES:
        return "single_level", (), SINGLE_LEVEL_LEAD_TIMES[variable], control_only
    if variable in PRESSURE_LEVEL_VARIABLES:
        return (
            "pressure",
            PRESSURE_LEVEL_VARIABLES[variable],
            DAILY_LEAD_TIMES,
            control_only,
        )
    if variable in ISENTROPIC_VARIABLES:
        return "isentropic", ISENTROPIC_LEVELS, DAILY_LEAD_TIMES, control_only
    raise ValueError(f"{variable} is not an ECMWF-origin S2S variable")
