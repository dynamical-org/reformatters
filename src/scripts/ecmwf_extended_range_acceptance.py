import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Protocol

from gribberish import parse_grib_message_metadata  # ty: ignore[unresolved-import]

from reformatters.common.logging import get_logger

log = get_logger(__name__)

VARIABLE_GRIB_SELECTORS = {
    "2_m_dewpoint_temperature": ("DPT", "specific height level above ground"),
    "2_m_temperature": ("TMP", "specific height level above ground"),
    "10_m_u_component_of_wind": ("UGRD", "specific height level above ground"),
    "10_m_v_component_of_wind": ("VGRD", "specific height level above ground"),
    "geopotential_height": ("HGT", "isobaric surface"),
    "mean_sea_level_pressure": ("PRES", "mean sea level"),
    "soil_moisture_top_20_cm": ("SM", "depth below land surface"),
    "specific_humidity": ("SPFH", "isobaric surface"),
    "surface_pressure": ("PRES", "ground or water surface"),
    "surface_runoff": ("missing", "ground or water surface"),
    "surface_solar_radiation_downwards": ("dswrf", "ground or water surface"),
    "surface_thermal_radiation_downwards": ("DLWRF", "ground or water surface"),
    "temperature": ("TMP", "isobaric surface"),
    "total_cloud_cover": ("TCDC", "ground or water surface"),
    "total_precipitation": ("TP", "ground or water surface"),
    "u_component_of_wind": ("UGRD", "isobaric surface"),
    "v_component_of_wind": ("VGRD", "isobaric surface"),
}


class GribMetadata(Protocol):
    message_size: int
    var_abbrev: str
    level_type: str
    level_value: float | None
    perturbation_number: int | None
    reference_date: datetime
    forecast_date: datetime
    forecast_date_end: datetime | None
    units: str
    grid_shape: tuple[int, int]


@dataclass(frozen=True, order=True)
class InventoryKey:
    variable: str
    level: str | None
    ensemble_member: int
    leadtime_hour: str


@dataclass(frozen=True)
class InventoryRecord:
    offset: int
    message_size: int
    variable: str
    level: str | None
    ensemble_member: int
    leadtime_hour: str
    units: str
    grib_level_type: str
    grib_level_value: float | None
    grid_shape: tuple[int, int]

    @property
    def key(self) -> InventoryKey:
        return InventoryKey(
            self.variable,
            self.level,
            self.ensemble_member,
            self.leadtime_hour,
        )


@dataclass(frozen=True)
class GribInventory:
    source: str
    valid: bool
    byte_count: int
    message_count: int
    grid_shape: list[int]
    variables: list[str]
    levels: list[str]
    members: list[int]
    leadtimes: list[str]
    records: list[InventoryRecord]


def inspect_grib(source: Path, payload: Mapping[str, object]) -> GribInventory:
    requested_variables = _strings(payload.get("variable"))
    requested_leadtimes = _strings(payload.get("leadtime_hour"))
    expected = Counter(
        InventoryKey(variable, level, member, leadtime)
        for variable, level, member, leadtime in product(
            requested_variables,
            _expected_levels(payload),
            _expected_members(payload),
            requested_leadtimes,
        )
    )

    records: list[InventoryRecord] = []
    with source.open("rb") as source_file:
        offset = 0
        while header := source_file.read(16):
            assert len(header) == 16
            assert header[:4] == b"GRIB", f"Expected GRIB message at byte {offset}"
            message_size = int.from_bytes(header[8:16], byteorder="big")
            assert message_size >= 20
            message = header + source_file.read(message_size - len(header))
            assert len(message) == message_size
            assert message.endswith(b"7777")
            metadata: GribMetadata = parse_grib_message_metadata(message, 0)
            assert metadata.message_size == message_size
            records.append(
                _inventory_record(
                    offset,
                    metadata,
                    payload,
                    requested_variables,
                    requested_leadtimes,
                )
            )
            offset += message_size
        assert offset == source.stat().st_size

    actual = Counter(record.key for record in records)
    missing = sorted((expected - actual).elements())
    unexpected = sorted((actual - expected).elements())
    assert not missing, f"Missing inventory: {missing}"
    assert not unexpected, f"Unexpected inventory: {unexpected}"

    grid_shapes = {record.grid_shape for record in records}
    assert len(grid_shapes) == 1
    grid_shape = list(grid_shapes.pop())
    return GribInventory(
        source=str(source),
        valid=True,
        byte_count=source.stat().st_size,
        message_count=len(records),
        grid_shape=grid_shape,
        variables=sorted({record.variable for record in records}),
        levels=sorted(
            {record.level for record in records if record.level is not None},
            key=_level_sort_key,
        ),
        members=sorted({record.ensemble_member for record in records}),
        leadtimes=sorted(
            {record.leadtime_hour for record in records}, key=_leadtime_sort_key
        ),
        records=records,
    )


def _inventory_record(
    offset: int,
    metadata: GribMetadata,
    payload: Mapping[str, object],
    requested_variables: Sequence[str],
    requested_leadtimes: Sequence[str],
) -> InventoryRecord:
    matching_variables = [
        variable
        for variable in requested_variables
        if VARIABLE_GRIB_SELECTORS.get(variable)
        == (metadata.var_abbrev, metadata.level_type)
    ]
    assert len(matching_variables) == 1, (
        f"Could not map {metadata.var_abbrev!r} to one requested variable: "
        f"{matching_variables}"
    )
    level = (
        _pressure_level(metadata.level_value)
        if payload.get("level_type") == "pressure"
        else None
    )
    return InventoryRecord(
        offset=offset,
        message_size=metadata.message_size,
        variable=matching_variables[0],
        level=level,
        ensemble_member=metadata.perturbation_number or 0,
        leadtime_hour=_requested_leadtime(metadata, requested_leadtimes),
        units=metadata.units,
        grib_level_type=metadata.level_type,
        grib_level_value=metadata.level_value,
        grid_shape=metadata.grid_shape,
    )


def _requested_leadtime(
    metadata: GribMetadata, requested_leadtimes: Sequence[str]
) -> str:
    start = _hours(metadata.forecast_date - metadata.reference_date)
    end = _hours(
        (metadata.forecast_date_end or metadata.forecast_date) - metadata.reference_date
    )
    matches = [
        leadtime
        for leadtime in requested_leadtimes
        if _leadtime_matches(leadtime, start, end)
    ]
    assert len(matches) == 1, (
        f"Could not map GRIB interval {start}-{end} to one requested lead time: "
        f"{matches}"
    )
    return matches[0]


def _leadtime_matches(leadtime: str, start: int, end: int) -> bool:
    if "_" not in leadtime:
        return int(leadtime) == end
    requested_start, requested_end = (int(value) for value in leadtime.split("_"))
    return (requested_start, requested_end) == (start, end)


def _expected_members(payload: Mapping[str, object]) -> list[int]:
    if payload.get("forecast_type") == "control_forecast":
        return [0]
    members = [int(value) for value in _strings(payload.get("number"))]
    assert members, "Perturbed requests must specify number for strict inventory"
    return members


def _expected_levels(payload: Mapping[str, object]) -> Sequence[str | None]:
    if payload.get("level_type") != "pressure":
        return [None]
    levels = _strings(payload.get("level_value"))
    assert levels
    return levels


def _pressure_level(level_value: float | None) -> str:
    assert level_value is not None
    level_hpa = level_value / 100 if level_value > 2_000 else level_value
    assert level_hpa.is_integer()
    return f"{int(level_hpa)}_hpa"


def _hours(duration: timedelta) -> int:
    total_seconds = duration.total_seconds()
    hours = total_seconds / 3_600
    assert hours.is_integer()
    return int(hours)


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return []


def _leadtime_sort_key(value: str) -> tuple[int, int]:
    parts = [int(part) for part in value.split("_")]
    return (parts[-1], parts[0])


def _level_sort_key(value: str) -> int:
    return int(value.removesuffix("_hpa"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    payload = json.loads(arguments.payload.read_text())
    inventory = inspect_grib(arguments.source, payload)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = arguments.output.with_suffix(f"{arguments.output.suffix}.tmp")
    temporary_path.write_text(json.dumps(asdict(inventory), indent=2, sort_keys=True))
    temporary_path.replace(arguments.output)
    log.info(
        "Validated %d GRIB messages in %s",
        inventory.message_count,
        arguments.source,
    )


if __name__ == "__main__":
    main()
