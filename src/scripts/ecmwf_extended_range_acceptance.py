import argparse
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Protocol

import numpy as np
import rasterio
import zarr
from gribberish import parse_grib_message_metadata  # ty: ignore[unresolved-import]

from reformatters.common.deaccumulation import (
    PRECIPITATION_RATE_INVALID_BELOW_THRESHOLD,
    RADIATION_INVALID_BELOW_THRESHOLD,
)
from reformatters.common.logging import get_logger

log = get_logger(__name__)

ENSEMBLE_CHUNK_SIZE = 10

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


@dataclass(frozen=True)
class VariableSpec:
    name: str
    long_name: str
    units: str
    standard_name: str | None = None
    cumulative_rate: bool = False


VARIABLE_SPECS = {
    "2_m_dewpoint_temperature": VariableSpec(
        "dew_point_temperature_2m",
        "2 metre dewpoint temperature",
        "degree_Celsius",
        "dew_point_temperature",
    ),
    "2_m_temperature": VariableSpec(
        "temperature_2m",
        "2 metre temperature",
        "degree_Celsius",
        "air_temperature",
    ),
    "10_m_u_component_of_wind": VariableSpec(
        "wind_u_10m", "10 metre U wind component", "m s-1", "eastward_wind"
    ),
    "10_m_v_component_of_wind": VariableSpec(
        "wind_v_10m", "10 metre V wind component", "m s-1", "northward_wind"
    ),
    "mean_sea_level_pressure": VariableSpec(
        "pressure_reduced_to_mean_sea_level",
        "Pressure reduced to MSL",
        "Pa",
        "air_pressure_at_mean_sea_level",
    ),
    "soil_moisture_top_20_cm": VariableSpec(
        "soil_moisture_0_20cm", "Soil moisture top 20 cm", "kg m-3"
    ),
    "surface_pressure": VariableSpec(
        "pressure_surface", "Surface pressure", "Pa", "surface_air_pressure"
    ),
    "surface_runoff": VariableSpec(
        "runoff_surface",
        "Surface runoff rate",
        "kg m-2 s-1",
        cumulative_rate=True,
    ),
    "surface_solar_radiation_downwards": VariableSpec(
        "downward_short_wave_radiation_flux_surface",
        "Surface downward short-wave radiation flux",
        "W m-2",
        "surface_downwelling_shortwave_flux_in_air",
        cumulative_rate=True,
    ),
    "surface_thermal_radiation_downwards": VariableSpec(
        "downward_long_wave_radiation_flux_surface",
        "Surface downward long-wave radiation flux",
        "W m-2",
        "surface_downwelling_longwave_flux_in_air",
        cumulative_rate=True,
    ),
    "total_cloud_cover": VariableSpec(
        "total_cloud_cover_atmosphere",
        "Total cloud cover",
        "percent",
        "cloud_area_fraction",
    ),
    "total_precipitation": VariableSpec(
        "precipitation_surface",
        "Precipitation rate",
        "kg m-2 s-1",
        "precipitation_flux",
        cumulative_rate=True,
    ),
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


@dataclass(frozen=True)
class DevelopmentSource:
    source: Path
    payload: dict[str, object]


@dataclass(frozen=True)
class ZarrMeasurement:
    target: str
    complete: bool
    source_bytes: int
    zarr_bytes: int
    message_count: int
    transformation_seconds: float
    variables: list[str]
    members: list[int]
    leadtime_hours: list[int]


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


def build_development_zarr(
    sources: Sequence[DevelopmentSource],
    target: Path,
    *,
    require_complete: bool = True,
) -> ZarrMeasurement:
    started = time.monotonic()
    inventories = [inspect_grib(source.source, source.payload) for source in sources]
    assert inventories
    records = [record for inventory in inventories for record in inventory.records]
    members, leadtime_hours, canonical_names, complete = _zarr_axes(
        records, require_complete
    )

    init_times = {_init_time(source.payload) for source in sources}
    assert len(init_times) == 1
    grid_shapes = {tuple(inventory.grid_shape) for inventory in inventories}
    assert len(grid_shapes) == 1
    grid_shape_values = grid_shapes.pop()
    assert len(grid_shape_values) == 2
    grid_shape = (grid_shape_values[0], grid_shape_values[1])
    with rasterio.open(sources[0].source) as first_reader:
        transform = first_reader.transform
    root = _initialize_development_zarr(
        target,
        init_times.pop(),
        grid_shape,
        transform,
        members,
        leadtime_hours,
        canonical_names,
        records,
    )

    member_indexes = {member: index for index, member in enumerate(members)}
    lead_indexes = {lead: index for index, lead in enumerate(leadtime_hours)}
    for source, inventory in zip(sources, inventories, strict=True):
        _write_source_to_zarr(
            root,
            source,
            inventory,
            transform,
            member_indexes,
            lead_indexes,
        )

    for name in canonical_names:
        source_record = next(
            record for record in records if _variable_spec(record).name == name
        )
        spec = _variable_spec(source_record)
        if spec.cumulative_rate:
            _convert_cumulative_to_rate(
                _zarr_array(root, name),
                members,
                leadtime_hours,
                invalid_below_threshold_rate=(
                    RADIATION_INVALID_BELOW_THRESHOLD
                    if spec.units == "W m-2"
                    else PRECIPITATION_RATE_INVALID_BELOW_THRESHOLD
                ),
            )

    zarr.consolidate_metadata(target)
    return ZarrMeasurement(
        target=str(target),
        complete=complete,
        source_bytes=sum(source.source.stat().st_size for source in sources),
        zarr_bytes=sum(
            path.stat().st_size for path in target.rglob("*") if path.is_file()
        ),
        message_count=len(records),
        transformation_seconds=time.monotonic() - started,
        variables=canonical_names,
        members=members,
        leadtime_hours=leadtime_hours,
    )


def _zarr_axes(
    records: Sequence[InventoryRecord], require_complete: bool
) -> tuple[list[int], list[int], list[str], bool]:
    members = sorted({record.ensemble_member for record in records})
    leadtime_hours = sorted({_leadtime_end(record.leadtime_hour) for record in records})
    canonical_names = sorted({_variable_spec(record).name for record in records})
    coverage = {
        canonical_name: {
            (record.ensemble_member, _leadtime_end(record.leadtime_hour))
            for record in records
            if _variable_spec(record).name == canonical_name
        }
        for canonical_name in canonical_names
    }
    expected_coverage = set(product(members, leadtime_hours))
    complete = all(found == expected_coverage for found in coverage.values())
    if require_complete:
        incomplete = {
            name: sorted(expected_coverage - found)
            for name, found in coverage.items()
            if found != expected_coverage
        }
        assert not incomplete, f"Incomplete Zarr source inventory: {incomplete}"
    return members, leadtime_hours, canonical_names, complete


def _initialize_development_zarr(
    target: Path,
    init_time: np.datetime64,
    grid_shape: tuple[int, int],
    transform: rasterio.transform.Affine,
    members: Sequence[int],
    leadtime_hours: Sequence[int],
    canonical_names: Sequence[str],
    records: Sequence[InventoryRecord],
) -> zarr.Group:
    latitude_count, longitude_count = grid_shape
    latitude = transform.f + transform.e * (np.arange(latitude_count) + 0.5)
    longitude = transform.c + transform.a * (np.arange(longitude_count) + 0.5)
    root = zarr.open_group(target, mode="w")
    root.attrs.update(
        {
            "title": "ECMWF Extended Range ECDS development dataset",
            "source": "ECMWF Data Store s2s-forecasts",
            "spatial_resolution": f"{abs(transform.a):g} degrees",
        }
    )
    _create_coordinate(
        root,
        "init_time",
        np.array([init_time], dtype="datetime64[ns]"),
        ("init_time",),
        {"standard_name": "forecast_reference_time"},
    )
    _create_coordinate(
        root,
        "lead_time",
        np.array(leadtime_hours, dtype="timedelta64[h]"),
        ("lead_time",),
        {"standard_name": "forecast_period"},
    )
    _create_coordinate(
        root,
        "ensemble_member",
        np.array(members, dtype=np.int16),
        ("ensemble_member",),
        {"standard_name": "realization"},
    )
    _create_coordinate(
        root,
        "latitude",
        latitude.astype(np.float64),
        ("latitude",),
        {
            "standard_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        },
    )
    _create_coordinate(
        root,
        "longitude",
        longitude.astype(np.float64),
        ("longitude",),
        {
            "standard_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        },
    )
    valid_time = (
        np.array([init_time], dtype="datetime64[ns]")[:, None]
        + np.array(leadtime_hours, dtype="timedelta64[h]")[None, :]
    )
    _create_coordinate(
        root,
        "valid_time",
        valid_time,
        ("init_time", "lead_time"),
        {"standard_name": "time"},
    )
    _create_coordinate(
        root,
        "spatial_ref",
        np.array(0, dtype=np.int64),
        (),
        {
            "grid_mapping_name": "latitude_longitude",
            "earth_radius": 6_367_470.0,
        },
    )

    shape = (1, len(leadtime_hours), len(members), latitude_count, longitude_count)
    dimensions = (
        "init_time",
        "lead_time",
        "ensemble_member",
        "latitude",
        "longitude",
    )
    for name in canonical_names:
        source_record = next(
            record for record in records if _variable_spec(record).name == name
        )
        spec = _variable_spec(source_record)
        attributes = {
            "coordinates": "valid_time spatial_ref",
            "long_name": spec.long_name,
            "units": spec.units,
            "grid_mapping": "spatial_ref",
        }
        if spec.standard_name is not None:
            attributes["standard_name"] = spec.standard_name
        root.create_array(
            name,
            shape=shape,
            chunks=(
                1,
                1,
                min(ENSEMBLE_CHUNK_SIZE, len(members)),
                latitude_count,
                longitude_count,
            ),
            dtype=np.float32,
            fill_value=np.nan,
            attributes=attributes,  # ty: ignore[invalid-argument-type]
            dimension_names=dimensions,
        )
    return root


def _write_source_to_zarr(
    root: zarr.Group,
    source: DevelopmentSource,
    inventory: GribInventory,
    transform: rasterio.transform.Affine,
    member_indexes: Mapping[int, int],
    lead_indexes: Mapping[int, int],
) -> None:
    grouped_records: dict[tuple[str, int, int], list[tuple[int, InventoryRecord]]] = {}
    for band, record in enumerate(inventory.records, start=1):
        member_index = member_indexes[record.ensemble_member]
        key = (
            _variable_spec(record).name,
            lead_indexes[_leadtime_end(record.leadtime_hour)],
            member_index // ENSEMBLE_CHUNK_SIZE,
        )
        grouped_records.setdefault(key, []).append((band, record))

    with rasterio.open(source.source) as reader:
        assert reader.count == inventory.message_count
        assert reader.transform == transform
        for (name, lead_index, member_chunk), entries in grouped_records.items():
            member_start = member_chunk * ENSEMBLE_CHUNK_SIZE
            member_stop = min(member_start + ENSEMBLE_CHUNK_SIZE, len(member_indexes))
            target = _zarr_array(root, name)
            data = np.array(
                target[0, lead_index, member_start:member_stop, :, :], copy=True
            )
            bands = [band for band, _record in entries]
            values = reader.read(bands, out_dtype=np.float32, masked=True)
            decoded = np.asarray(values.filled(np.nan), dtype=np.float32)
            for source_index, (_band, record) in enumerate(entries):
                member_index = member_indexes[record.ensemble_member]
                data[member_index - member_start, :, :] = decoded[source_index]
            target[0, lead_index, member_start:member_stop, :, :] = data


def _zarr_array(root: zarr.Group, name: str) -> zarr.Array:
    result = root[name]
    assert isinstance(result, zarr.Array)
    return result


def _create_coordinate(
    root: zarr.Group,
    name: str,
    data: np.ndarray,
    dimension_names: tuple[str, ...],
    attributes: Mapping[str, object],
) -> None:
    root.create_array(
        name,
        data=data,
        chunks=data.shape or (),
        attributes=dict(attributes),  # ty: ignore[invalid-argument-type]
        dimension_names=dimension_names,
    )


def _convert_cumulative_to_rate(
    target: zarr.Array,
    members: Sequence[int],
    leadtime_hours: Sequence[int],
    *,
    invalid_below_threshold_rate: float,
) -> None:
    for member_start in range(0, len(members), ENSEMBLE_CHUNK_SIZE):
        member_stop = min(member_start + ENSEMBLE_CHUNK_SIZE, len(members))
        previous = np.zeros(
            (member_stop - member_start, *target.shape[-2:]), dtype=np.float32
        )
        previous_hour = 0
        for lead_index, leadtime_hour in enumerate(leadtime_hours):
            current = np.array(
                target[0, lead_index, member_start:member_stop, :, :], copy=True
            )
            duration_seconds = (leadtime_hour - previous_hour) * 3_600
            rate = (current - previous) / duration_seconds
            assert not np.any(rate < invalid_below_threshold_rate)
            rate[(rate < 0) & (rate >= invalid_below_threshold_rate)] = 0
            target[0, lead_index, member_start:member_stop, :, :] = rate
            previous = current
            previous_hour = leadtime_hour


def _variable_spec(record: InventoryRecord) -> VariableSpec:
    if record.level is None:
        return VARIABLE_SPECS[record.variable]
    level = record.level.replace("_", "")
    match record.variable:
        case "geopotential_height":
            return VariableSpec(
                f"geopotential_height_{level}",
                "Geopotential height",
                "m",
                "geopotential_height",
            )
        case "specific_humidity":
            return VariableSpec(
                f"specific_humidity_{level}",
                "Specific humidity",
                "1",
                "specific_humidity",
            )
        case "temperature":
            return VariableSpec(
                f"temperature_{level}",
                "Temperature",
                "degree_Celsius",
                "air_temperature",
            )
        case "u_component_of_wind":
            return VariableSpec(
                f"wind_u_{level}", "U wind component", "m s-1", "eastward_wind"
            )
        case "v_component_of_wind":
            return VariableSpec(
                f"wind_v_{level}", "V wind component", "m s-1", "northward_wind"
            )
        case _:
            raise AssertionError(record)


def _leadtime_end(leadtime: str) -> int:
    return int(leadtime.rsplit("_", maxsplit=1)[-1])


def _init_time(payload: Mapping[str, object]) -> np.datetime64:
    year = _strings(payload.get("year"))
    month = _strings(payload.get("month"))
    day = _strings(payload.get("day"))
    forecast_time = _strings(payload.get("time"))
    assert len(year) == len(month) == len(day) == len(forecast_time) == 1
    return np.datetime64(f"{year[0]}-{month[0]}-{day[0]}T{forecast_time[0]}")


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
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--source", type=Path, required=True)
    inventory.add_argument("--payload", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    build_zarr = commands.add_parser("build-zarr")
    build_zarr.add_argument("--manifest", type=Path, required=True)
    build_zarr.add_argument("--target", type=Path, required=True)
    build_zarr.add_argument("--measurement", type=Path, required=True)
    build_zarr.add_argument("--allow-incomplete", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    match arguments.command:
        case "inventory":
            payload = json.loads(arguments.payload.read_text())
            inventory = inspect_grib(arguments.source, payload)
            _write_json(arguments.output, inventory)
            log.info(
                "Validated %d GRIB messages in %s",
                inventory.message_count,
                arguments.source,
            )
        case "build-zarr":
            manifest = json.loads(arguments.manifest.read_text())
            sources = [
                DevelopmentSource(
                    Path(source["source"]),
                    json.loads(Path(source["payload"]).read_text()),
                )
                for source in manifest["sources"]
            ]
            measurement = build_development_zarr(
                sources,
                arguments.target,
                require_complete=not arguments.allow_incomplete,
            )
            _write_json(arguments.measurement, measurement)
            log.info(
                "Wrote %d messages to %s",
                measurement.message_count,
                arguments.target,
            )
        case _:
            raise AssertionError(arguments.command)


def _write_json(path: Path, value: GribInventory | ZarrMeasurement) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(asdict(value), indent=2, sort_keys=True))
    temporary_path.replace(path)


if __name__ == "__main__":
    main()
