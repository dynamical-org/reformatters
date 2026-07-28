from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scripts.ecmwf_extended_range_acceptance import inspect_grib


@dataclass
class FakeMetadata:
    message_size: int
    var_abbrev: str
    level_type: str
    level_value: float | None
    perturbation_number: int
    reference_date: datetime
    forecast_date: datetime
    forecast_date_end: datetime | None
    units: str
    grid_shape: tuple[int, int] = (121, 240)


def metadata(
    *,
    variable: str,
    member: int = 0,
    lead: int = 24,
    level: float | None = None,
    interval_start: int | None = None,
) -> FakeMetadata:
    reference = datetime(2026, 7, 24)
    level_type = (
        "isobaric surface"
        if level is not None and level != 2
        else "specific height level above ground"
    )
    return FakeMetadata(
        message_size=24,
        var_abbrev=variable,
        level_type=level_type,
        level_value=level,
        perturbation_number=member,
        reference_date=reference,
        forecast_date=reference + timedelta(hours=lead)
        if interval_start is None
        else reference + timedelta(hours=interval_start),
        forecast_date_end=reference + timedelta(hours=lead)
        if interval_start is not None
        else None,
        units="K",
    )


def write_messages(path: Path, count: int) -> None:
    message = b"GRIB\x00\x00\x00\x02" + (24).to_bytes(8) + b"data7777"
    path.write_bytes(message * count)


def test_inspect_grib_validates_exact_surface_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "surface.grib"
    write_messages(source, 4)
    messages = iter(
        [
            metadata(variable="UGRD", member=0, lead=24),
            metadata(variable="VGRD", member=0, lead=24),
            metadata(variable="UGRD", member=0, lead=48),
            metadata(variable="VGRD", member=0, lead=48),
        ]
    )
    monkeypatch.setattr(
        "scripts.ecmwf_extended_range_acceptance.parse_grib_message_metadata",
        lambda data, offset: next(messages),
    )

    inventory = inspect_grib(
        source,
        {
            "forecast_type": "control_forecast",
            "level_type": "single_level",
            "variable": [
                "10_m_u_component_of_wind",
                "10_m_v_component_of_wind",
            ],
            "leadtime_hour": ["24", "48"],
        },
    )

    assert inventory.valid
    assert inventory.message_count == 4
    assert inventory.members == [0]
    assert inventory.leadtimes == ["24", "48"]
    assert inventory.variables == [
        "10_m_u_component_of_wind",
        "10_m_v_component_of_wind",
    ]
    assert inventory.grid_shape == [121, 240]


def test_inspect_grib_validates_pressure_levels_and_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pressure.grib"
    write_messages(source, 4)
    messages = iter(
        [
            metadata(variable="TMP", member=1, level=500),
            metadata(variable="TMP", member=1, level=850),
            metadata(variable="TMP", member=2, level=500),
            metadata(variable="TMP", member=2, level=850),
        ]
    )
    monkeypatch.setattr(
        "scripts.ecmwf_extended_range_acceptance.parse_grib_message_metadata",
        lambda data, offset: next(messages),
    )

    inventory = inspect_grib(
        source,
        {
            "forecast_type": "perturbed_forecast",
            "number": ["1", "2"],
            "level_type": "pressure",
            "level_value": ["500_hpa", "850_hpa"],
            "variable": ["temperature"],
            "leadtime_hour": ["24"],
        },
    )

    assert inventory.valid
    assert inventory.members == [1, 2]
    assert inventory.levels == ["500_hpa", "850_hpa"]


def test_inspect_grib_matches_interval_leadtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "daily.grib"
    write_messages(source, 1)
    monkeypatch.setattr(
        "scripts.ecmwf_extended_range_acceptance.parse_grib_message_metadata",
        lambda data, offset: metadata(
            variable="TMP", lead=24, interval_start=0, level=2
        ),
    )

    inventory = inspect_grib(
        source,
        {
            "forecast_type": "control_forecast",
            "level_type": "single_level",
            "variable": ["2_m_temperature"],
            "leadtime_hour": ["0_24"],
        },
    )

    assert inventory.valid
    assert inventory.leadtimes == ["0_24"]


def test_inspect_grib_rejects_missing_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "incomplete.grib"
    write_messages(source, 1)
    monkeypatch.setattr(
        "scripts.ecmwf_extended_range_acceptance.parse_grib_message_metadata",
        lambda data, offset: metadata(variable="UGRD"),
    )

    with pytest.raises(AssertionError, match="Missing inventory"):
        inspect_grib(
            source,
            {
                "forecast_type": "control_forecast",
                "level_type": "single_level",
                "variable": [
                    "10_m_u_component_of_wind",
                    "10_m_v_component_of_wind",
                ],
                "leadtime_hour": ["24"],
            },
        )
