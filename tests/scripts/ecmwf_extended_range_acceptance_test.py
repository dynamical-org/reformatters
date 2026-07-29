from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Self

import numpy as np
import pytest
import xarray as xr
from affine import Affine

from scripts.ecmwf_extended_range_acceptance import (
    DevelopmentSource,
    GribInventory,
    InventoryRecord,
    build_development_zarr,
    inspect_grib,
)


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


def test_build_development_zarr_streams_complete_realization_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.grib"
    source.write_bytes(b"source")
    payload: dict[str, object] = {
        "year": ["2026"],
        "month": ["07"],
        "day": ["24"],
        "time": ["00:00"],
        "forecast_type": "perturbed_forecast",
        "number": ["1", "2"],
        "level_type": "single_level",
        "variable": ["total_precipitation"],
        "leadtime_hour": ["24", "48"],
    }
    records = [
        InventoryRecord(
            offset=(band - 1) * 24,
            message_size=24,
            variable="total_precipitation",
            level=None,
            ensemble_member=member,
            leadtime_hour=lead,
            units="ms-1",
            grib_level_type="specific height level above ground",
            grib_level_value=10,
            grid_shape=(2, 3),
        )
        for band, (lead, member) in enumerate(
            [("24", 1), ("24", 2), ("48", 1), ("48", 2)], start=1
        )
    ]
    inventory = GribInventory(
        source=str(source),
        valid=True,
        byte_count=96,
        message_count=4,
        grid_shape=[2, 3],
        variables=["total_precipitation"],
        levels=[],
        members=[1, 2],
        leadtimes=["24", "48"],
        records=records,
    )
    monkeypatch.setattr(
        "scripts.ecmwf_extended_range_acceptance.inspect_grib",
        lambda source, payload: inventory,
    )

    class FakeReader:
        count = 4
        transform = Affine(1.5, 0, -180.75, 0, -1.5, 90.75)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(
            self, bands: list[int], *, out_dtype: object, masked: bool
        ) -> np.ma.MaskedArray:
            assert out_dtype is np.float32
            assert masked
            return np.ma.array(
                np.stack(
                    [np.full((2, 3), band * 86_400, dtype=np.float32) for band in bands]
                )
            )

    monkeypatch.setattr(
        "scripts.ecmwf_extended_range_acceptance.rasterio.open",
        lambda source: FakeReader(),
    )
    target = tmp_path / "development.zarr"

    measurement = build_development_zarr([DevelopmentSource(source, payload)], target)
    rerun = build_development_zarr([DevelopmentSource(source, payload)], target)

    reopened = xr.open_zarr(target)
    assert measurement.complete
    assert rerun.complete
    assert {"valid_time", "spatial_ref"} <= set(reopened.coords)
    assert reopened.sizes == {
        "init_time": 1,
        "lead_time": 2,
        "ensemble_member": 2,
        "latitude": 2,
        "longitude": 3,
    }
    assert (
        reopened.precipitation_surface.sel(lead_time="24h", ensemble_member=2).values[
            0, 0, 0
        ]
        == 2
    )
    assert (
        reopened.precipitation_surface.sel(lead_time="48h", ensemble_member=1).values[
            0, 0, 0
        ]
        == 2
    )
    assert np.array_equal(reopened.valid_time, reopened.init_time + reopened.lead_time)
    assert reopened.longitude.values.tolist() == [-180.0, -178.5, -177.0]
    assert reopened.latitude.values.tolist() == [90.0, 88.5]
