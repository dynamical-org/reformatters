from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from zarr.storage import MemoryStore

from scripts.validation.manifest_scan import (
    ManifestScanResult,
    _file_availability,
    _probe_coord_for_var,
    _probe_jobs,
    _var_availability,
    _var_chunk_key,
    result_availability_series,
)
from scripts.validation.scan_common import evenly_spaced_subset


class _Coord:
    def __init__(
        self,
        position: pd.Timestamp,
        url: str,
        lead_time: pd.Timedelta | None = None,
        data_vars: list[Any] | None = None,
    ) -> None:
        self._position = position
        self._url = url
        self._lead_time = lead_time
        self.data_vars = data_vars

    def out_loc(self) -> dict[str, Any]:
        loc: dict[str, Any] = {"init_time": self._position}
        if self._lead_time is not None:
            loc["lead_time"] = self._lead_time
        return loc

    @property
    def append_dim_coord(self) -> pd.Timestamp:
        return self._position

    def get_url(self) -> str:
        return self._url


class _Job:
    """Stub VirtualRegionJob exposing only what the scan uses."""

    def __init__(
        self,
        coords: list[_Coord],
        missing: list[_Coord],
        data_vars: list[Any] | None = None,
        template_ds: xr.Dataset | None = None,
    ) -> None:
        self._coords = coords
        self._missing = missing
        self.data_vars = data_vars or []
        self.template_ds = template_ds

    def source_file_coords(self) -> list[_Coord]:
        return self._coords

    def filter_already_present(self, candidates: object, store: object) -> list[_Coord]:  # noqa: ARG002
        return self._missing


def _var(name: str, step_type: str = "instant") -> SimpleNamespace:
    return SimpleNamespace(
        name=name, path=name, attrs=SimpleNamespace(step_type=step_type)
    )


def test_evenly_spaced_subset() -> None:
    assert evenly_spaced_subset([1, 2, 3], 5) == [1, 2, 3]
    assert evenly_spaced_subset(list(range(10)), 3) == [0, 4, 9]
    assert evenly_spaced_subset(list(range(11)), 3) == [0, 5, 10]
    assert evenly_spaced_subset([1, 2, 3], 0) == [1, 2, 3]


def test_availability_counts_present_and_missing() -> None:
    p1 = pd.Timestamp("2024-01-01")
    p2 = pd.Timestamp("2024-01-02")
    coords = [
        _Coord(p1, "a"),
        _Coord(p1, "b"),
        _Coord(p2, "a"),
        _Coord(p2, "b"),
    ]
    job = _Job(coords, missing=[coords[3]])  # p2/b is missing

    probed = _probe_jobs([job], store=None)  # ty: ignore[invalid-argument-type]
    availability = _file_availability(probed, lead_limits={})

    assert availability[p1] == (2, 2)
    assert availability[p2] == (1, 2)


def test_availability_dedups_file_shared_across_jobs() -> None:
    p1 = pd.Timestamp("2024-01-01")
    # Two jobs (e.g. variable groups) both reference the same source file at p1.
    job_a = _Job([_Coord(p1, "shared")], missing=[])
    job_b_coord = _Coord(p1, "shared")
    job_b = _Job([job_b_coord], missing=[job_b_coord])  # one job sees it missing

    probed = _probe_jobs(
        [job_a, job_b],  # ty: ignore[invalid-argument-type]
        store=None,  # ty: ignore[invalid-argument-type]
    )
    availability = _file_availability(probed, lead_limits={})

    # Deduped to a single expected file; present because at least one job saw it present.
    assert availability[p1] == (1, 1)


def test_file_availability_trims_to_expected_forecast_length() -> None:
    p1 = pd.Timestamp("2024-01-01")  # 6-hour era
    p2 = pd.Timestamp("2024-01-02")  # 12-hour era
    coords = [
        _Coord(p1, "p1/f06", lead_time=pd.Timedelta(hours=6)),
        _Coord(p1, "p1/f12", lead_time=pd.Timedelta(hours=12)),
        _Coord(p2, "p2/f06", lead_time=pd.Timedelta(hours=6)),
        _Coord(p2, "p2/f12", lead_time=pd.Timedelta(hours=12)),
    ]
    # p1/f12 never existed upstream (beyond p1's expected length) and is missing.
    job = _Job(coords, missing=[coords[1]])
    probed = _probe_jobs([job], store=None)  # ty: ignore[invalid-argument-type]
    lead_limits = {p1: pd.Timedelta(hours=6), p2: pd.Timedelta(hours=12)}

    availability = _file_availability(probed, lead_limits)

    assert availability[p1] == (1, 1)  # f12 not expected, so p1 is complete
    assert availability[p2] == (2, 2)


def test_probe_coord_prefers_smallest_nonzero_lead() -> None:
    p = pd.Timestamp("2024-01-01")
    lead0 = _Coord(p, "f00", lead_time=pd.Timedelta(0))
    lead6 = _Coord(p, "f06", lead_time=pd.Timedelta(hours=6))
    lead12 = _Coord(p, "f12", lead_time=pd.Timedelta(hours=12))

    chosen = _probe_coord_for_var([lead0, lead12, lead6], _var("temperature_2m"))  # ty: ignore[invalid-argument-type]
    assert chosen is lead6


def test_probe_coord_accumulated_var_skips_lead_zero() -> None:
    p = pd.Timestamp("2024-01-01")
    lead0 = _Coord(p, "f00", lead_time=pd.Timedelta(0))

    # Only lead 0 present: an accumulated var has no ref there by design -> not probed.
    assert _probe_coord_for_var([lead0], _var("precipitation", "accum")) is None  # ty: ignore[invalid-argument-type]
    # An instant var probes fine at lead 0.
    assert _probe_coord_for_var([lead0], _var("temperature_2m")) is lead0  # ty: ignore[invalid-argument-type]


def test_probe_coord_respects_coord_data_vars() -> None:
    p = pd.Timestamp("2024-01-01")
    temperature = _var("temperature_2m")
    sfc = _Coord(p, "sfc", lead_time=pd.Timedelta(hours=6), data_vars=[temperature])
    prs = _Coord(p, "prs", lead_time=pd.Timedelta(hours=1), data_vars=[_var("other")])

    # prs has the smaller lead but does not carry temperature.
    assert _probe_coord_for_var([sfc, prs], temperature) is sfc  # ty: ignore[invalid-argument-type]


def _template() -> xr.Dataset:
    init_time = pd.date_range("2024-01-01", periods=3, freq="D")
    lead_time = pd.to_timedelta([0, 6, 12], unit="h")
    level = np.array([1000.0, 500.0, 100.0])
    ds = xr.Dataset(
        {
            "temperature_2m": (
                ("init_time", "lead_time"),
                np.zeros((3, 3), dtype=np.float32),
            ),
            "temperature": (
                ("init_time", "lead_time", "pressure_level"),
                np.zeros((3, 3, 3), dtype=np.float32),
            ),
        },
        coords={
            "init_time": init_time,
            "lead_time": lead_time,
            "pressure_level": level,
        },
    )
    ds["temperature_2m"].encoding["chunks"] = (1, 1)
    ds["temperature"].encoding["chunks"] = (1, 1, 1)
    return ds


def test_var_chunk_key_uses_middle_chunk_for_unlabeled_dims() -> None:
    template = _template()
    store = MemoryStore()
    array = zarr.create_array(
        store, name="temperature", shape=(3, 3, 3), chunks=(1, 1, 1), dtype="f4"
    )

    out_loc = {
        "init_time": pd.Timestamp("2024-01-02"),
        "lead_time": pd.Timedelta(hours=6),
    }
    key = _var_chunk_key(
        template,  # ty: ignore[invalid-argument-type]
        array.metadata,  # ty: ignore[invalid-argument-type]
        _var("temperature"),  # ty: ignore[invalid-argument-type]
        out_loc,
    )
    # pressure_level is not in out_loc -> middle of its 3 chunks.
    assert key == "temperature/c/1/1/1"


def test_var_availability_probes_written_chunks() -> None:
    template = _template()
    store = MemoryStore()
    root = zarr.open_group(store, mode="w")
    array = root.create_array("temperature_2m", shape=(3, 3), chunks=(1, 1), dtype="f4")
    # Chunks exist at (0, 1) only: position 0 probed present, position 1 absent.
    array[0, 1] = 1.0

    p0, p1 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    coord_p0 = _Coord(p0, "p0/f06", lead_time=pd.Timedelta(hours=6))
    coord_p1 = _Coord(p1, "p1/f06", lead_time=pd.Timedelta(hours=6))
    coord_p2_missing = _Coord(
        pd.Timestamp("2024-01-03"), "p2/f06", lead_time=pd.Timedelta(hours=6)
    )
    job = _Job(
        [coord_p0, coord_p1, coord_p2_missing],
        missing=[coord_p2_missing],
        data_vars=[_var("temperature_2m")],
        template_ds=template,
    )
    probed = [(job, [(coord_p0, True), (coord_p1, True), (coord_p2_missing, False)])]

    availability = _var_availability(probed, store)  # ty: ignore[invalid-argument-type]

    # Position 2 has no present source file -> not probed (absent from the mapping).
    assert availability == {"temperature_2m": {p0: True, p1: False}}


def test_result_availability_series_marks_unprobed_positions_nan() -> None:
    p0, p1, p2 = pd.date_range("2024-01-01", periods=3, freq="D")
    result = ManifestScanResult(
        append_dim="init_time",
        file_availability={p0: (2, 2), p1: (2, 2), p2: (0, 2)},
        var_availability={"temperature_2m": {p0: True, p1: False}},
    )

    series = result_availability_series(result)["temperature_2m"]

    np.testing.assert_array_equal(
        series.positions, np.array([p0, p1, p2], dtype="datetime64[ns]")
    )
    np.testing.assert_array_equal(series.fraction[:2], [1.0, 0.0])
    assert np.isnan(series.fraction[2])
