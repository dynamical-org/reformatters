import pandas as pd

from scripts.validation.manifest_scan import _availability_by_position
from scripts.validation.scan_common import evenly_spaced_subset


class _Coord:
    def __init__(self, position: pd.Timestamp, url: str) -> None:
        self._position = position
        self._url = url

    def out_loc(self) -> dict[str, pd.Timestamp]:
        return {"init_time": self._position}

    def get_url(self) -> str:
        return self._url


class _Job:
    """Stub VirtualRegionJob exposing only what _availability_by_position uses."""

    def __init__(self, coords: list[_Coord], missing: list[_Coord]) -> None:
        self._coords = coords
        self._missing = missing

    def source_file_coords(self) -> list[_Coord]:
        return self._coords

    def filter_already_present(self, candidates: object, store: object) -> list[_Coord]:  # noqa: ARG002
        return self._missing


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

    availability = _availability_by_position([job], store=None, append_dim="init_time")  # ty: ignore[invalid-argument-type]

    assert availability[p1] == (2, 2)
    assert availability[p2] == (1, 2)


def test_availability_dedups_file_shared_across_jobs() -> None:
    p1 = pd.Timestamp("2024-01-01")
    # Two jobs (e.g. variable groups) both reference the same source file at p1.
    job_a = _Job([_Coord(p1, "shared")], missing=[])
    job_b_coord = _Coord(p1, "shared")
    job_b = _Job([job_b_coord], missing=[job_b_coord])  # one job sees it missing

    availability = _availability_by_position(
        [job_a, job_b],
        store=None,  # ty: ignore[invalid-argument-type]
        append_dim="init_time",
    )

    # Deduped to a single expected file; present because at least one job saw it present.
    assert availability[p1] == (1, 1)
