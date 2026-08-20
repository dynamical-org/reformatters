from collections.abc import Mapping

import obstore
import pytest

from reformatters.common import virtual_source_listing
from reformatters.common.region_job import CoordinateValue, SourceFileCoord
from reformatters.common.types import Dim
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,
)

_PREFIX = "s3://bucket/"


class _Coord(SourceFileCoord):
    key: str

    def get_url(self) -> str:
        return f"{_PREFIX}dir/{self.key}.grib2"

    def get_index_url(self) -> str:
        return f"{_PREFIX}dir/{self.key}.index"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {}


def _fake_listing(monkeypatch: pytest.MonkeyPatch, listed: dict[str, int]) -> list[str]:
    seen_prefixes: list[str] = []

    def fake(store: obstore.store.ObjectStore, prefixes: list[str]) -> dict[str, int]:
        seen_prefixes.extend(prefixes)
        return listed

    monkeypatch.setattr(virtual_source_listing, "_list_objects", fake)
    return seen_prefixes


_LISTING = {
    "dir/both.grib2": 9000,
    "dir/both.index": 200,
    "dir/index_only.index": 150,
    "dir/data_only.grib2": 9000,
}


def test_require_index_needs_both_data_and_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    both, index_only, data_only = (
        _Coord(key="both"),
        _Coord(key="index_only"),
        _Coord(key="data_only"),
    )
    seen = _fake_listing(monkeypatch, _LISTING)

    result = discover_available_by_obstore_listing(
        [both, index_only, data_only],
        store=obstore.store.MemoryStore(),  # unused; _list_objects is faked
        location_prefix=_PREFIX,
        require_index=True,
    )

    # Only the file with both objects listed; same coord object, paired with its size.
    assert len(result) == 1
    (coord, size) = result[0]
    assert coord is both
    assert size == 9000
    assert seen == ["dir/"]  # the files' shared directory prefix


def test_without_require_index_data_object_suffices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    both, index_only, data_only = (
        _Coord(key="both"),
        _Coord(key="index_only"),
        _Coord(key="data_only"),
    )
    _fake_listing(monkeypatch, _LISTING)

    result = discover_available_by_obstore_listing(
        [both, index_only, data_only],
        store=obstore.store.MemoryStore(),
        location_prefix=_PREFIX,
        require_index=False,
    )

    # index_only has no data object listed; the other two do.
    assert {coord.key for coord, _ in result} == {"both", "data_only"}


def test_empty_when_nothing_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_listing(monkeypatch, {})
    assert (
        discover_available_by_obstore_listing(
            [_Coord(key="a")],
            store=obstore.store.MemoryStore(),
            location_prefix=_PREFIX,
            require_index=True,
        )
        == []
    )


class _TwoDirCoord(SourceFileCoord):
    directory: str
    key: str

    def get_url(self) -> str:
        return f"{_PREFIX}{self.directory}/{self.key}.grib2"

    def get_index_url(self) -> str:
        return f"{_PREFIX}{self.directory}/{self.key}.index"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {}


def test_coords_spanning_two_directories(monkeypatch: pytest.MonkeyPatch) -> None:
    # Coords in different directories must each get their directory listed; a
    # single-prefix listing would only ever discover one directory's files.
    coord_a = _TwoDirCoord(directory="dir_a", key="a")
    coord_b = _TwoDirCoord(directory="dir_b", key="b")
    seen = _fake_listing(
        monkeypatch,
        {
            "dir_a/a.grib2": 100,
            "dir_a/a.index": 10,
            "dir_b/b.grib2": 200,
            "dir_b/b.index": 20,
        },
    )

    result = discover_available_by_obstore_listing(
        [coord_a, coord_b],
        store=obstore.store.MemoryStore(),
        location_prefix=_PREFIX,
        require_index=True,
    )

    assert seen == ["dir_a/", "dir_b/"]
    assert {(coord.directory, size) for coord, size in result} == {
        ("dir_a", 100),
        ("dir_b", 200),
    }


def test_discovery_against_real_memory_store() -> None:
    # No fakes: _list_objects and the discovery filters against a real obstore
    # backend, across two prefixes.
    store = obstore.store.MemoryStore()
    obstore.put(store, "dir/both.grib2", b"x" * 9000)
    obstore.put(store, "dir/both.index", b"i" * 200)
    obstore.put(store, "dir/data_only.grib2", b"y" * 50)
    obstore.put(store, "other/extra.grib2", b"z" * 7)

    assert virtual_source_listing._list_objects(store, ["dir/", "other/"]) == {
        "dir/both.grib2": 9000,
        "dir/both.index": 200,
        "dir/data_only.grib2": 50,
        "other/extra.grib2": 7,
    }

    both, data_only, absent = (
        _Coord(key="both"),
        _Coord(key="data_only"),
        _Coord(key="absent"),
    )
    with_index = discover_available_by_obstore_listing(
        [both, data_only, absent],
        store=store,
        location_prefix=_PREFIX,
        require_index=True,
    )
    assert [(coord.key, size) for coord, size in with_index] == [("both", 9000)]

    without_index = discover_available_by_obstore_listing(
        [both, data_only, absent],
        store=store,
        location_prefix=_PREFIX,
        require_index=False,
    )
    assert {(coord.key, size) for coord, size in without_index} == {
        ("both", 9000),
        ("data_only", 50),
    }
