import os
import time
from pathlib import Path
from unittest.mock import Mock

import obstore.store
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.validation import ValidationContext
from reformatters.noaa.hrrr import nomads_cache
from reformatters.noaa.hrrr.nomads_cache import (
    REPOINTED_MARKER_SUFFIX,
    CheckNomadsCacheRepointed,
    cache_key,
    list_cache,
    mark_repointed,
    parse_cache_key,
    unrepointed_data_files,
)
from reformatters.noaa.hrrr.region_job import NoaaHrrrSourceFileCoord

KEY = "hrrr.20260907/conus/hrrr.t19z.wrfsfcf01.grib2"


def test_cache_key_round_trips_through_parse() -> None:
    coord = NoaaHrrrSourceFileCoord(
        init_time=pd.Timestamp("2026-09-07T19:00"),
        lead_time=pd.Timedelta("1h"),
        domain="conus",
        file_type="sfc",
        data_vars=[],
    )
    assert cache_key(coord) == KEY
    assert parse_cache_key(KEY) == (
        pd.Timestamp("2026-09-07T19:00"),
        pd.Timedelta("1h"),
        "sfc",
    )
    assert parse_cache_key(KEY + ".idx") is None
    assert parse_cache_key(KEY + REPOINTED_MARKER_SUFFIX) is None


def _write(cache: Path, key: str, age_hours: float = 0) -> None:
    path = cache / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    stamp = time.time() - age_hours * 3600
    os.utime(path, (stamp, stamp))


def test_unrepointed_data_files_need_an_index_and_no_marker(tmp_path: Path) -> None:
    _write(tmp_path, KEY)
    _write(tmp_path, KEY + ".idx")
    no_index = KEY.replace("f01", "f02")
    _write(tmp_path, no_index)
    done = KEY.replace("f01", "f03")
    _write(tmp_path, done)
    _write(tmp_path, done + ".idx")
    _write(tmp_path, done + REPOINTED_MARKER_SUFFIX)

    listing = list_cache(obstore.store.LocalStore(tmp_path))
    assert unrepointed_data_files(listing) == [KEY]


def test_mark_repointed_writes_the_marker(tmp_path: Path) -> None:
    store = obstore.store.LocalStore(tmp_path)
    _write(tmp_path, KEY)
    mark_repointed(KEY, store)
    assert (tmp_path / (KEY + REPOINTED_MARKER_SUFFIX)).exists()


def _check(tmp_path: Path) -> CheckNomadsCacheRepointed:
    class LocalCheck(CheckNomadsCacheRepointed):
        def cache_store(self) -> obstore.store.ObjectStore:
            return obstore.store.LocalStore(tmp_path)

    return LocalCheck()


def _context() -> ValidationContext:
    return ValidationContext(store=Mock(), ds=xr.Dataset(), append_dim="init_time")


def test_check_passes_when_every_old_file_is_repointed(tmp_path: Path) -> None:
    _write(tmp_path, KEY, age_hours=5)
    _write(tmp_path, KEY + ".idx", age_hours=5)
    _write(tmp_path, KEY + REPOINTED_MARKER_SUFFIX, age_hours=4)
    fresh = KEY.replace("f01", "f02")
    _write(tmp_path, fresh)
    _write(tmp_path, fresh + ".idx")
    result = _check(tmp_path).check(_context())
    assert result.passed, result.message


def test_check_fails_on_an_old_file_nodd_never_published(tmp_path: Path) -> None:
    _write(tmp_path, KEY, age_hours=5)
    _write(tmp_path, KEY + ".idx", age_hours=5)
    result = _check(tmp_path).check(_context())
    assert not result.passed
    assert KEY in result.message


def test_mark_repointed_tags_before_it_writes_the_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_tag(keys: list[str]) -> None:
        raise RuntimeError("tagging denied")

    monkeypatch.setattr(nomads_cache, "_tag_repointed", failing_tag)
    store = obstore.store.S3Store(
        "unused-bucket", region="us-west-2", skip_signature=True
    )
    with pytest.raises(RuntimeError, match="tagging denied"):
        mark_repointed(KEY, store)
    assert not list(tmp_path.iterdir())
