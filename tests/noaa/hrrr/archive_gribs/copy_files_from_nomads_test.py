from collections.abc import Sequence
from datetime import timedelta

import pandas as pd
import pytest

from reformatters.noaa.hrrr.archive_gribs import copy_files_from_nomads as module
from reformatters.noaa.hrrr.archive_gribs.copy_files_from_nomads import (
    copy_files_from_nomads,
    destination_dir,
    grib_file_name,
    source_dir,
)

INIT = pd.Timestamp("2026-07-28T12:00")


def _copy(
    dst_root: str = ":s3:cache/",
    lead_hours: tuple[int, ...] = (0, 1),
    max_duration: timedelta = timedelta(seconds=0),
) -> None:
    copy_files_from_nomads(
        dst_root_path=dst_root,
        init_times=[INIT],
        lead_hours=lead_hours,
        max_duration=max_duration,
        poll_interval=timedelta(seconds=0),
        transfer_parallelism=4,
        checkers=8,
        stats_logging_freq="1m",
        env_vars={},
    )


@pytest.fixture
def copies(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, list[str]]]:
    """Records (src, dst, file_names) per rclone invocation instead of running it."""
    recorded: list[tuple[str, str, list[str]]] = []

    def fake_copy(
        src_path: str, dst_path: str, file_names: Sequence[str], **_kwargs: object
    ) -> None:
        recorded.append((src_path, dst_path, list(file_names)))

    monkeypatch.setattr(module, "_rclone_copy", fake_copy)
    return recorded


def test_paths_match_the_aws_archive_layout() -> None:
    """Cache keys must equal the AWS archive's keys so refs can be repointed by
    prefix swap alone."""
    assert source_dir(INIT).endswith("/hrrr.20260728/conus")
    assert destination_dir(":s3:cache/", INIT) == ":s3:cache/hrrr.20260728/conus"
    assert grib_file_name(INIT, 6) == "hrrr.t12z.wrfsfcf06.grib2"
    assert grib_file_name(INIT, 48) == "hrrr.t12z.wrfsfcf48.grib2"


def test_copies_grib_before_its_index(
    copies: list[tuple[str, str, list[str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index present in the cache must imply its data file is complete."""
    published = {
        "hrrr.t12z.wrfsfcf00.grib2",
        "hrrr.t12z.wrfsfcf00.grib2.idx",
        "hrrr.t12z.wrfsfcf01.grib2",
        "hrrr.t12z.wrfsfcf01.grib2.idx",
    }
    monkeypatch.setattr(module, "_published_file_names", lambda _init: published)
    _copy()

    assert [names for _src, _dst, names in copies] == [
        ["hrrr.t12z.wrfsfcf00.grib2", "hrrr.t12z.wrfsfcf01.grib2"],
        ["hrrr.t12z.wrfsfcf00.grib2.idx", "hrrr.t12z.wrfsfcf01.grib2.idx"],
    ]


def test_skips_a_grib_whose_index_has_not_published(
    copies: list[tuple[str, str, list[str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        module,
        "_published_file_names",
        lambda _init: {"hrrr.t12z.wrfsfcf00.grib2", "hrrr.t12z.wrfsfcf01.grib2"},
    )
    _copy()
    assert copies == []


def test_copies_each_file_once_across_polls(
    copies: list[tuple[str, str, list[str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Already-copied files must not be re-passed to rclone: a source stat per file
    per poll would exhaust the NOMADS request budget."""
    polls = iter(
        [
            {"hrrr.t12z.wrfsfcf00.grib2", "hrrr.t12z.wrfsfcf00.grib2.idx"},
            {
                "hrrr.t12z.wrfsfcf00.grib2",
                "hrrr.t12z.wrfsfcf00.grib2.idx",
                "hrrr.t12z.wrfsfcf01.grib2",
                "hrrr.t12z.wrfsfcf01.grib2.idx",
            },
        ]
    )
    monkeypatch.setattr(module, "_published_file_names", lambda _init: next(polls))
    _copy(max_duration=timedelta(seconds=30))

    copied = [name for _src, _dst, names in copies for name in names]
    assert copied.count("hrrr.t12z.wrfsfcf00.grib2") == 1
    assert copied.count("hrrr.t12z.wrfsfcf01.grib2") == 1


def test_gives_up_on_a_file_that_never_publishes(
    copies: list[tuple[str, str, list[str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "_published_file_names", lambda _init: set())
    _copy()  # max_duration=0, so one sweep then return
    assert copies == []


def test_parses_only_grib_and_index_hrefs() -> None:
    html = """
    <a href="hrrr.t12z.wrfsfcf00.grib2">x</a>
    <a href="hrrr.t12z.wrfsfcf00.grib2.idx">x</a>
    <a href="hrrr.t12z.wrfprsf00.grib2">x</a>
    <a href="bufrsnd.t12z/">dir</a>
    <a href="/pub/data/nccf/com/hrrr/prod/">parent</a>
    """
    assert set(module._HREF_RE.findall(html)) == {
        "hrrr.t12z.wrfsfcf00.grib2",
        "hrrr.t12z.wrfsfcf00.grib2.idx",
        "hrrr.t12z.wrfprsf00.grib2",
    }
