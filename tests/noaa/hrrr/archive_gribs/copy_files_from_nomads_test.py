from collections.abc import Sequence
from datetime import timedelta
from unittest.mock import patch

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
        stats_logging_freq="1m",
        env_vars={},
    )


@pytest.fixture
def copies(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, list[str]]]:
    """Records (src, dst, file_names) per rclone invocation instead of running it."""
    recorded: list[tuple[str, str, list[str]]] = []
    destination_sizes: dict[str, int] = {}

    def fake_copy(
        src_path: str, dst_path: str, file_names: Sequence[str], **_kwargs: object
    ) -> None:
        recorded.append((src_path, dst_path, list(file_names)))
        destination_sizes.update(dict.fromkeys(file_names, 1))

    monkeypatch.setattr(module, "_rclone_copy", fake_copy)
    monkeypatch.setattr(
        module,
        "list_file_sizes",
        lambda _path, **_kwargs: destination_sizes.copy(),
    )
    return recorded


def test_paths_match_the_aws_archive_layout() -> None:
    """Cache keys must equal the AWS archive's keys so refs can be repointed by
    prefix swap alone."""
    assert source_dir(INIT).endswith("/hrrr.20260728/conus")
    assert destination_dir(":s3:cache/", INIT) == ":s3:cache/hrrr.20260728/conus"
    assert grib_file_name(INIT, 6) == "hrrr.t12z.wrfsfcf06.grib2"
    assert grib_file_name(INIT, 48) == "hrrr.t12z.wrfsfcf48.grib2"


def test_copies_grib_before_its_index(
    copies: list[tuple[str, str, list[str]]],
) -> None:
    """An index present in the cache must imply its data file is complete."""
    _copy(lead_hours=(0,))

    assert [names for _src, _dst, names in copies] == [
        ["hrrr.t12z.wrfsfcf00.grib2"],
        ["hrrr.t12z.wrfsfcf00.grib2.idx"],
    ]


def test_attempts_only_the_next_deterministic_pair(
    copies: list[tuple[str, str, list[str]]],
) -> None:
    _copy()
    assert [names for _src, _dst, names in copies] == [
        ["hrrr.t12z.wrfsfcf00.grib2"],
        ["hrrr.t12z.wrfsfcf00.grib2.idx"],
    ]


def test_copies_each_file_once_across_polls(
    copies: list[tuple[str, str, list[str]]],
) -> None:
    """Already-copied files must not be re-passed to rclone: a source stat per file
    per poll would exhaust the NOMADS request budget."""
    _copy(max_duration=timedelta(seconds=30))

    copied = [name for _src, _dst, names in copies for name in names]
    assert copied.count("hrrr.t12z.wrfsfcf00.grib2") == 1
    assert copied.count("hrrr.t12z.wrfsfcf01.grib2") == 1


def test_retries_a_grib_when_rclone_transfers_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grib = "hrrr.t12z.wrfsfcf00.grib2"
    copies: list[list[str]] = []
    destination_polls = iter(
        [
            {},
            {},
            {},
            {grib: 123},
            {grib: 123, f"{grib}.idx": 45},
        ]
    )
    monkeypatch.setattr(
        module,
        "_rclone_copy",
        lambda _src, _dst, names, **_kwargs: copies.append(list(names)),
    )
    monkeypatch.setattr(
        module,
        "list_file_sizes",
        lambda _path, **_kwargs: next(destination_polls),
    )

    _copy(lead_hours=(0,), max_duration=timedelta(seconds=30))

    assert copies == [[grib], [grib], [f"{grib}.idx"]]


def test_does_not_copy_index_for_a_zero_byte_grib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grib = "hrrr.t12z.wrfsfcf00.grib2"
    copies: list[list[str]] = []
    monkeypatch.setattr(
        module,
        "_rclone_copy",
        lambda _src, _dst, names, **_kwargs: copies.append(list(names)),
    )
    monkeypatch.setattr(
        module,
        "list_file_sizes",
        lambda _path, **_kwargs: {grib: 0},
    )

    _copy(lead_hours=(0,))

    assert copies == [[grib]]


def test_does_not_depend_on_a_nomads_directory_listing() -> None:
    assert not hasattr(module, "_published_file_names")


def test_rclone_serializes_nomads_requests() -> None:
    with patch.object(
        module, "run_command_with_concurrent_logging", return_value=0
    ) as run:
        module._rclone_copy(
            source_dir(INIT),
            destination_dir(":s3:cache/", INIT),
            [grib_file_name(INIT, 0)],
            stats_logging_freq="1m",
            env_vars={},
        )

    cmd = run.call_args.args[0]
    assert "--transfers=1" in cmd
    assert "--checkers=1" in cmd
    assert "--retries=1" in cmd
    assert "--low-level-retries=1" in cmd
    assert not any(arg.startswith("--multi-thread") for arg in cmd)


def test_rclone_failure_leaves_the_deterministic_path_pending() -> None:
    with patch.object(module, "run_command_with_concurrent_logging", return_value=1):
        module._rclone_copy(
            source_dir(INIT),
            destination_dir(":s3:cache/", INIT),
            [grib_file_name(INIT, 0)],
            stats_logging_freq="1m",
            env_vars={},
        )
