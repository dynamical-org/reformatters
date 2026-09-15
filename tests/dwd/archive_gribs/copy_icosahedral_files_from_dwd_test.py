from collections.abc import Sequence
from datetime import timedelta
from pathlib import PurePosixPath
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from reformatters.dwd.archive_gribs.copy_icosahedral_files_from_dwd import (
    copy_icosahedral_files_from_dwd_https,
    icosahedral_dst_path,
    icosahedral_include_filters,
    incomplete_runs,
)

MODULE = "reformatters.dwd.archive_gribs.copy_icosahedral_files_from_dwd"


@pytest.mark.parametrize(
    ("src_path", "expected_dst_path"),
    [
        (
            "T_2M/r/2026-09-15T00:00/s/PT000H00M.grib2",
            "2026-09-15T00/T_2M/PT000H00M.grib2",
        ),
        (
            "T/lvt1/150/lv1/1/r/2026-09-15T03:00/s/PT051H00M.grib2",
            "2026-09-15T03/T/lvt1/150/lv1/1/PT051H00M.grib2",
        ),
        (
            "T/lvt1/100/lv1/85000/r/2026-09-15T12:00/s/PT120H00M.grib2",
            "2026-09-15T12/T/lvt1/100/lv1/85000/PT120H00M.grib2",
        ),
        (
            "T_SO/lvt1/106/lv1/0.005/r/2026-09-15T21:00/s/PT002H00M.grib2",
            "2026-09-15T21/T_SO/lvt1/106/lv1/0.005/PT002H00M.grib2",
        ),
    ],
)
def test_icosahedral_dst_path(src_path: str, expected_dst_path: str) -> None:
    assert icosahedral_dst_path(PurePosixPath(src_path)) == PurePosixPath(
        expected_dst_path
    )


def test_icosahedral_dst_paths_keep_every_source_distinct() -> None:
    src_paths = [
        PurePosixPath(p)
        for p in (
            "T/r/2026-09-15T00:00/s/PT000H00M.grib2",
            "T/lvt1/150/lv1/1/r/2026-09-15T00:00/s/PT000H00M.grib2",
            "T/lvt1/100/lv1/1/r/2026-09-15T00:00/s/PT000H00M.grib2",
            "T/lvt1/150/lv1/10/r/2026-09-15T00:00/s/PT000H00M.grib2",
            "T/lvt1/150/lv1/1/r/2026-09-16T00:00/s/PT000H00M.grib2",
            "T/lvt1/150/lv1/1/r/2026-09-15T00:00/s/PT001H00M.grib2",
            "CLC/lvt1/150/lv1/1/r/2026-09-15T00:00/s/PT000H00M.grib2",
        )
    ]
    assert len({icosahedral_dst_path(p) for p in src_paths}) == len(src_paths)


@pytest.mark.parametrize(
    "src_path",
    [
        # Old regular lat/lon scheme
        "alb_rad/icon-eu_europe_regular-lat-lon_single-level_2026011400_000_ALB_RAD.grib2.bz2",
        "T_2M/r/2026-09-15T00:00/s/PT000H00M.grib2.bz2",
        "T_2M/r/2026-09-15T00:30/s/PT000H00M.grib2",
        "T_2M/r/2026-09-15T00:00/PT000H00M.grib2",
        "T/lvt2/150/lv1/1/r/2026-09-15T00:00/s/PT000H00M.grib2",
        "T/lvt1/150/r/2026-09-15T00:00/s/PT000H00M.grib2",
    ],
)
def test_icosahedral_dst_path_rejects_unexpected_paths(src_path: str) -> None:
    with pytest.raises(ValueError, match="Unexpected"):
        icosahedral_dst_path(PurePosixPath(src_path))


def test_icosahedral_include_filters_select_plain_grib2_for_runs_and_levels() -> None:
    assert icosahedral_include_filters(
        nwp_init_hours=[0, 12], level_types=[100, 106], params=[]
    ) == [
        "--include=/*/r/*T{00,12}:00/s/*.grib2",
        "--include=/*/lvt1/100/lv1/*/r/*T{00,12}:00/s/*.grib2",
        "--include=/*/lvt1/106/lv1/*/r/*T{00,12}:00/s/*.grib2",
    ]


def test_icosahedral_include_filters_single_level_only_for_selected_params() -> None:
    assert icosahedral_include_filters(
        nwp_init_hours=[3], level_types=[], params=["T_2M", "CLAT"]
    ) == ["--include=/{T_2M,CLAT}/r/*T{03}:00/s/*.grib2"]


NOW = pd.Timestamp("2026-09-15T12:00Z")
SRC_ROOT = ":http:/weather/nwp/v1/m/icon-eu/p/"
URL_ROOT = "https://opendata.dwd.de/weather/nwp/v1/m/icon-eu/p"


def _fake_list_files(files_by_path: dict[str, list[str]]) -> MagicMock:
    def list_files(path: str, **_kwargs: object) -> list[PurePosixPath]:
        return [PurePosixPath(p) for p in files_by_path[path]]

    return MagicMock(side_effect=list_files)


def _copy(
    mock_list_files: MagicMock,
    nwp_init_hours: Sequence[int] = (0, 3),
    level_types: Sequence[int] = (100,),
    time_budget: timedelta = timedelta(hours=1),
    now: pd.Timestamp = NOW,
) -> None:
    with (
        patch(f"{MODULE}.list_files", mock_list_files),
        patch("pandas.Timestamp.now", return_value=now),
    ):
        copy_icosahedral_files_from_dwd_https(
            dst_root_path=":s3:bucket/icosahedral",
            nwp_init_hours=nwp_init_hours,
            level_types=level_types,
            params=[],
            time_budget=time_budget,
            transfer_parallelism=8,
            checkers=4,
            stats_logging_freq="1m",
        )


# A main run (00) and an intermediate run (03), which are not compared with each other.
TWO_RUNS_ON_DWD = [
    "T_2M/r/2026-09-15T00:00/s/PT000H00M.grib2",
    "T_2M/r/2026-09-15T00:00/s/PT001H00M.grib2",
    "T/lvt1/100/lv1/85000/r/2026-09-15T00:00/s/PT000H00M.grib2",
    "T_2M/r/2026-09-15T03:00/s/PT000H00M.grib2",
    "T/lvt1/100/lv1/85000/r/2026-09-15T03:00/s/PT000H00M.grib2",
]


@patch(f"{MODULE}.copy_urls")
def test_copy_icosahedral_files_copies_missing_files_one_run_at_a_time_oldest_first(
    mock_copy_urls: MagicMock,
) -> None:
    mock_list_files = _fake_list_files(
        {
            SRC_ROOT: TWO_RUNS_ON_DWD,
            ":s3:bucket/icosahedral/2026-09-15T00/": ["T_2M/PT000H00M.grib2"],
            ":s3:bucket/icosahedral/2026-09-15T03/": [],
        }
    )
    _copy(mock_list_files)

    src_listing = mock_list_files.call_args_list[0]
    assert src_listing.kwargs["path"] == SRC_ROOT
    rclone_args = src_listing.kwargs["rclone_args"]
    assert "--http-url=https://opendata.dwd.de" in rclone_args
    assert "--http-no-head" in rclone_args
    assert "--include=/*/lvt1/100/lv1/*/r/*T{00,03}:00/s/*.grib2" in rclone_args
    # DWD's index pages carry no modification times, so an age filter would match everything.
    assert not any(arg.startswith("--min-age") for arg in rclone_args)

    assert [c.kwargs["dst_root_path"] for c in mock_copy_urls.call_args_list] == [
        ":s3:bucket/icosahedral/"
    ] * 2
    assert [
        c.kwargs["sources_and_dst_paths"] for c in mock_copy_urls.call_args_list
    ] == [
        [
            (
                f"{URL_ROOT}/T/lvt1/100/lv1/85000/r/2026-09-15T00%3A00/s/PT000H00M.grib2",
                PurePosixPath("2026-09-15T00/T/lvt1/100/lv1/85000/PT000H00M.grib2"),
            ),
            (
                f"{URL_ROOT}/T_2M/r/2026-09-15T00%3A00/s/PT001H00M.grib2",
                PurePosixPath("2026-09-15T00/T_2M/PT001H00M.grib2"),
            ),
        ],
        [
            (
                f"{URL_ROOT}/T/lvt1/100/lv1/85000/r/2026-09-15T03%3A00/s/PT000H00M.grib2",
                PurePosixPath("2026-09-15T03/T/lvt1/100/lv1/85000/PT000H00M.grib2"),
            ),
            (
                f"{URL_ROOT}/T_2M/r/2026-09-15T03%3A00/s/PT000H00M.grib2",
                PurePosixPath("2026-09-15T03/T_2M/PT000H00M.grib2"),
            ),
        ],
    ]


@patch(f"{MODULE}.copy_urls")
def test_copy_icosahedral_files_is_a_no_op_when_dst_is_complete(
    mock_copy_urls: MagicMock,
) -> None:
    mock_list_files = _fake_list_files(
        {
            SRC_ROOT: ["T_2M/r/2026-09-15T00:00/s/PT000H00M.grib2"],
            ":s3:bucket/icosahedral/2026-09-15T00/": ["T_2M/PT000H00M.grib2"],
        }
    )
    _copy(mock_list_files, nwp_init_hours=[0], level_types=[])

    mock_copy_urls.assert_not_called()


@patch(f"{MODULE}.copy_urls")
def test_copy_icosahedral_files_skips_runs_that_may_still_be_publishing(
    mock_copy_urls: MagicMock,
) -> None:
    # No destination listing for the 03 run: listing it would raise KeyError.
    mock_list_files = _fake_list_files(
        {
            SRC_ROOT: TWO_RUNS_ON_DWD,
            ":s3:bucket/icosahedral/2026-09-15T00/": [],
        }
    )
    _copy(mock_list_files, now=pd.Timestamp("2026-09-15T06:59Z"))

    mock_copy_urls.assert_called_once()
    assert {
        dst_path.parts[0]
        for _url, dst_path in mock_copy_urls.call_args.kwargs["sources_and_dst_paths"]
    } == {"2026-09-15T00"}


@patch(f"{MODULE}.copy_urls")
def test_copy_icosahedral_files_raises_when_dwd_lists_no_files(
    mock_copy_urls: MagicMock,
) -> None:
    with pytest.raises(RuntimeError, match="Found no icosahedral files"):
        _copy(_fake_list_files({SRC_ROOT: []}))

    mock_copy_urls.assert_not_called()


@patch(f"{MODULE}.copy_urls")
def test_copy_icosahedral_files_does_not_list_dwd_with_no_time_budget(
    mock_copy_urls: MagicMock,
) -> None:
    mock_list_files = _fake_list_files({})
    with pytest.raises(RuntimeError, match="Stopped before listing DWD's files"):
        _copy(mock_list_files, time_budget=timedelta(0))

    mock_list_files.assert_not_called()
    mock_copy_urls.assert_not_called()


@patch(f"{MODULE}.copy_urls")
def test_copy_icosahedral_files_stops_before_a_run_once_the_time_budget_runs_out(
    mock_copy_urls: MagicMock,
) -> None:
    mock_list_files = _fake_list_files(
        {
            SRC_ROOT: TWO_RUNS_ON_DWD,
            ":s3:bucket/icosahedral/2026-09-15T00/": [],
            ":s3:bucket/icosahedral/2026-09-15T03/": [],
        }
    )
    # Readings: budget start, before the listing, before the 00 run, before the 03 run.
    monotonic = MagicMock(side_effect=[0.0, 0.0, 0.0, 7200.0])
    with (
        patch(f"{MODULE}.time.monotonic", monotonic),
        pytest.raises(RuntimeError, match="Stopped before copying run 2026-09-15T03"),
    ):
        _copy(mock_list_files, time_budget=timedelta(hours=1))

    mock_copy_urls.assert_called_once()


@patch(f"{MODULE}.copy_urls")
def test_copy_icosahedral_files_copies_an_incomplete_run_then_raises(
    mock_copy_urls: MagicMock,
) -> None:
    mock_list_files = _fake_list_files(
        {
            SRC_ROOT: [
                "T_2M/r/2026-09-15T00:00/s/PT000H00M.grib2",
                "T_2M/r/2026-09-15T00:00/s/PT001H00M.grib2",
                "T_2M/r/2026-09-15T06:00/s/PT000H00M.grib2",
            ],
            ":s3:bucket/icosahedral/2026-09-15T00/": [],
            ":s3:bucket/icosahedral/2026-09-15T06/": [],
        }
    )
    with pytest.raises(RuntimeError, match="2026-09-15T06 lacks 1 files"):
        _copy(mock_list_files, nwp_init_hours=[0, 6], level_types=[])

    assert mock_copy_urls.call_count == 2


def _runs(*runs: Sequence[str]) -> list[list[PurePosixPath]]:
    return [[PurePosixPath(path) for path in run] for run in runs]


def test_incomplete_runs_compares_main_and_intermediate_runs_separately() -> None:
    main_run = [
        "T_2M/PT000H00M.grib2",
        "T_2M/PT120H00M.grib2",
        "T/lvt1/100/lv1/500/PT000H00M.grib2",
    ]
    runs = _runs(
        [f"2026-09-15T00/{path}" for path in main_run],
        [
            "2026-09-15T03/T_2M/PT000H00M.grib2",
            "2026-09-15T03/T/lvt1/100/lv1/500/PT000H00M.grib2",
        ],
        [f"2026-09-15T06/{path}" for path in main_run],
    )
    assert incomplete_runs(runs, level_types=[100], params=[]) == []


def test_incomplete_runs_reports_files_another_run_of_its_kind_has() -> None:
    runs = _runs(
        ["2026-09-15T00/T_2M/PT000H00M.grib2", "2026-09-15T00/T_2M/PT120H00M.grib2"],
        ["2026-09-15T06/T_2M/PT000H00M.grib2"],
    )
    assert incomplete_runs(runs, level_types=[], params=[]) == [
        (
            "2026-09-15T06 lacks 1 files that other runs of its kind have,"
            " e.g. T_2M/PT120H00M.grib2"
        )
    ]


def test_incomplete_runs_reports_a_selected_level_type_missing_from_every_run() -> None:
    runs = _runs(
        [
            "2026-09-15T00/T_2M/PT000H00M.grib2",
            "2026-09-15T00/T/lvt1/100/lv1/500/PT000H00M.grib2",
        ],
        [
            "2026-09-15T06/T_2M/PT000H00M.grib2",
            "2026-09-15T06/T/lvt1/100/lv1/500/PT000H00M.grib2",
        ],
    )
    assert incomplete_runs(runs, level_types=[100, 106], params=[]) == [
        "2026-09-15T00 has no lvt1/106 files",
        "2026-09-15T06 has no lvt1/106 files",
    ]


def test_incomplete_runs_reports_a_requested_param_but_not_level_types() -> None:
    runs = _runs(["2026-09-15T00/T_2M/PT000H00M.grib2"])
    assert incomplete_runs(runs, level_types=[100], params=["T_2M", "TYPO"]) == [
        "2026-09-15T00 has no TYPO files"
    ]
