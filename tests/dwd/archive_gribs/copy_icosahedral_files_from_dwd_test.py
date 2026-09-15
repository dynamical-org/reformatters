from pathlib import PurePosixPath
from unittest.mock import MagicMock, patch

import pytest

from reformatters.dwd.archive_gribs.copy_icosahedral_files_from_dwd import (
    copy_icosahedral_files_from_dwd_https,
    icosahedral_dst_path,
    icosahedral_include_filters,
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


def _fake_list_files(
    files_by_path: dict[str, list[str]],
) -> MagicMock:
    def list_files(path: str, **_kwargs: object) -> list[PurePosixPath]:
        return [PurePosixPath(p) for p in files_by_path[path]]

    return MagicMock(side_effect=list_files)


@patch(f"{MODULE}.copy_urls")
def test_copy_icosahedral_files_copies_only_files_missing_from_dst(
    mock_copy_urls: MagicMock,
) -> None:
    mock_list_files = _fake_list_files(
        {
            ":http:/weather/nwp/v1/m/icon-eu/p/": [
                "T_2M/r/2026-09-15T00:00/s/PT000H00M.grib2",
                "T_2M/r/2026-09-15T00:00/s/PT001H00M.grib2",
                "T/lvt1/100/lv1/85000/r/2026-09-15T00:00/s/PT000H00M.grib2",
                "T_2M/r/2026-09-15T03:00/s/PT000H00M.grib2",
            ],
            ":s3:bucket/icosahedral/2026-09-15T00/": ["T_2M/PT000H00M.grib2"],
            ":s3:bucket/icosahedral/2026-09-15T03/": [],
        }
    )
    with patch(f"{MODULE}.list_files", mock_list_files):
        copy_icosahedral_files_from_dwd_https(
            dst_root_path=":s3:bucket/icosahedral",
            nwp_init_hours=[0, 3],
            level_types=[100],
            params=[],
            transfer_parallelism=8,
            checkers=4,
            stats_logging_freq="1m",
        )

    src_listing = mock_list_files.call_args_list[0]
    assert src_listing.kwargs["path"] == ":http:/weather/nwp/v1/m/icon-eu/p/"
    rclone_args = src_listing.kwargs["rclone_args"]
    assert "--http-url=https://opendata.dwd.de" in rclone_args
    assert "--http-no-head" in rclone_args
    assert "--min-age=1m" in rclone_args
    assert "--include=/*/lvt1/100/lv1/*/r/*T{00,03}:00/s/*.grib2" in rclone_args

    url_root = "https://opendata.dwd.de/weather/nwp/v1/m/icon-eu/p"
    mock_copy_urls.assert_called_once()
    assert mock_copy_urls.call_args.kwargs["dst_root_path"] == (
        ":s3:bucket/icosahedral/"
    )
    # Ordered by destination path, so the oldest run is copied first.
    assert mock_copy_urls.call_args.kwargs["sources_and_dst_paths"] == [
        (
            f"{url_root}/T/lvt1/100/lv1/85000/r/2026-09-15T00%3A00/s/PT000H00M.grib2",
            PurePosixPath("2026-09-15T00/T/lvt1/100/lv1/85000/PT000H00M.grib2"),
        ),
        (
            f"{url_root}/T_2M/r/2026-09-15T00%3A00/s/PT001H00M.grib2",
            PurePosixPath("2026-09-15T00/T_2M/PT001H00M.grib2"),
        ),
        (
            f"{url_root}/T_2M/r/2026-09-15T03%3A00/s/PT000H00M.grib2",
            PurePosixPath("2026-09-15T03/T_2M/PT000H00M.grib2"),
        ),
    ]


@patch(f"{MODULE}.copy_urls")
def test_copy_icosahedral_files_is_a_no_op_when_dst_is_complete(
    mock_copy_urls: MagicMock,
) -> None:
    mock_list_files = _fake_list_files(
        {
            ":http:/weather/nwp/v1/m/icon-eu/p/": [
                "T_2M/r/2026-09-15T00:00/s/PT000H00M.grib2",
            ],
            "/dst/2026-09-15T00/": ["T_2M/PT000H00M.grib2"],
        }
    )
    with patch(f"{MODULE}.list_files", mock_list_files):
        copy_icosahedral_files_from_dwd_https(
            dst_root_path="/dst/",
            nwp_init_hours=[0],
            level_types=[],
            params=[],
            transfer_parallelism=8,
            checkers=4,
            stats_logging_freq="1m",
        )

    mock_copy_urls.assert_not_called()
