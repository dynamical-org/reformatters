import pandas as pd

from reformatters.common.kubernetes import ReformatCronJob
from scripts.timing_profile.datasets import Unit
from scripts.timing_profile.report import (
    aggregate_unit,
    cron_interval,
    cycle_classes,
    recommended_cron,
)
from scripts.timing_profile.sources import Listing


def test_index_arrival_controls_usable_file_and_missing_member_blocks_unit() -> None:
    anchor = pd.Timestamp("2026-10-26")
    first = frozenset(["s3://bucket/a/file", "s3://bucket/a/file.idx"])
    second = frozenset(["s3://bucket/a/member", "s3://bucket/a/member.idx"])
    objects = {
        "s3://bucket/a/file": anchor + pd.Timedelta(seconds=60),
        "s3://bucket/a/file.idx": anchor + pd.Timedelta(seconds=120),
        "s3://bucket/a/member": anchor + pd.Timedelta(seconds=180),
    }
    listings = {"s3://bucket/a/": Listing(objects, "S3")}
    unit = Unit(anchor, "run", (first, second), 0)
    measured = aggregate_unit(unit, listings)
    assert measured.first == 120
    assert measured.complete is None
    assert measured.missing_files == 1
    objects["s3://bucket/a/member.idx"] = anchor + pd.Timedelta(seconds=240)
    measured = aggregate_unit(unit, listings)
    assert measured.complete == 240
    assert measured.missing_files == 0
    censored = aggregate_unit(unit, listings, as_of=anchor + pd.Timedelta(seconds=200))
    assert censored.complete is None
    assert censored.missing_files == 1


def test_cycle_classes_split_medians_at_fifteen_minutes() -> None:
    units = []
    for hour, offset in [(0, 1800), (6, 600), (12, 1810), (18, 610)]:
        anchor = pd.Timestamp("2026-10-26") + pd.Timedelta(hours=hour)
        unit = Unit(anchor, "run", (frozenset(["s3://bucket/a/file"]),), 0)
        units.append(
            aggregate_unit(
                unit,
                {
                    "s3://bucket/a/": Listing(
                        {"s3://bucket/a/file": anchor + pd.Timedelta(seconds=offset)},
                        "S3",
                    )
                },
            )
        )
    assert [name for name, _ in cycle_classes(units)] == ["06/18Z", "00/12Z"]


def test_recommended_cron_wraps_offsets_and_preserves_day_fields() -> None:
    cron = ReformatCronJob(
        name="test-update",
        schedule="5 1,7,13,19 * * *",
        image="test",
        dataset_id="test",
        cpu="1",
        memory="1G",
    )
    assert recommended_cron(8 * 3600 + 15 * 60, 6 * 3600, cron) == "15 2,8,14,20 * * *"
    assert recommended_cron(17 * 60, 3600, cron) == "17 * * * *"


def test_calendar_intervals_bound_deadline_and_scalar_currency() -> None:
    cron = ReformatCronJob(
        name="test-update",
        schedule="0 0 */3 * *",
        image="test",
        dataset_id="test",
        cpu="1",
        memory="1G",
    )
    assert cron_interval(cron, pd.Timestamp("2026-10-26")) == 86400
    assert cron_interval(cron, pd.Timestamp("2026-10-26"), maximum=True) == 3 * 86400
