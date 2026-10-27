import pandas as pd
import pytest

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common.kubernetes import ReformatCronJob
from scripts.timing_profile.datasets import Unit
from scripts.timing_profile.report import (
    MeasuredUnit,
    aggregate_unit,
    cron_interval,
    cycle_classes,
    recommend,
    recommended_cron,
    render_dataset,
)
from scripts.timing_profile.runtime import RuntimeStats
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


def runtime() -> RuntimeStats:
    return RuntimeStats(30, 120, 20, 240, 10, 10, 0, 0, None, None)


def measured_units(
    samples: list[tuple[int, tuple[float, ...]]],
    *,
    gated: bool = False,
    tier: str = "run",
) -> list[MeasuredUnit]:
    result = []
    for index, (hour, offsets) in enumerate(samples):
        anchor = pd.Timestamp("2026-10-20") + pd.Timedelta(days=index, hours=hour)
        files = tuple(
            frozenset([f"s3://bucket/data/{index}-{number}"])
            for number in range(len(offsets))
        )
        groups = ((files[0], files[1]), (files[2],)) if gated else ()
        unit = Unit(anchor, tier, files, 0, groups)
        listing = Listing(
            {
                next(iter(file)): anchor + pd.Timedelta(seconds=offset)
                for file, offset in zip(files, offsets, strict=True)
            },
            "S3",
        )
        result.append(aggregate_unit(unit, {"s3://bucket/data/": listing}))
    return result


@pytest.mark.parametrize(
    ("samples", "virtual", "gated", "interval", "spacing", "expected"),
    [
        (
            [(0, (600, 3000)), (6, (660, 4000))],
            True,
            False,
            21600,
            21600,
            (480, 1800, 21960, 270),
        ),
        (
            [(0, (600.5,)), (6, (1800.5,))],
            False,
            False,
            21600,
            21600,
            (1920, 1920, 21600, None),
        ),
        (
            [(0, (600,)), (1, (600,))],
            False,
            False,
            10800,
            3600,
            (660, 7860, 10800, None),
        ),
        ([(0, (60, 600, 3000, 6000))], True, True, 3600, 3600, (-60, 1740, 3960, 270)),
    ],
    ids=[
        "virtual-forecast-max-validation",
        "materialized-latest-class-ceil",
        "three-hourly-analysis",
        "gated-analysis-visibility",
    ],
)
def test_final_recommendations(
    samples: list[tuple[int, tuple[float, ...]]],
    virtual: bool,
    gated: bool,
    interval: float,
    spacing: float,
    expected: tuple[float | None, ...],
) -> None:
    units = measured_units(samples, gated=gated)
    recommendation = recommend(
        units,
        runtime(),
        virtual=virtual,
        interval=interval,
        currency_interval=interval,
        spacing=spacing,
    )
    assert (
        recommendation.offset,
        recommendation.delay,
        recommendation.deadline,
        recommendation.validation_grace,
    ) == expected


def test_virtual_classes_split_on_visibility_and_polling_uses_all_tiers() -> None:
    units = measured_units([(0, (600, 3000)), (6, (1800, 3000))], tier="z first")
    assert len(cycle_classes(units, virtual=True)) == 2
    assert len(cycle_classes(units)) == 1
    units += measured_units([(0, (60, 5000))], tier="a second")
    result = recommend(
        units,
        runtime(),
        virtual=True,
        interval=21600,
        currency_interval=21600,
        spacing=21600,
    )
    assert result.offset == -60
    assert result.delay == 2940
    assert [value.tier for value in result.classes] == [
        "z first",
        "z first",
        "a second",
    ]
    materialized = recommend(
        units,
        runtime(),
        virtual=False,
        interval=21600,
        currency_interval=21600,
        spacing=21600,
    )
    assert materialized.offset == 3060


def test_mrms_reports_pass_one_and_keeps_constants() -> None:
    dataset = next(
        ds
        for ds in DYNAMICAL_DATASETS
        if ds.dataset_id == "noaa-mrms-conus-analysis-hourly"
    )
    cron = next(
        resource
        for resource in dataset.operational_kubernetes_resources("test")
        if isinstance(resource, ReformatCronJob)
    )
    units = measured_units([(0, (3600,))], tier="position") + measured_units(
        [(0, (900,))], tier="pass 1 (exception)"
    )
    result = recommend(
        units,
        runtime(),
        virtual=False,
        interval=3600,
        currency_interval=3600,
        spacing=3600,
        manual=True,
    )
    assert result.offset is None
    assert result.delay is None
    assert result.classes == ()
    rendered = render_dataset(
        dataset, cron, units, runtime(), pd.Timestamp("2026-10-26"), None, {"S3"}
    )
    assert "### pass 1 (exception)" in rendered
    assert "keep constants; review manually" in rendered
    assert "| Cron |" not in rendered
    assert "S candidate" not in rendered
    assert "| chosen |" not in rendered


def test_rendered_constants_and_after_s_or_never_include_older_missing_unit() -> None:
    dataset = next(
        ds for ds in DYNAMICAL_DATASETS if ds.dataset_id == "noaa-gfs-forecast"
    )
    cron = next(
        resource
        for resource in dataset.operational_kubernetes_resources("test")
        if isinstance(resource, ReformatCronJob)
    )
    units = measured_units([(0, (600,)), (0, (1200,))])
    missing = Unit(
        pd.Timestamp("2026-10-22"), "run", (frozenset(["s3://bucket/data/missing"]),), 0
    )
    units.append(aggregate_unit(missing, {"s3://bucket/data/": Listing({}, "S3")}))
    rendered = render_dataset(
        dataset, cron, units, runtime(), pd.Timestamp("2026-10-26"), None, {"S3"}
    )
    assert "| 00Z | draft | 00:24:30 | 0.00% | 33.33% |" in rendered
    assert "| Cron |" in rendered
    assert "| 00Z | chosen | 00:20:54 | 0.00% | 33.33% |" in rendered
    assert "`21 0,6,12,18 * * *`" in rendered
    rows = {
        cells[1].strip(): cells[3].strip()
        for line in rendered.splitlines()
        if line.startswith("| ") and len(cells := line.split("|")) == 6
    }
    assert rows["max_delay"] == "00:21:00"
    assert rows["Pod deadline"] == "06:00:00"
    assert rows["Job deadline"] == "06:00:00"


@pytest.mark.parametrize(
    ("arrivals", "expected"),
    [([600] * 20, 660), ([600] * 20 + [3600], 900)],
    ids=["p99-cap-wins", "spread-wins-with-rare-outlier"],
)
def test_chosen_schedule_drives_final_cron_offset_and_currency(
    arrivals: list[float], expected: float
) -> None:
    units = measured_units([(0, (arrival,)) for arrival in arrivals])
    result = recommend(
        units,
        runtime(),
        virtual=False,
        interval=21600,
        currency_interval=21600,
        spacing=21600,
    )
    assert result.offset == expected
    assert result.delay == expected
    assert result.classes[0].offset == expected
