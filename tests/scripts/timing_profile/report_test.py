import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common.kubernetes import ReformatCronJob
from scripts.timing_profile.datasets import Unit
from scripts.timing_profile.report import (
    MeasuredUnit,
    aggregate_unit,
    class_history,
    cron_interval,
    cycle_classes,
    rate_arrivals,
    recommend,
    recommended_cron,
    render_class_evidence,
    render_dataset,
    virtual_window_contains_anchor,
)
from scripts.timing_profile.runtime import RuntimeStats
from scripts.timing_profile.sources import Listing
from scripts.timing_profile.statistics import read_rates


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
    quarter_hour = recommended_cron(6 * 60, 15 * 60, cron)
    assert quarter_hour == "6,21,36,51 * * * *"
    shifted = cron.model_copy(update={"schedule": quarter_hour})
    assert shifted.next_fire_time(pd.Timestamp("2026-10-29T23:51")) == pd.Timestamp(
        "2026-10-30T00:06"
    )
    assert shifted.previous_fire_time(
        pd.Timestamp("2026-10-30T00:05:59")
    ) == pd.Timestamp("2026-10-29T23:51")


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
    return RuntimeStats(30, 120, 20, 240, 10, 10, 0, 0, None, None, 10, 30, 20)


def measured_units(
    samples: list[tuple[int, tuple[float, ...]]],
    *,
    gated: bool = False,
    tier: str = "run",
    copies: int = 1,
) -> list[MeasuredUnit]:
    result = []
    for index, (hour, offsets) in enumerate(samples * copies):
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
            (60, 1080, 21960, 270),
        ),
        (
            [(0, (600.5,)), (6, (1800.5,))],
            False,
            False,
            21600,
            21600,
            (2040, 2040, 21600, None),
        ),
        (
            [(0, (600,)), (1, (600,))],
            False,
            False,
            10800,
            3600,
            (780, 7980, 10800, None),
        ),
        ([(0, (600,))], False, False, 86400, 86400, (780, 780, 21600, None)),
        ([(0, (60, 600, 3000, 6000))], True, True, 3600, 3600, (360, 1020, 3960, 270)),
    ],
    ids=[
        "virtual-forecast-max-validation",
        "materialized-latest-class-ceil",
        "three-hourly-analysis",
        "materialized-daily-deadline-cap",
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
    units = measured_units(samples, gated=gated, copies=300)
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
    units = measured_units(
        [(0, (600, 3000)), (6, (1800, 3000))], tier="z first", copies=300
    )
    assert len(cycle_classes(units, virtual=True)) == 2
    assert len(cycle_classes(units)) == 1
    units += measured_units([(0, (60, 5000))], tier="a second", copies=300)
    result = recommend(
        units,
        runtime(),
        virtual=True,
        interval=21600,
        currency_interval=21600,
        spacing=21600,
    )
    assert result.offset == 60
    assert result.delay == 2220
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
    assert materialized.offset == 3180


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


def test_rates_at_recommended_single_cron_exclude_unobserved_units() -> None:
    dataset = next(
        ds for ds in DYNAMICAL_DATASETS if ds.dataset_id == "noaa-gfs-forecast"
    )
    cron = next(
        resource
        for resource in dataset.operational_kubernetes_resources("test")
        if isinstance(resource, ReformatCronJob)
    )
    units = measured_units([(0, (600,)), (0, (1200,))], copies=300)
    missing = Unit(
        pd.Timestamp("2026-10-22"), "run", (frozenset(["s3://bucket/data/missing"]),), 0
    )
    units.append(aggregate_unit(missing, {"s3://bucket/data/": Listing({}, "S3")}))
    rendered = render_dataset(
        dataset, cron, units, runtime(), pd.Timestamp("2026-10-26"), None, {"S3"}
    )
    assert "| 00Z | 00:23:00 | 00:23:00 | 0.00% | 0.00% | 1 |" in rendered
    assert "| Cron |" in rendered
    assert "first validation failure is quiet" in rendered
    assert "`23 0,6,12,18 * * *`" in rendered
    rows = {
        cells[1].strip(): cells[3].strip()
        for line in rendered.splitlines()
        if line.startswith("| ") and len(cells := line.split("|")) == 6
    }
    assert rows["max_delay"] == "00:23:00"
    assert rows["Pod deadline"] == "06:00:00"
    assert rows["Job deadline"] == "06:00:00"


@pytest.mark.parametrize(
    ("arrivals", "expected"),
    [([600] * 20, 780), ([600] * 20 + [3600], None)],
    ids=["p99-plus-three", "tail-needs-hand-value"],
)
def test_chosen_schedule_drives_final_cron_offset_and_currency(
    arrivals: list[float], expected: float | None
) -> None:
    units = measured_units([(0, (arrival,)) for arrival in arrivals], copies=300)
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
    assert result.classes[0].hand_value == (expected is None)


def test_evidence_restarts_shifted_history_and_flags_daily_small_samples() -> None:
    units = measured_units(
        [(0, (600,)) for _ in range(172)] + [(0, (1800,)) for _ in range(8)]
    )
    end = pd.Timestamp("2026-10-20") + pd.Timedelta(days=180)
    active, boundary, suspected = class_history(units, end, virtual=False)
    assert not suspected
    assert active == units[172:]
    assert boundary == units[172].unit.anchor
    rendered = "\n".join(render_class_evidence("00Z", units, end, virtual=False))
    assert "source **listings**" in rendered
    assert "wxopticon 0/180 units" in rendered
    assert "history 8 sampled days / 8 calendar days; n=8" in rendered
    assert "current_object_write=8" in rendered
    assert "Short history" in rendered
    assert "Small sample" in rendered
    assert "Regime restart" in rendered
    assert "p99 + 3 min: 00:33:00; max normal + 3 min: 00:33:00" in rendered
    result = recommend(
        units,
        runtime(),
        virtual=False,
        interval=86400,
        currency_interval=86400,
        spacing=86400,
        end=end,
    )
    assert result.offset is None
    assert result.classes[0].hand_value
    assert result.classes[0].threshold == 1980


def test_tail_dominated_report_tags_outliers_and_withholds_constants() -> None:
    units = measured_units(
        [(0, (600,)) for _ in range(97)] + [(0, (3000,)), (0, (600,)), (0, (9000,))]
    )
    end = units[-1].unit.anchor + pd.Timedelta(hours=12)
    rendered = "\n".join(render_class_evidence("00Z", units, end, virtual=False))
    assert "tail-dominated, hand value needed" in rendered
    assert "02:30:00; listings; current_object_write" in rendered
    assert "First gap above the median" in rendered
    result = recommend(
        units,
        runtime(),
        virtual=False,
        interval=86400,
        currency_interval=86400,
        spacing=86400,
        end=end,
    )
    assert result.offset is None
    assert result.delay is None


def test_rates_use_latest_class_single_cron() -> None:
    dataset = next(
        ds for ds in DYNAMICAL_DATASETS if ds.dataset_id == "noaa-gfs-forecast"
    )
    cron = next(
        resource
        for resource in dataset.operational_kubernetes_resources("test")
        if isinstance(resource, ReformatCronJob)
    )
    units = measured_units(
        [(0, (600,)), (0, (900,)), (6, (1800,)), (6, (1800,))], copies=300
    )
    rendered = render_dataset(
        dataset, cron, units, runtime(), pd.Timestamp("2026-10-26"), None, {"S3"}
    )
    assert "| 00Z | 00:18:00 | 00:33:00 | 0.00% | 0.00% |" in rendered
    assert "| 06Z | 00:33:00 | 00:33:00 | 0.00% | 0.00% |" in rendered


def test_rate_estimates_use_runtime_medians_rather_than_budget_p95s() -> None:
    dataset = next(
        ds for ds in DYNAMICAL_DATASETS if ds.dataset_id == "noaa-gfs-forecast"
    )
    cron = next(
        resource
        for resource in dataset.operational_kubernetes_resources("test")
        if isinstance(resource, ReformatCronJob)
    )
    units = measured_units([(0, (600,)) for _ in range(100)] + [(0, (900,))], copies=3)
    budgets = RuntimeStats(200, 1000, 1000, 1200, 10, 10, 0, 0, None, None, 10, 30, 20)
    end = units[-1].unit.anchor + pd.Timedelta(hours=12)
    rendered = render_dataset(dataset, cron, units, budgets, end, None, {"S3"})
    assert "| 00Z | 00:13:00 | 00:13:00 | 0.99% | 0.00% |" in rendered


@pytest.mark.parametrize("slow_days", [2, 3])
def test_short_daily_outage_requires_a_hand_value(slow_days: int) -> None:
    units = measured_units(
        [(0, (600,))] * (180 - slow_days) + [(0, (7200,))] * slow_days
    )
    end = units[-1].unit.anchor + pd.Timedelta(days=1)
    active, boundary, suspected = class_history(units, end, virtual=False)
    assert len(active) == (180 if slow_days == 2 else 5)
    assert suspected == (slow_days == 2)
    assert (boundary is None) == (slow_days == 2)
    recommendation = recommend(
        units,
        runtime(),
        virtual=False,
        interval=86400,
        currency_interval=86400,
        spacing=86400,
        end=end,
    )
    assert recommendation.offset is None
    assert recommendation.delay is None
    assert recommendation.classes[0].hand_value


@pytest.mark.parametrize("usable_days", [77, 100])
def test_single_slow_day_does_not_erase_partial_history(usable_days: int) -> None:
    units = measured_units([(0, (3600,))] * (usable_days * 4))
    units = [
        replace(
            unit,
            unit=replace(
                unit.unit,
                anchor=pd.Timestamp("2026-01-01") + pd.Timedelta(hours=6 * index),
            ),
        )
        for index, unit in enumerate(units)
    ]
    units = [
        replace(unit, complete=7200, first=7200) if index >= len(units) - 4 else unit
        for index, unit in enumerate(units)
    ]
    end = units[-1].unit.anchor + pd.Timedelta(hours=6)
    active, boundary, suspected = class_history(units, end, virtual=False)
    assert active == units
    assert boundary is None
    assert suspected
    rendered = "\n".join(
        render_class_evidence("00/06/12/18Z", units, end, virtual=False)
    )
    assert "suspected change, review" in rendered
    assert "Regime restart" not in rendered


def test_icon_shaped_rate_denominator_excludes_empty_but_keeps_partial_units() -> None:
    units = measured_units([(0, (600,)), (0, (700,)), (0, (900,))])
    old = Unit(
        pd.Timestamp("2026-01-01"), "run", (frozenset(["s3://bucket/data/none"]),), 0
    )
    units.extend(
        [
            aggregate_unit(old, {"s3://bucket/data/": Listing({}, "S3")})
            for _ in range(10)
        ]
    )
    end = pd.Timestamp("2027-01-01")
    arrivals = rate_arrivals(units, end, 900)
    assert arrivals == [600, 700, 900]
    assert read_rates(arrivals, 750, 10, 30, 20) == (1 / 3, 0)
    partial = measured_units([(0, (600, 700))])[0]
    partial = replace(partial, complete=None, missing_files=1)
    arrivals = rate_arrivals([*units, partial], end, 900)
    assert arrivals == [600, 700, 900, None]
    assert read_rates(arrivals, 750, 10, 30, 20) == (0.5, 0.25)


def test_virtual_hand_value_preserves_polling_fire() -> None:
    units = measured_units(
        [(0, (600,))] * 97 + [(0, (3000,)), (0, (600,)), (0, (9000,))], copies=4
    )
    end = units[-1].unit.anchor + pd.Timedelta(hours=12)
    recommendation = recommend(
        units,
        runtime(),
        virtual=True,
        interval=21600,
        currency_interval=21600,
        spacing=21600,
        end=end,
    )
    assert recommendation.classes[0].hand_value
    assert recommendation.offset == 60
    assert recommendation.delay is None
    dataset = next(
        ds for ds in DYNAMICAL_DATASETS if ds.dataset_id == "noaa-gfs-forecast-virtual"
    )
    cron = next(
        resource
        for resource in dataset.operational_kubernetes_resources("test")
        if isinstance(resource, ReformatCronJob)
    )
    rendered = render_dataset(dataset, cron, units, runtime(), end, None, {"S3"})
    assert "| Cron |" in rendered
    assert "| max_delay |" in rendered
    assert "hand value needed" in rendered
    assert "One-cron offset: 00:01:00" in rendered


def test_virtual_e_ignores_exclusions_and_regime_trimming() -> None:
    units = measured_units([(0, (2400,))] * 10 + [(0, (6000,))] * 10)
    end = units[-1].unit.anchor + pd.Timedelta(hours=12)
    assert len(class_history(units, end, virtual=True)[0]) == 10
    result = recommend(
        units,
        runtime(),
        virtual=True,
        interval=21600,
        currency_interval=21600,
        spacing=21600,
        end=end,
        excluded={units[0].unit.anchor},
    )
    assert result.offset == 1440
    assert result.classes[0].threshold == 6180
    dataset = next(
        ds for ds in DYNAMICAL_DATASETS if ds.dataset_id == "noaa-gfs-forecast-virtual"
    )
    cron = next(
        resource
        for resource in dataset.operational_kubernetes_resources("test")
        if isinstance(resource, ReformatCronJob)
    )
    report = render_dataset(
        dataset, cron, measured_units([(0, (60,))]), runtime(), end, None, set()
    )
    assert "Anchor guard" in report
    assert "clamped to the first minute after the source anchor" in report


def test_anchor_guard_includes_cycle_in_exclusive_template_window() -> None:
    dataset = next(
        ds for ds in DYNAMICAL_DATASETS if ds.dataset_id == "noaa-gfs-forecast-virtual"
    )
    units = measured_units([(0, (60,))])
    result = recommend(
        units,
        runtime(),
        virtual=True,
        interval=21600,
        currency_interval=21600,
        spacing=21600,
    )
    assert result.offset is not None
    anchor = units[0].unit.anchor
    labels = dataset.template_config.append_dim_coordinates(
        anchor + pd.Timedelta(seconds=result.offset)
    )
    assert labels[-1] == anchor


def test_excluding_every_anchor_keeps_virtual_e() -> None:
    units = measured_units([(0, (2400,))])
    result = recommend(
        units,
        runtime(),
        virtual=True,
        interval=21600,
        currency_interval=21600,
        spacing=21600,
        excluded={units[0].unit.anchor},
    )
    assert result.offset == 1440
    assert result.delay is None


def test_subhourly_window_guard_checks_actual_window_override() -> None:
    dataset = next(
        ds
        for ds in DYNAMICAL_DATASETS
        if ds.dataset_id == "noaa-rrfs-forecast-sub-hourly-virtual"
    )
    anchor = pd.Timestamp("2026-10-27")
    assert not virtual_window_contains_anchor(dataset, anchor, 60)
    assert not virtual_window_contains_anchor(dataset, anchor, 3600)
    assert virtual_window_contains_anchor(dataset, anchor, 3660)


def test_year_of_real_aifs_history_requires_a_hand_value_for_latest_regime() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/aifs_ens_arrivals.json").read_text()
    )
    anchors = pd.date_range(
        fixture["start"], periods=len(fixture["arrivals_s"]), freq="6h"
    ).tz_convert(None)
    units = [
        MeasuredUnit(Unit(anchor, "run", (), 0), value, value, 0, {})
        for anchor, value in zip(anchors, fixture["arrivals_s"], strict=True)
    ]
    result = recommend(
        units,
        runtime(),
        virtual=False,
        interval=21600,
        currency_interval=21600,
        spacing=21600,
        end=pd.Timestamp("2026-10-28T20"),
    )
    assert result.classes[0].threshold == 7 * 3600 + 44 * 60
    assert result.classes[0].hand_value
    assert result.offset is None
    assert result.delay is None


def test_polling_e_uses_first_visibility_instead_of_earlier_raw_file() -> None:
    anchor = pd.Timestamp("2026-01-01")
    early = frozenset(["s3://bucket/cycle/prior-f01"])
    gate = frozenset(["s3://bucket/cycle/own-f00"])
    unit = Unit(anchor, "position", (early, gate), 0, ((gate,),))
    measured = aggregate_unit(
        unit,
        {
            "s3://bucket/cycle/": Listing(
                {
                    next(iter(early)): anchor - pd.Timedelta(seconds=701),
                    next(iter(gate)): anchor + pd.Timedelta(seconds=2840),
                },
                "S3",
            )
        },
    )
    assert measured.file_arrivals[early] == -701
    assert measured.first == 2840
    units = [measured]
    result = recommend(
        units,
        runtime(),
        virtual=True,
        interval=3600,
        currency_interval=3600,
        spacing=3600,
    )
    assert result.offset == 1860
    dataset = next(
        ds for ds in DYNAMICAL_DATASETS if ds.dataset_id == "noaa-hrrr-analysis-virtual"
    )
    cron = next(
        resource
        for resource in dataset.operational_kubernetes_resources("test")
        if isinstance(resource, ReformatCronJob)
    )
    rendered = render_dataset(
        dataset,
        cron,
        units,
        runtime(),
        units[0].unit.anchor + pd.Timedelta("1h"),
        None,
        set(),
    )
    assert "without exclusions: 00:47:20." in rendered
    assert "One-cron offset: 00:31:00" in rendered
    assert "Anchor guard" not in rendered


@pytest.mark.parametrize(
    ("completion", "lead", "fire", "relation"),
    [
        (6000, 180, 2160, "precedes"),
        (4800, 900, 1440, "is at or after"),
        (5130, 900, 1440, "is at or after"),
    ],
    ids=["cutoff-before-p99", "cutoff-after-p99", "cutoff-equals-p99"],
)
def test_virtual_lead_compares_poll_cutoff_with_retained_completion_p99(
    completion: float,
    lead: float,
    fire: float,
    relation: str,
) -> None:
    units = measured_units([(0, (2400, completion))], copies=300)
    end = units[-1].unit.anchor + pd.Timedelta("1h")
    result = recommend(
        units,
        runtime(),
        virtual=True,
        interval=3600,
        currency_interval=3600,
        spacing=3600,
        end=end,
    )
    assert result.poll_lead == lead
    assert result.offset == fire
    assert result.completion_p99 == completion
    dataset = next(
        ds
        for ds in DYNAMICAL_DATASETS
        if ds.dataset_id == "noaa-hrrr-forecast-18-hour-virtual"
    )
    cron = next(
        resource
        for resource in dataset.operational_kubernetes_resources("test")
        if isinstance(resource, ReformatCronJob)
    )
    rendered = render_dataset(dataset, cron, units, runtime(), end, None, set())
    assert f"15-minute-lead cutoff 01:25:30 {relation}" in rendered
    assert "maximum retained-class completion p99" in rendered
    assert f"Polling lead: 00:{int(lead / 60):02}:00" in rendered


def test_virtual_poll_lead_uses_latest_retained_class_completion_p99() -> None:
    units = measured_units([(0, (2400, 4800)), (6, (3600, 6000))], copies=300)
    assert len(cycle_classes(units, virtual=True)) == 2
    result = recommend(
        units,
        runtime(),
        virtual=True,
        interval=3600,
        currency_interval=3600,
        spacing=3600,
        end=units[-1].unit.anchor + pd.Timedelta("1h"),
    )
    assert result.completion_p99 == 6000
    assert result.poll_lead == 180
    assert result.offset == 2160
