import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import ReformatCronJob
from reformatters.common.template_config import TemplateConfig
from reformatters.google.weathernext3.forecast_virtual.template_config import STATISTICS
from scripts.timing_profile import __main__ as cli
from scripts.timing_profile.__main__ import (
    main,
    parse_args,
    select_datasets,
    trigger_producers,
)
from scripts.timing_profile.datasets import Unit
from scripts.timing_profile.sources import Listing, TimestampSource


def test_offline_cli_reports_protected_source_without_changing_constants(
    tmp_path: Path,
) -> None:
    output = tmp_path / "values.md"
    main(
        [
            "--dataset",
            "nasa-imerg-analysis-early",
            "--as-of",
            "2026-10-26T00:00:00Z",
            "--checkins-dir",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "capture"),
            "--output",
            str(output),
        ]
    )
    report = output.read_text()
    assert "not measured; keep constants" in report
    assert "PPS timestamps require credentials" in report
    assert "No production constants are changed" in report
    assert "offline exports" in report


def test_default_history_is_365_days() -> None:
    assert parse_args([]).days == 365


def test_proxy_capture_estimate_includes_all_four_wn2_cycles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = next(
        d
        for d in DYNAMICAL_DATASETS
        if d.dataset_id == "google-weathernext2-forecast-operational-virtual"
    )
    (cron,) = dataset.operational_kubernetes_resources("test")
    assert isinstance(cron, ReformatCronJob)
    source = TimestampSource(tmp_path, 1)
    start = pd.Timestamp("2026-10-27")
    end = pd.Timestamp("2026-10-29")
    logger = Mock()
    monkeypatch.setattr(cli, "log", logger)
    monkeypatch.setattr(cli, "expected_units", lambda *args: iter(()))
    monkeypatch.setattr(cli, "render_dataset", Mock(return_value=""))
    runtime = cli.measure_runtime(None, cron, start, end, virtual=True)
    cli.measure_source(dataset, cron, runtime, source, start, end, set())
    estimate = logger.info.call_args_list[0].args
    assert "uncached proxy capture estimate" in estimate[0]
    assert estimate[2] == 2656
    assert estimate[3] == 2656 / 120
    assert source.proxy_requests == 0
    source.close()


@pytest.mark.parametrize(
    "variant",
    [
        "15-day-0-1-degree",
        "15-day-0-05-degree",
        "48-hour-0-1-degree",
        "48-hour-0-05-degree",
    ],
)
def test_weathernext3_measures_proxy_finalization_with_compact_summaries(
    variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_id = f"google-weathernext3-forecast-{variant}-virtual"
    dataset = next(d for d in DYNAMICAL_DATASETS if d.dataset_id == dataset_id)
    config = dataset.template_config
    coords = config.dimension_coordinates()
    coords["lead_time"] = pd.to_timedelta([1, 2], unit="h")

    def small_coords(self: TemplateConfig[Any]) -> dict[str, Any]:
        return coords

    generated: list[pd.Timestamp] = []
    logger = Mock()
    monkeypatch.setattr(cli, "log", logger)
    original_units = cli.expected_units

    def streaming_units(
        dataset: DynamicalDataset[Any, Any],
        start: pd.Timestamp,
        end: pd.Timestamp,
        source: TimestampSource,
    ) -> Iterator[Unit]:
        for unit in original_units(dataset, start, end, source):
            generated.append(unit.anchor)
            yield unit

    monkeypatch.setattr(cli, "expected_units", streaming_units)

    def list_proxy(self: TimestampSource, prefix: str) -> Listing:
        assert any(
            "uncached proxy capture estimate" in call.args[0]
            for call in logger.info.call_args_list
        )
        assert prefix.startswith("https://wn.dynamical.org/chunks/")
        assert prefix.endswith("_")
        match = re.search(r"/(\d{8})_(\d{2})hr_", prefix)
        assert match is not None
        anchor = pd.to_datetime(match[1] + match[2], format="%Y%m%d%H")
        assert generated[-1] <= anchor + config.append_dim_frequency
        objects = {
            f"{prefix}{statistic}/c/{lead - 1}/0/0": anchor
            + pd.Timedelta(hours=6, minutes=lead * 2)
            for lead in [1, 2]
            for statistic in STATISTICS
        }
        return Listing(
            objects,
            "WeatherNext proxy timeFinalized; not first observed proxy visibility",
            "reconstructed_source_finalization",
        )

    monkeypatch.setattr(type(config), "dimension_coordinates", small_coords)
    monkeypatch.setattr(TimestampSource, "list", list_proxy)
    rendered = Mock(wraps=cli.render_dataset)
    monkeypatch.setattr(cli, "render_dataset", rendered)
    output = tmp_path / "values.md"
    main(
        [
            "--dataset",
            dataset_id,
            "--days",
            "1",
            "--workers",
            "1",
            "--as-of",
            "2026-10-29T12:00",
            "--checkins-dir",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "capture"),
            "--output",
            str(output),
        ]
    )
    report = output.read_text()
    assert "WeatherNext proxy timeFinalized" in report
    assert "not first observed proxy visibility" in report
    assert "reconstructed_source_finalization" in report
    assert "06:02:00" in report
    assert f"suspended: `{variant.startswith('48-hour')}`" in report
    measurements = rendered.call_args.args[2]
    assert measurements
    assert all(not unit.unit.files and not unit.file_arrivals for unit in measurements)
    assert all(
        unit.expected_files == len(config.data_vars) * 2 for unit in measurements
    )
    assert any(unit.complete == 6 * 3600 + 4 * 60 for unit in measurements)


def test_default_selection_skips_suspended_cronjobs_unless_requested() -> None:
    selected, suspended = select_datasets(None)
    assert suspended
    assert all(not cron.suspend for _, cron in selected)
    assert {dataset.dataset_id for dataset, _ in selected}.isdisjoint(suspended)
    requested, skipped = select_datasets([suspended[0]])
    assert [dataset.dataset_id for dataset, _ in requested] == [suspended[0]]
    assert skipped == []


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(AssertionError, match="Unknown dataset ID"):
        select_datasets(["not-a-dataset"])


def test_trigger_producers_include_archivers() -> None:
    producers = trigger_producers()
    assert producers["ecmwf-ifs-ens-46-day-daily-update"] == [
        "ecmwf-ifs-ens-46-day-gribs"
    ]
    assert producers["noaa-gfs-forecast-update"] == ["noaa-gfs-forecast-virtual"]
    assert all(
        name.endswith("-update")
        and any(
            cron.name == name
            for dataset in DYNAMICAL_DATASETS
            for cron in dataset.operational_kubernetes_resources("test")
        )
        for name in producers
    )


def test_triggered_consumer_is_found_by_its_cronjob_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        TimestampSource, "list", lambda self, prefix: Listing({}, "no objects")
    )
    output = tmp_path / "values.md"
    main(
        [
            "--dataset",
            "noaa-gfs-forecast",
            "--days",
            "1",
            "--source",
            "listings",
            "--as-of",
            "2026-10-26T00:00",
            "--checkins-dir",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / "capture"),
            "--output",
            str(output),
        ]
    )
    assert (
        "Triggered by noaa-gfs-forecast-virtual; the cron is a backstop at the first readiness tier's T, and max_delay matches its offset."
        in output.read_text()
    )
