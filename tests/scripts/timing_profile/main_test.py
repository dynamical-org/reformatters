from pathlib import Path

import pytest

from scripts.timing_profile.__main__ import main, parse_args
from scripts.timing_profile.sources import TimestampSource


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


@pytest.mark.parametrize(
    "variant",
    [
        "15-day-0-1-degree",
        "15-day-0-05-degree",
        "48-hour-0-1-degree",
        "48-hour-0-05-degree",
    ],
)
def test_weathernext3_reports_missing_anonymous_timestamps(
    variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def protected_bucket(self: TimestampSource, bucket: str) -> str:
        assert bucket == "weathernext3_statistics_spatial"
        return "GCS HTTP 401; no anonymous source timestamps"

    monkeypatch.setattr(TimestampSource, "probe_gcs", protected_bucket)
    output = tmp_path / "values.md"
    main(
        [
            "--dataset",
            f"google-weathernext3-forecast-{variant}-virtual",
            "--days",
            "365",
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
    assert "GCS HTTP 401; no anonymous source timestamps" in report
    assert "not measured; keep constants" in report
    assert "suspended: `True`" in report
