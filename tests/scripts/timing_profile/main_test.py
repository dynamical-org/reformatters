from pathlib import Path

from scripts.timing_profile.__main__ import main, parse_args


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
