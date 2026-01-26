from pathlib import Path

from reformatters.dwd.parse_rclone_log import TransferSummary, parse_and_log_rclone_json


def test_parse_rclone_log() -> None:
    example_log_path = Path(__file__).parent / "example_rclone_copy.ndjson"
    stderr = example_log_path.read_text()

    log_entries = parse_and_log_rclone_json(stderr)
    assert len(log_entries) == 54

    summary = TransferSummary.from_rclone_stats(log_entries)

    assert summary.total_transfers == 52
    assert summary.total_bytes == 60540318
    assert summary.total_checks == 0
    assert summary.errors == 0
    assert summary.elapsed_time == 2.6359994970000002
    assert summary.transfer_time == 2.501648604
    assert summary.listed == 155


def test_transfer_summary_str() -> None:
    summary = TransferSummary(
        total_transfers=52,
        total_bytes=60540318,
        total_checks=0,
        errors=0,
        elapsed_time=2.636,
        transfer_time=2.502,
        listed=155,
    )
    expected = (
        "52 files transferred, 23.076 MiB/sec, 0.056 GiB total transferred, "
        "0 files checked, 0 errors, 2.636 seconds rclone runtime, "
        "2.502 seconds transfer time, 155 directories listed."
    )
    assert str(summary) == expected
