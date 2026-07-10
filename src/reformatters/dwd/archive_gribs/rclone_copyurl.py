from pathlib import Path, PurePosixPath
from typing import Any

from reformatters.common.rclone import run_command_with_concurrent_logging


def run_rclone_copyurl(
    csv_of_files_to_transfer: str,
    dst_root_path: PurePosixPath,
    transfer_parallelism: int,
    checkers: int,
    stats_logging_freq: str,  # e.g. "1m" to log stats every minute.
    env_vars: dict[str, Any] | None = None,
) -> None:
    csv_file = Path("copyurls.csv")
    csv_file.write_text(csv_of_files_to_transfer)
    cmd = (
        "/usr/bin/rclone",
        "copyurl",  # https://rclone.org/commands/rclone_copyurl
        "--urls",
        str(csv_file),
        str(dst_root_path),
        "--s3-no-check-bucket",  # Workaround for reformatters issue #428
        # Performance:
        "--fast-list",
        f"--transfers={transfer_parallelism:d}",
        f"--checkers={checkers:d}",
        # Logging:
        f"--stats={stats_logging_freq}",
        "--stats-log-level=ERROR",  # Output stats to stderr.
        "--quiet",  # Only output logs at error level.
        "--stats-one-line",  # Output stats as a single line.
    )
    run_command_with_concurrent_logging(cmd, env_vars=env_vars)
    csv_file.unlink()
