import subprocess
import time
from pathlib import PurePosixPath
from subprocess import Popen

import requests

from reformatters.common.logging import get_logger

log = get_logger(__name__)

RCD_URL = "http://localhost:5572"


def list_https(path: PurePosixPath, checkers: int = 32) -> list[PurePosixPath]:
    cmd = [
        "rclone",
        "lsf",
        str(path),
        "--fast-list",
        "--recursive",
        "--files-only",
        # The ordering of these filters matters:
        "--filter=- *pressure-level*",
        "--filter=+ *.grib2.bz2",
        "--filter=- *",
        f"--checkers={checkers}",
    ]
    result = run_command(cmd)
    return [PurePosixPath(p) for p in result.stdout.splitlines()]


def start_rclone_remote_control_daemon() -> Popen[str]:
    cmd = [
        "rclone",
        "rcd",
        "--rc-no-auth",  # Don't require auth on the rc interface to use methods which access rclone remotes.
        # "--rc-web-gui",  # TODO(Jack): Remove this in production!
        "--rc-enable-metrics",  # Enable OpenMetrics/Prometheus compatible endpoint at /metrics.
        "--no-check-dest",  # TODO(Jack): Check through no-check-dest, ignore-times, and no-traverse
        "--ignore-times",
        "--no-traverse",
        "--transfers=256",
        "--checkers=32",
    ]
    rclone_daemon = Popen(
        cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Wait for the RC server to initialise
    for _ in range(10):
        try:
            requests.get(f"{RCD_URL}/core/version", timeout=1)
        except requests.exceptions.ConnectionError:
            log.info("rclone daemon is not ready yet... waiting...")
            time.sleep(0.1)
        else:
            log.info("rclone remote control daemon is ready.")
            return rclone_daemon
    raise RuntimeError("Failed to connect to rclone daemon!")


def run_command(
    cmd: list[str], log_stdout: bool = False
) -> subprocess.CompletedProcess[str]:
    log.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.stderr:
        log.info("rclone stderr: %s", result.stderr)
    if log_stdout and result.stdout:
        log.info("rclone stdout: %s", result.stdout)
    return result


def wait_for_jobs_to_finish() -> None:
    """Polls the rclone remote control daemon until all queued transfers are finished."""
    log.info("--- Waiting for transfers to complete ---")
    while True:
        try:
            # TODO(Jack): Fix this AI slop!
            stats = requests.post(f"{RCD_URL}/core/stats", timeout=1).json()
            print(stats)
            eta = stats.get("eta")
            if eta is None:
                # Double check the 'jobid' list if you want to be extra sure
                break

            # Print progress
            megabytes_moved = stats.get("totalBytes", 0) / 1e6  # MB
            print(
                f"Active transfers: {eta} | Progress: {megabytes_moved:.2f} MB",
                end="\r",
            )
            time.sleep(1)
        except Exception as e:
            print(f"Monitoring error: {e}")
            break
    print("\nAll transfers finished.")


def main() -> None:
    # lst = list_https(path=PurePosixPath("dwd-http:/weather/nwp/icon-eu/grib/03/"))
    # print()
    # print(f"Found {len(lst):,d} files.\nFirst='{lst[0]}'\nLast='{lst[-1]}'")
    start_rclone_remote_control_daemon()
    payload = {
        "srcFs": "dwd-http:",
        "srcRemote": "/weather/nwp/icon-eu/grib/00/alhfl_s/icon-eu_europe_regular-lat-lon_single-level_2026013000_000_ALHFL_S.grib2.bz2",
        "dstFs": "/",
        "dstRemote": "/home/jack/data/ICON-EU/grib/rclone_remote_control/00/foo.grib2.bz2",
        "_async": True,
    }
    result = requests.post(f"{RCD_URL}/operations/copyfile", json=payload, timeout=1)
    result.raise_for_status()
    print(result.content)
    time.sleep(3)
    wait_for_jobs_to_finish()


if __name__ == "__main__":
    rclone_daemon = start_rclone_remote_control_daemon()
    try:
        main()
    finally:
        run_command(["rclone", "rc", "core/quit"], log_stdout=True)
        return_code = rclone_daemon.wait()
        log.info("rclone daemon return code = %d", return_code)
        if rclone_daemon.stdout and (stdout := rclone_daemon.stdout.read()):
            log.info("rclone daemon stdout: %s", stdout)
        if rclone_daemon.stderr and (stderr := rclone_daemon.stderr.read()):
            log.info("rclone daemon stderr: %s", stderr)
