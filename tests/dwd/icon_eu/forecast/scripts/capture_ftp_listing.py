"""Script for retrieving a real FTP listing and saving as a CSV file.

The CSV file is committed to the git repo, so most users will never have
to run this script. Instead most users can just use the pre-saved CSV
file.
"""

import asyncio
import csv
from pathlib import PurePosixPath

import aioftp

# --- Configuration ---
FTP_HOST = "opendata.dwd.de"
FTP_PATH_TO_LIST = PurePosixPath("/weather/nwp/icon-eu/grib/00")
OUTPUT_FILE = PurePosixPath("tests/dwd/icon_eu/forecast/fixtures/ftp_listing_00z.csv")


async def _capture_ftp_listing_async() -> list[tuple[PurePosixPath, int]]:
    print(f"Connecting to FTP host: {FTP_HOST}")  # noqa: T201
    async with aioftp.Client.context(FTP_HOST) as ftp_client:
        print(f"Listing files recursively under FTP path: {FTP_PATH_TO_LIST}")  # noqa: T201
        ftp_listing: list[
            tuple[PurePosixPath, aioftp.client.UnixListInfo]
        ] = await ftp_client.list(FTP_PATH_TO_LIST, recursive=True)  # type: ignore[assignment]

    file_data = []
    for ftp_path, ftp_info in ftp_listing:
        # We only care about files and their sizes
        if ftp_info.get("type") == "file":
            file_size_bytes = int(ftp_info.get("size", 0))
            file_data.append((ftp_path, file_size_bytes))

    print(f"Retrieved listing with {len(file_data)} files.")  # noqa: T201
    return file_data


def capture_ftp_listing() -> None:
    file_data = asyncio.run(_capture_ftp_listing_async())

    print(f"Saving FTP listing to: {OUTPUT_FILE}")  # noqa: T201
    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["path", "file_size_bytes"])
        for path, size in file_data:
            csv_writer.writerow([str(path), size])

    print("FTP listing captured successfully.")  # noqa: T201


if __name__ == "__main__":
    capture_ftp_listing()
