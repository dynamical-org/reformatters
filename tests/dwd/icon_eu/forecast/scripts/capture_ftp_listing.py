import asyncio
import csv
from pathlib import PurePosixPath

import aioftp

# --- Configuration ---
FTP_HOST = "opendata.dwd.de"
FTP_PATH_TO_LIST = PurePosixPath("/weather/nwp/icon-eu/grib/00")
OUTPUT_FILE = PurePosixPath("tests/dwd/icon_eu/forecast/fixtures/ftp_listing_00z.csv")


async def _capture_ftp_listing_async() -> list[tuple[PurePosixPath, int, str]]:
    print(f"Connecting to FTP host: {FTP_HOST}")  # noqa: T201
    async with aioftp.Client.context(FTP_HOST) as ftp_client:
        print(f"Listing files recursively under FTP path: {FTP_PATH_TO_LIST}")  # noqa: T201
        ftp_listing: list[
            tuple[PurePosixPath, aioftp.client.UnixListInfo]
        ] = await ftp_client.list(FTP_PATH_TO_LIST, recursive=True)  # type: ignore[assignment]

    file_data = []
    for ftp_path, ftp_info in ftp_listing:
        item_type = ftp_info.get("type", "unknown")
        # For directories, size might not be meaningful or consistently present, default to 0.
        file_size_bytes = int(ftp_info.get("size", 0))
        file_data.append((ftp_path, file_size_bytes, item_type))

    print(f"Retrieved listing with {len(file_data)} items (including dirs).")  # noqa: T201
    return file_data


def capture_ftp_listing() -> None:
    all_ftp_items = asyncio.run(_capture_ftp_listing_async())

    print(f"Saving FTP listing to: {OUTPUT_FILE}")  # noqa: T201
    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["path", "file_size_bytes", "type"])  # Updated header
        for path, size, item_type in all_ftp_items:
            csv_writer.writerow([str(path), size, item_type])

    print("FTP listing captured successfully.")  # noqa: T201


if __name__ == "__main__":
    capture_ftp_listing()
