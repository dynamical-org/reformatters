"""Re-fetch the real CHIRPS responses that source_data_test.py runs against.

Run from the repository root: `uv run python -m tests.ucsb_chc.chirps.refresh_fixtures`.
It makes one request per fixture, two seconds apart, and stops on any non-200.

Each GeoTIFF is reduced before it is saved: the full grid, georeferencing and
fill-value mask are kept, real values are kept inside `REAL_VALUES_WINDOW`, and
other estimated cells are set to 0. Before reducing, it asserts that the fill
value is the only negative value in the full real field.
"""

import time
from pathlib import Path

import httpx
import numpy as np
import rasterio

from reformatters.common.download import USER_AGENT
from reformatters.ucsb_chc.chirps.region_job import _BASE_URL
from reformatters.ucsb_chc.chirps.template_config import SOURCE_FILL_VALUE

FIXTURES_DIR = Path(__file__).parent / "fixtures"

FIXTURE_PATHS = (
    "final/rnl/",
    "final/rnl/2025/",
    "prelim/sat/",
    "prelim/sat/2025/",
    "final/rnl/1981/chirps-v3.0.rnl.1981.01.01.tif",
    "prelim/sat/2025/chirps-v3.0.prelim.2025.01.01.tif",
)

# Rows, columns: 0.05 degree cells over equatorial Africa.
REAL_VALUES_WINDOW = (slice(900, 1156), slice(3800, 4056))


def fixture_path(url: str) -> Path:
    relative = url.removeprefix(f"{_BASE_URL}/")
    return FIXTURES_DIR / (
        f"{relative}index.html" if relative.endswith("/") else relative
    )


def reduce_geotiff(source: Path, destination: Path) -> None:
    with rasterio.open(source) as reader:
        profile = reader.profile
        data = reader.read(1)
    np.testing.assert_array_equal(
        np.unique(data[data < 0]), np.array([SOURCE_FILL_VALUE], dtype=np.float32)
    )
    reduced = np.where(data == SOURCE_FILL_VALUE, data, np.float32(0))
    reduced[REAL_VALUES_WINDOW] = data[REAL_VALUES_WINDOW]
    profile.update(
        compress="deflate",
        predictor=3,
        zlevel=9,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as writer:
        writer.write(reduced, 1)


def main() -> None:
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120) as client:
        for path in FIXTURE_PATHS:
            url = f"{_BASE_URL}/{path}"
            response = client.get(url)
            response.raise_for_status()
            assert response.status_code == 200, f"{url}: HTTP {response.status_code}"
            destination = fixture_path(url)
            if path.endswith(".tif"):
                download = destination.with_name(f"{destination.name}.download")
                download.write_bytes(response.content)
                reduce_geotiff(download, destination)
                download.unlink()
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(response.content)
            time.sleep(2)


if __name__ == "__main__":
    main()
