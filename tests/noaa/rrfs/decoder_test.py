from pathlib import Path

import gribberish
import numpy as np
import pytest
import rasterio

from reformatters.common.logging import get_logger

log = get_logger(__name__)
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name", ["spfh", "pevpr", "pevap", "aotk", "wildfire"])
@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="gribberish 1.8.0 ignores GRIB DRT 5.2 missing management.",
)
def test_real_noaa_missing_management_agrees_with_gdal(name: str) -> None:
    content = (FIXTURES / f"{name}-all-missing.grib2").read_bytes()
    assert len(content) in {217, 220, 241, 244}
    with rasterio.MemoryFile(content) as file, file.open() as source:
        missing = source.read_masks(1) == 0
    decoded = np.asarray(
        gribberish.parse_grib_array(content, 0, north_up=True)  # ty: ignore[unresolved-attribute]
    ).reshape(missing.shape)
    log.info(
        f"RAW MISSING MANAGEMENT gribberish unique={np.unique(decoded).tolist()!r} GDAL missing={int(missing.sum())}"
    )
    assert missing.all()
    np.testing.assert_array_equal(np.isnan(decoded), missing)
