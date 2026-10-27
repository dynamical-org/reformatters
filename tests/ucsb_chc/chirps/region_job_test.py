from pathlib import Path
from unittest.mock import MagicMock

import httpx
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import download
from reformatters.common.download import RateLimiter
from reformatters.common.pydantic import replace
from reformatters.ucsb_chc.chirps import region_job
from reformatters.ucsb_chc.chirps.analysis_final.region_job import (
    UcsbChcChirpsAnalysisFinalRegionJob,
)
from reformatters.ucsb_chc.chirps.analysis_final.template_config import (
    UcsbChcChirpsAnalysisFinalTemplateConfig,
)
from reformatters.ucsb_chc.chirps.analysis_preliminary.region_job import (
    UcsbChcChirpsAnalysisPreliminaryRegionJob,
)
from reformatters.ucsb_chc.chirps.analysis_preliminary.template_config import (
    UcsbChcChirpsAnalysisPreliminaryTemplateConfig,
)
from reformatters.ucsb_chc.chirps.chirps_config_models import ChirpsProduct
from reformatters.ucsb_chc.chirps.region_job import (
    UcsbChcChirpsAnalysisMaterializedRegionJob,
    UcsbChcChirpsAnalysisSourceFileCoord,
)
from reformatters.ucsb_chc.chirps.template_config import (
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    MM_PER_DAY_TO_KG_M2_S,
    SOURCE_FILL_VALUE,
)


def _job(
    product: ChirpsProduct = "final",
) -> UcsbChcChirpsAnalysisMaterializedRegionJob:
    job_class = (
        UcsbChcChirpsAnalysisFinalRegionJob
        if product == "final"
        else UcsbChcChirpsAnalysisPreliminaryRegionJob
    )
    config = (
        UcsbChcChirpsAnalysisFinalTemplateConfig()
        if product == "final"
        else UcsbChcChirpsAnalysisPreliminaryTemplateConfig()
    )
    return job_class(
        tmp_store=Path("unused.zarr"),
        template_ds=xr.DataTree(xr.Dataset(attrs={"dataset_id": config.dataset_id})),
        data_vars=list(config.data_vars),
        append_dim="time",
        region=slice(0, 1),
        reformat_job_name="test",
    )


def test_variant_region_jobs_carry_product() -> None:
    assert (
        UcsbChcChirpsAnalysisFinalRegionJob.model_fields["product"].default == "final"
    )
    assert (
        UcsbChcChirpsAnalysisPreliminaryRegionJob.model_fields["product"].default
        == "preliminary"
    )


def test_final_url() -> None:
    coord = UcsbChcChirpsAnalysisSourceFileCoord(
        product="final", time=pd.Timestamp("2020-06-15")
    )
    assert coord.get_url() == (
        "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/2020/"
        "chirps-v3.0.rnl.2020.06.15.tif"
    )


def test_preliminary_url() -> None:
    coord = UcsbChcChirpsAnalysisSourceFileCoord(
        product="preliminary", time=pd.Timestamp("2026-08-25")
    )
    assert coord.get_url() == (
        "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/prelim/sat/2026/"
        "chirps-v3.0.prelim.2026.08.25.tif"
    )


def test_download_file_never_falls_back_to_the_other_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []

    def fake_download(url: str, dataset_id: str, *, rate_limiter: RateLimiter) -> Path:
        requested.append(url)
        if "/prelim/sat/" in url:
            raise FileNotFoundError(url)
        return Path(url)

    monkeypatch.setattr(
        "reformatters.ucsb_chc.chirps.region_job.httpx_download_to_disk", fake_download
    )
    coord = UcsbChcChirpsAnalysisSourceFileCoord(
        product="preliminary", time=pd.Timestamp("2025-06-15")
    )

    with pytest.raises(FileNotFoundError):
        _job("preliminary").download_file(coord)

    assert requested == [coord.get_url()]
    assert "/prelim/sat/" in requested[0]


def test_download_file_propagates_missing_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download(url: str, dataset_id: str, *, rate_limiter: RateLimiter) -> Path:
        raise FileNotFoundError(url)

    monkeypatch.setattr(
        "reformatters.ucsb_chc.chirps.region_job.httpx_download_to_disk", fake_download
    )
    coord = UcsbChcChirpsAnalysisSourceFileCoord(
        product="final", time=pd.Timestamp("2030-01-01")
    )

    with pytest.raises(FileNotFoundError):
        _job().download_file(coord)


@pytest.mark.parametrize("product", ["final", "preliminary"])
def test_generate_source_file_coords(
    monkeypatch: pytest.MonkeyPatch,
    product: ChirpsProduct,
) -> None:
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2025-01-03")),
    )
    times = pd.date_range("2024-12-31", "2025-01-03", freq="1D")
    processing_region_ds = xr.Dataset(coords={"time": times})
    job = _job(product)
    present = [
        UcsbChcChirpsAnalysisSourceFileCoord(product=product, time=time)
        for time in times[:3]
    ]
    requested: list[str] = []

    def get(url: str, *, rate_limiter: RateLimiter) -> httpx.Response:
        assert rate_limiter is region_job.chirps_rate_limiter
        requested.append(url)
        if not url.rstrip("/").rsplit("/", 1)[-1].isdigit():
            return httpx.Response(
                200,
                text='<a href="../">Parent</a><a href="2024/">2024</a><a href="2025/">2025</a>',
            )
        links = "".join(
            f"<tr><td><a href='{coord.get_url().rsplit('/', 1)[-1]}'>{coord.time}</a></td></tr>"
            for coord in present
            if coord.get_url().startswith(url)
        )
        absent_url = UcsbChcChirpsAnalysisSourceFileCoord(
            product=product, time=times[-1]
        ).get_url()
        return httpx.Response(
            200,
            text=f"""<html><h1>Index of {url}</h1><table>
            <a href="../">Parent Directory</a><a href="?C=N&amp;O=D">Name</a>
            {links}<a href="{absent_url}.gz">compressed</a>
            <a href="https://example.org/{absent_url.rsplit("/", 1)[-1]}">other host</a>
            {absent_url}</table></html>""",
        )

    monkeypatch.setattr(region_job, "httpx_get", get)
    coords = job.generate_source_file_coords(processing_region_ds, [])
    assert coords == present
    assert coords[0].out_loc() == {"time": times[0]}
    assert len(requested) == 3

    present.append(
        UcsbChcChirpsAnalysisSourceFileCoord(product=product, time=times[-1])
    )
    assert job.generate_source_file_coords(processing_region_ds, []) == present
    assert len(requested) == 6


@pytest.mark.parametrize("product", ["final", "preliminary"])
@pytest.mark.parametrize("missing_index", [1, 3])
def test_missing_historical_dates_fail(
    monkeypatch: pytest.MonkeyPatch,
    product: ChirpsProduct,
    missing_index: int,
) -> None:
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2026-10-26")),
    )
    times = pd.date_range("2026-04-14", periods=4)
    available = {
        UcsbChcChirpsAnalysisSourceFileCoord(product=product, time=time).get_url()
        for index, time in enumerate(times)
        if index != missing_index
    }
    monkeypatch.setattr(region_job, "_available_file_urls", lambda *args: available)

    with pytest.raises(
        AssertionError, match="source listing is missing historical dates"
    ):
        _job(product).generate_source_file_coords(
            xr.Dataset(coords={"time": times}), []
        )


def test_empty_listing_and_empty_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2025-01-01")),
    )
    get = MagicMock(
        side_effect=[
            httpx.Response(
                200, text='<a href="../">Parent</a><a href="2025/">2025</a>'
            ),
            httpx.Response(200, text='<a href="../">Parent</a>'),
            httpx.Response(
                200, text='<a href="../">Parent</a><a href="2025/">2025</a>'
            ),
        ]
    )
    monkeypatch.setattr(region_job, "httpx_get", get)
    job = _job()
    assert job.generate_source_file_coords(xr.Dataset(coords={"time": []}), []) == []
    get.assert_not_called()
    assert (
        job.generate_source_file_coords(
            xr.Dataset(coords={"time": pd.date_range("2025-01-01", periods=2)}), []
        )
        == []
    )
    assert get.call_count == 2
    assert (
        job.generate_source_file_coords(
            xr.Dataset(coords={"time": pd.date_range("2026-01-01", periods=2)}), []
        )
        == []
    )
    assert get.call_count == 3


@pytest.mark.parametrize("missing_year", [True, False])
def test_missing_historical_listing_fails(
    monkeypatch: pytest.MonkeyPatch, missing_year: bool
) -> None:
    get = MagicMock(
        side_effect=[
            httpx.Response(
                200,
                text=f'<a href="../">Parent</a><a href="{2026 if missing_year else 2025}/">year</a>',
            ),
            httpx.Response(
                200,
                text='<a href="../">Parent</a><a href="chirps-v3.1.rnl.2025.01.01.tif">wrong version</a>',
            ),
        ]
    )
    monkeypatch.setattr(region_job, "httpx_get", get)
    with pytest.raises(
        AssertionError, match=r"Missing historical directory|No CHIRPS files"
    ):
        _job().generate_source_file_coords(
            xr.Dataset(coords={"time": pd.date_range("2025-01-01", periods=2)}), []
        )


@pytest.mark.parametrize("status", [403, 404, 503, None])
@pytest.mark.parametrize("year_listing", [False, True])
def test_listing_failure_propagates(
    monkeypatch: pytest.MonkeyPatch, status: int | None, year_listing: bool
) -> None:
    def get(url: str, *, rate_limiter: RateLimiter) -> httpx.Response:
        if year_listing and not url.endswith("2025/"):
            return httpx.Response(
                200, text='<a href="../">Parent</a><a href="2025/">2025</a>'
            )
        if status is None:
            raise httpx.ReadTimeout("listing timed out")
        response = httpx.Response(status, request=httpx.Request("GET", url))
        response.raise_for_status()
        return response

    monkeypatch.setattr(region_job, "httpx_get", get)
    with pytest.raises(httpx.HTTPError):
        _job().generate_source_file_coords(
            xr.Dataset(coords={"time": pd.date_range("2025-01-01", periods=2)}), []
        )


def test_malformed_listing_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        region_job,
        "httpx_get",
        MagicMock(return_value=httpx.Response(200, text="<html>Access denied</html>")),
    )
    with pytest.raises(AssertionError, match="no parent link"):
        _job().generate_source_file_coords(
            xr.Dataset(coords={"time": pd.date_range("2025-01-01", periods=2)}), []
        )


@pytest.mark.slow
@pytest.mark.parametrize("product", ["final", "preliminary"])
def test_live_directory_listing(product: ChirpsProduct) -> None:
    times = pd.date_range("2025-01-01", periods=3)
    coords = _job(product).generate_source_file_coords(
        xr.Dataset(coords={"time": times}), []
    )
    assert [coord.time for coord in coords] == list(times)


def test_read_data_masks_fill_value_and_converts_to_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = np.full((GRID_LAT_SIZE, GRID_LON_SIZE), 36.0, dtype=np.float32)
    raw[0, 0] = np.float32(SOURCE_FILL_VALUE)

    reader = MagicMock()
    reader.read.side_effect = lambda *args, **kwargs: raw.copy()
    reader.__enter__ = lambda self: self
    reader.__exit__ = lambda self, *args: None
    monkeypatch.setattr("rasterio.open", lambda _path: reader)

    job = _job()
    (precip,) = job.data_vars
    coord = replace(
        UcsbChcChirpsAnalysisSourceFileCoord(
            product="final", time=pd.Timestamp("1981-01-01")
        ),
        downloaded_path=Path("chirps.tif"),
    )

    data = job.read_data(coord, precip)

    assert data.shape == (GRID_LAT_SIZE, GRID_LON_SIZE)
    assert np.isnan(data[0, 0])
    np.testing.assert_allclose(data[0, 1], 36.0 * MM_PER_DAY_TO_KG_M2_S, rtol=1e-6)
    assert np.isfinite(data).mean() > 0.999

    other_data = job.read_data(coord, replace(precip, name="other"))
    assert np.isnan(other_data[0, 0])
    np.testing.assert_allclose(other_data[0, 1], 36.0, rtol=1e-6)


def test_downloads_share_one_per_second_limit_including_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    now = 0.0
    starts: list[float] = []

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    def get(url: str, headers: dict[str, str] | None = None) -> httpx.Response:
        starts.append(now)
        return httpx.Response(
            503 if len(starts) == 1 else 200,
            request=httpx.Request("GET", url),
            content=b'<a href="../">Parent</a>' if url.endswith("/") else b"tiff",
        )

    monkeypatch.setattr(download.time, "monotonic", lambda: now)
    monkeypatch.setattr(download.time, "sleep", sleep)
    monkeypatch.setattr(download, "exponential_backoff_time", lambda _: 0.0)
    monkeypatch.setattr(download, "DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(download, "_httpx_client", lambda: MagicMock(get=get))
    assert region_job.chirps_rate_limiter._rate == 1.0
    assert region_job.chirps_rate_limiter._burst == 1
    monkeypatch.setattr(
        region_job, "chirps_rate_limiter", RateLimiter(max_per_minute=60)
    )

    for product in ("final", "preliminary"):
        job = _job(product)
        assert job.download_parallelism == 1
        coord = UcsbChcChirpsAnalysisSourceFileCoord(
            product=product, time=pd.Timestamp("2025-06-15")
        )
        assert job.download_file(coord).read_bytes() == b"tiff"

    region_job._list_directory_urls("https://example.org/list/")
    assert starts == [0.0, 1.0, 2.0, 3.0]
    now += 60.0
    job.download_file(coord)
    job.download_file(coord)
    assert starts[-2:] == [63.0, 64.0]
