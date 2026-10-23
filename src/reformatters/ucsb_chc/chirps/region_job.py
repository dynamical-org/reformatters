import re
from collections.abc import Callable, Mapping, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.config_models import (
    BaseInternalAttrs,
    DataVar,
    mask_source_fill_value_inplace,
)
from reformatters.common.download import RateLimiter, httpx_download_to_disk, httpx_get
from reformatters.common.logging import get_logger
from reformatters.common.materialized_region_job import MaterializedRegionJob
from reformatters.common.region_job import (
    CoordinateValue,
    RegionJob,
    SourceFileCoord,
)
from reformatters.common.types import (
    AppendDim,
    ArrayFloat32,
    DatetimeLike,
    Dim,
    Timestamp,
)
from reformatters.ucsb_chc.chirps.chirps_config_models import ChirpsProduct
from reformatters.ucsb_chc.chirps.template_config import (
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    MM_PER_DAY_TO_KG_M2_S,
)

chirps_rate_limiter = RateLimiter(max_per_minute=60)
log = get_logger(__name__)

_BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily"


class _ProductPath(NamedTuple):
    directory: str
    filename_infix: str


_PRODUCT_PATHS: dict[ChirpsProduct, _ProductPath] = {
    "final": _ProductPath("final/rnl", "rnl"),
    "preliminary": _ProductPath("prelim/sat", "prelim"),
}


class _DirectoryLinks(HTMLParser):
    def __init__(self, directory_url: str) -> None:
        super().__init__()
        self.directory_url = directory_url
        self.urls: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value is not None:
                    self.urls.add(urljoin(self.directory_url, value))


def _list_directory_urls(directory_url: str) -> frozenset[str]:
    response = httpx_get(directory_url, rate_limiter=chirps_rate_limiter)
    parser = _DirectoryLinks(directory_url)
    parser.feed(response.text)
    assert urljoin(directory_url, "../") in parser.urls, (
        f"Directory index has no parent link: {directory_url}"
    )
    return frozenset(parser.urls)


def _available_file_urls(
    product: ChirpsProduct, requested_years: Sequence[int], current_year: int
) -> set[str]:
    product_url = f"{_BASE_URL}/{_PRODUCT_PATHS[product].directory}/"
    years = _list_directory_urls(product_url)
    assert any(
        re.fullmatch(re.escape(product_url) + r"\d{4}/", url) for url in years
    ), f"No year directories in {product_url}"
    available: set[str] = set()
    for year in dict.fromkeys(requested_years):
        directory = f"{product_url}{year}/"
        if directory not in years:
            assert year >= current_year, f"Missing historical directory: {directory}"
            continue
        pattern = (
            re.escape(
                f"{directory}chirps-v3.0.{_PRODUCT_PATHS[product].filename_infix}.{year}."
            )
            + r"\d{2}\.\d{2}\.tif"
        )
        files = {
            url for url in _list_directory_urls(directory) if re.fullmatch(pattern, url)
        }
        assert files or year >= current_year, f"No CHIRPS files in {directory}"
        if not files:
            log.warning(f"No CHIRPS files in {directory}")
        available.update(files)
    return available


class UcsbChcChirpsAnalysisSourceFileCoord(SourceFileCoord):
    """One CHIRPS daily GeoTIFF for a given product and time."""

    product: ChirpsProduct
    time: Timestamp

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"time": self.time}

    def get_url(self) -> str:
        path = _PRODUCT_PATHS[self.product]
        return (
            self.directory_url()
            + f"chirps-v3.0.{path.filename_infix}.{self.time:%Y.%m.%d}.tif"
        )

    def directory_url(self) -> str:
        return f"{_BASE_URL}/{_PRODUCT_PATHS[self.product].directory}/{self.time:%Y}/"


class UcsbChcChirpsAnalysisMaterializedRegionJob(
    MaterializedRegionJob[
        DataVar[BaseInternalAttrs], UcsbChcChirpsAnalysisSourceFileCoord
    ]
):
    product: ChirpsProduct

    download_parallelism: int = 1

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[DataVar[BaseInternalAttrs]],  # noqa: ARG002
    ) -> Sequence[UcsbChcChirpsAnalysisSourceFileCoord]:
        coords = [
            UcsbChcChirpsAnalysisSourceFileCoord(
                product=self.product, time=pd.Timestamp(time)
            )
            for time in processing_region_ds["time"].values
        ]
        if not coords:
            return []
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        available = _available_file_urls(
            self.product, [coord.time.year for coord in coords], now.year
        )
        missing_old = [
            coord.time
            for coord in coords
            if coord.get_url() not in available
            and coord.time < now - self.expected_unavailable_window
        ]
        if missing_old:
            log.error(
                f"CHIRPS {self.product} source listing is missing historical dates: {missing_old}"
            )
        return [coord for coord in coords if coord.get_url() in available]

    def download_file(self, coord: UcsbChcChirpsAnalysisSourceFileCoord) -> Path:
        return httpx_download_to_disk(
            coord.get_url(), self.dataset_id, rate_limiter=chirps_rate_limiter
        )

    def read_data(
        self,
        coord: UcsbChcChirpsAnalysisSourceFileCoord,
        data_var: DataVar[BaseInternalAttrs],
    ) -> ArrayFloat32:
        assert coord.downloaded_path is not None, "File must be downloaded first"
        with rasterio.open(coord.downloaded_path) as reader:
            data: ArrayFloat32 = reader.read(1, out_dtype=np.float32)

        assert data.shape == (GRID_LAT_SIZE, GRID_LON_SIZE), (
            f"unexpected source grid shape {data.shape} in {coord.get_url()}"
        )
        mask_source_fill_value_inplace(data, data_var.internal_attrs)
        if data_var.name == "precipitation_surface":
            data *= np.float32(MM_PER_DAY_TO_KG_M2_S)
        return data

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[DataVar[BaseInternalAttrs]],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[
            RegionJob[DataVar[BaseInternalAttrs], UcsbChcChirpsAnalysisSourceFileCoord]
        ],
        xr.DataTree,
    ]:
        existing_ds = xr.open_zarr(primary_store, chunks=None)
        append_dim_start = pd.Timestamp(existing_ds[append_dim].max().item())
        append_dim_end = pd.Timestamp.now()
        template_ds = get_template_fn(append_dim_end)

        jobs = cls.get_jobs(
            tmp_store=tmp_store,
            template_ds=template_ds,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            filter_start=append_dim_start,
        )
        return jobs, template_ds
