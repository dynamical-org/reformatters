from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

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
from reformatters.common.download import http_download_to_disk
from reformatters.common.materialized_region_job import MaterializedRegionJob
from reformatters.common.region_job import (
    CoordinateValue,
    RegionJob,
    SourceFileCoord,
    SourceFileResult,
    SourceFileStatus,
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

_BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily"


class _ProductPath(NamedTuple):
    directory: str
    filename_infix: str


_PRODUCT_PATHS: dict[ChirpsProduct, _ProductPath] = {
    "final": _ProductPath("final/rnl", "rnl"),
    "preliminary": _ProductPath("prelim/sat", "prelim"),
}


class UcsbChcChirpsAnalysisSourceFileCoord(SourceFileCoord):
    """One CHIRPS daily GeoTIFF for a given product and time."""

    product: ChirpsProduct
    time: Timestamp

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"time": self.time}

    def get_url(self) -> str:
        path = _PRODUCT_PATHS[self.product]
        return (
            f"{_BASE_URL}/{path.directory}/{self.time:%Y}/"
            f"chirps-v3.0.{path.filename_infix}.{self.time:%Y.%m.%d}.tif"
        )


class UcsbChcChirpsAnalysisMaterializedRegionJob(
    MaterializedRegionJob[
        DataVar[BaseInternalAttrs], UcsbChcChirpsAnalysisSourceFileCoord
    ]
):
    product: ChirpsProduct

    download_parallelism: int = 8

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[DataVar[BaseInternalAttrs]],  # noqa: ARG002
    ) -> Sequence[UcsbChcChirpsAnalysisSourceFileCoord]:
        return [
            UcsbChcChirpsAnalysisSourceFileCoord(
                product=self.product, time=pd.Timestamp(time)
            )
            for time in processing_region_ds["time"].values
        ]

    def download_file(self, coord: UcsbChcChirpsAnalysisSourceFileCoord) -> Path:
        return http_download_to_disk(coord.get_url(), self.dataset_id)

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
        data *= np.float32(MM_PER_DAY_TO_KG_M2_S)
        return data

    def update_template_with_results(
        self, process_results: Mapping[str, Sequence[SourceFileResult]]
    ) -> xr.DataTree:
        """Trim the template before the first day not successfully read."""
        # A day whose download failed is absent from the results, so a Succeeded
        # status is the only evidence a day was read.
        read_times = {
            result.out_loc[self.append_dim]
            for results in process_results.values()
            for result in results
            if result.status == SourceFileStatus.Succeeded
        }
        times = self.template_ds.coords[self.append_dim].values
        stop = self.region.start
        for i in range(self.region.start, len(times)):
            if pd.Timestamp(times[i]) not in read_times:
                break
            stop = i + 1
        return self.template_ds.isel({self.append_dim: slice(None, stop)})

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
        # Reprocess the store's newest day so the update always has at least one
        # available source file and trims, rather than publishing, the days beyond
        # the source's leading edge.
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
