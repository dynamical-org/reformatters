from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.config_models import mask_source_fill_value_inplace
from reformatters.common.download import http_download_to_disk
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
from reformatters.ucsb_chc.chirps.chirps_config_models import (
    ChirpsProduct,
    UcsbChcChirpsDataVar,
)
from reformatters.ucsb_chc.chirps.template_config import (
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    MM_PER_DAY_TO_KG_M2_S,
)

_BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily"

# product -> (archive directory, filename infix)
_PRODUCT_PATHS: dict[ChirpsProduct, tuple[str, str]] = {
    "final": ("final/rnl", "rnl"),
    "preliminary": ("prelim/sat", "prelim"),
}


class UcsbChcChirpsAnalysisSourceFileCoord(SourceFileCoord):
    """One CHIRPS daily GeoTIFF for a given product and time."""

    product: ChirpsProduct
    time: Timestamp

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"time": self.time}

    def get_url(self) -> str:
        directory, infix = _PRODUCT_PATHS[self.product]
        return (
            f"{_BASE_URL}/{directory}/{self.time:%Y}/"
            f"chirps-v3.0.{infix}.{self.time:%Y.%m.%d}.tif"
        )


class UcsbChcChirpsAnalysisMaterializedRegionJob(
    MaterializedRegionJob[UcsbChcChirpsDataVar, UcsbChcChirpsAnalysisSourceFileCoord]
):
    product: ChirpsProduct

    download_parallelism: int = 8

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[UcsbChcChirpsDataVar],  # noqa: ARG002
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
        data_var: UcsbChcChirpsDataVar,
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

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[UcsbChcChirpsDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[RegionJob[UcsbChcChirpsDataVar, UcsbChcChirpsAnalysisSourceFileCoord]],
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
