from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import rasterio  # type: ignore[import-untyped]
import xarray as xr
import zarr

from reformatters.common.download import get_local_path
from reformatters.common.logging import get_logger
from reformatters.common.region_job import (
    CoordinateValueOrRange,
    RegionJob,
    SourceFileCoord,
)
from reformatters.common.retry import retry
from reformatters.common.types import (
    AppendDim,
    ArrayFloat32,
    DatetimeLike,
    Dim,
    Timestamp,
)

from .earthdata_auth import get_authenticated_session
from .template_config import NasaSmapDataVar

log = get_logger(__name__)

_SOURCE_FILL_VALUE = -9999.0


class NasaSmapLevel336KmV9SourceFileCoord(SourceFileCoord):
    """Coordinates of a single source file to process."""

    time: Timestamp

    def get_url(self) -> str:
        base = "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/SMAP/SPL3SMP/009"
        year_month = self.time.strftime("%Y/%m")
        filename = f"SMAP_L3_SM_P_{self.time.strftime('%Y%m%d')}_R19240_001.h5"
        return f"{base}/{year_month}/{filename}"

    def out_loc(
        self,
    ) -> Mapping[Dim, CoordinateValueOrRange]:
        return {"time": self.time}


class NasaSmapLevel336KmV9RegionJob(
    RegionJob[NasaSmapDataVar, NasaSmapLevel336KmV9SourceFileCoord]
):
    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NasaSmapDataVar],
    ) -> Sequence[NasaSmapLevel336KmV9SourceFileCoord]:
        """Return a sequence of coords, one for each source file required to process the data covered by processing_region_ds."""
        return [
            NasaSmapLevel336KmV9SourceFileCoord(time=time)
            for time in processing_region_ds["time"].values
        ]

    def download_file(self, coord: NasaSmapLevel336KmV9SourceFileCoord) -> Path:
        """Download the file for the given coordinate and return the local path."""
        url = coord.get_url()
        relative_path = urlparse(url).path
        local_path = get_local_path(
            self.template_ds.attrs["dataset_id"], path=relative_path
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)

        def _download() -> Path:
            session = get_authenticated_session()
            response = session.get(url, timeout=10, stream=True, allow_redirects=True)
            response.raise_for_status()

            with open(local_path, "wb") as f:
                f.writelines(response.iter_content(chunk_size=8192))

            return local_path

        return retry(_download, max_attempts=10)

    def read_data(
        self,
        coord: NasaSmapLevel336KmV9SourceFileCoord,
        data_var: NasaSmapDataVar,
    ) -> ArrayFloat32:
        """Read and return an array of data for the given variable and source file coordinate."""
        assert coord.downloaded_path is not None, "File must be downloaded first"

        subdataset_path = (
            f"HDF5:{coord.downloaded_path}:{data_var.internal_attrs.h5_path}"
        )

        with rasterio.open(subdataset_path) as reader:
            data: ArrayFloat32 = reader.read(1, out_dtype=np.float32)

        data[data == _SOURCE_FILL_VALUE] = np.nan

        return data

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: zarr.abc.store.Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[NasaSmapDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence["RegionJob[NasaSmapDataVar, NasaSmapLevel336KmV9SourceFileCoord]"],
        xr.Dataset,
    ]:
        """
        Return the sequence of RegionJob instances necessary to update the dataset
        from its current state to include the latest available data.
        """
        existing_ds = xr.open_zarr(primary_store)
        append_dim_start = existing_ds[append_dim].max()
        append_dim_end = pd.Timestamp.now()
        template_ds = get_template_fn(append_dim_end)

        jobs = cls.get_jobs(
            kind="operational-update",
            tmp_store=tmp_store,
            template_ds=template_ds,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            filter_start=append_dim_start,
        )
        return jobs, template_ds
