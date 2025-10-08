from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from urllib.parse import urlparse

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

_DOWNLOAD_TIMEOUT_SECONDS = 300
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
            log.info(f"Downloading {url}")
            response = session.get(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS, stream=True)
            response.raise_for_status()

            with open(local_path, "wb") as f:
                f.writelines(response.iter_content(chunk_size=8192))

            log.info(f"Successfully downloaded to {local_path}")
            return local_path

        return retry(_download, max_attempts=10)

    def read_data(
        self,
        coord: NasaSmapLevel336KmV9SourceFileCoord,
        data_var: NasaSmapDataVar,
    ) -> ArrayFloat32:
        """Read and return an array of data for the given variable and source file coordinate."""
        # Use rasterio to read coord.downloaded_path
        raise NotImplementedError()

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

        Also return the template_ds, expanded along append_dim through the end of
        the data to process. The dataset returned here may extend beyond the
        available data at the source, in which case `update_template_with_results`
        will trim the dataset to the actual data processed.

        The exact logic is dataset-specific, but it generally follows this pattern:
        1. Figure out the range of time to process: append_dim_start (inclusive) and append_dim_end (exclusive)
            a. Read existing data from the primary store to determine what's already processed
            b. Optionally identify recent incomplete/non-final data for reprocessing
        2. Call get_template_fn(append_dim_end) to get the template_ds
        3. Create RegionJob instances by calling cls.get_jobs(..., filter_start=append_dim_start)

        Parameters
        ----------
        primary_store : zarr.abc.store.Store
            The primary store to read existing data from and write updates to.
        tmp_store : Path
            The temporary Zarr store to write into while processing.
        get_template_fn : Callable[[DatetimeLike], xr.Dataset]
            Function to get the template_ds for the operational update.
        append_dim : AppendDim
            The dimension along which data is appended (e.g., "time").
        all_data_vars : Sequence[NasaSmapDataVar]
            Sequence of all data variable configs for this dataset.
        reformat_job_name : str
            The name of the reformatting job, used for progress tracking.
            This is often the name of the Kubernetes job, or "local".

        Returns
        -------
        Sequence[RegionJob[NasaSmapDataVar, NasaSmapLevel336KmV9SourceFileCoord]]
            RegionJob instances that need processing for operational updates.
        xr.Dataset
            The template_ds for the operational update.
        """
        # existing_ds = xr.open_zarr(primary_store)
        # append_dim_start = existing_ds[append_dim].max()
        # append_dim_end = pd.Timestamp.now()
        # template_ds = get_template_fn(append_dim_end)

        # jobs = cls.get_jobs(
        #     kind="operational-update",
        #     tmp_store=tmp_store,
        #     template_ds=template_ds,
        #     append_dim=append_dim,
        #     all_data_vars=all_data_vars,
        #     reformat_job_name=reformat_job_name,
        #     filter_start=append_dim_start,
        # )
        # return jobs, template_ds

        raise NotImplementedError(
            "Subclasses implement operational_update_jobs() with dataset-specific logic"
        )
