from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import fsspec  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import rasterio  # type: ignore[import-untyped]
import xarray as xr
import zarr

from reformatters.common.download import http_download_to_disk
from reformatters.common.logging import get_logger
from reformatters.common.region_job import (
    CoordinateValueOrRange,
    RegionJob,
    SourceFileCoord,
)
from reformatters.common.types import (
    AppendDim,
    Array2D,
    ArrayFloat32,
    DatetimeLike,
    Dim,
    Timestamp,
)

from .template_config import NoaaNdviDataVar

log = get_logger(__name__)


class NoaaNdviAnalysisSourceFileCoord(SourceFileCoord):
    """Coordinates of a single source file to process."""

    time: Timestamp
    url: str

    def get_url(self) -> str:
        return self.url

    def out_loc(
        self,
    ) -> Mapping[Dim, CoordinateValueOrRange]:
        return {"time": self.time}


class NoaaNdviAnalysisRegionJob(
    RegionJob[NoaaNdviDataVar, NoaaNdviAnalysisSourceFileCoord]
):
    download_parallelism: int = 2
    # We observed deadlocks when using more than 2 threads to read data into shared memory.
    read_parallelism: int = 1

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaNdviDataVar],
    ) -> Sequence[NoaaNdviAnalysisSourceFileCoord]:
        """Return a sequence of coords, one for each source file required to process the data covered by processing_region_ds.

        The names for this dataset include the processing timestamp, which means that we cannot determine the filename and URL
        ourselves. When generating SourceFileCoords we need to enumerate the available files and match them to each time we are
        are attempting to process.

        The source data for this dataset is stored and available from a variety of sources. See the "Data Access" section of
        https://www.ncei.noaa.gov/products/climate-data-records/normalized-difference-vegetation-index. We are currently using
        the NOAA S3 bucket.
        """
        # set anon to True to not require AWS credentials, as this is a public bucket.
        fs = fsspec.filesystem("s3", anon=True)
        public_base_url = "https://noaa-cdr-ndvi-pds.s3.amazonaws.com/data"

        times = processing_region_ds["time"].values

        years = {pd.Timestamp(t).year for t in times}
        print("Listing files for Years", years)
        available_files_by_year = {
            year: fs.ls(f"noaa-cdr-ndvi-pds/data/{year}") for year in years
        }

        def _get_url(time: Timestamp) -> str:
            timestamp = pd.Timestamp(time)
            year = timestamp.year
            year_date_month_str = timestamp.strftime("%Y%m%d")
            for filepath in available_files_by_year[year]:
                _, date, _ = filepath.rsplit("_", 2)
                if year_date_month_str == date:
                    filename = Path(filepath).name
                    url = f"{public_base_url}/{year}/{filename}"
                    return url
            raise ValueError(f"No file found for {time}")

        print("Returning source file coords")
        return [NoaaNdviAnalysisSourceFileCoord(time=t, url=_get_url(t)) for t in times]

    def download_file(self, coord: NoaaNdviAnalysisSourceFileCoord) -> Path:
        """Download the file for the given coordinate and return the local path."""
        return http_download_to_disk(coord.get_url(), self.dataset_id)

    def read_data(
        self,
        coord: NoaaNdviAnalysisSourceFileCoord,
        data_var: NoaaNdviDataVar,
    ) -> ArrayFloat32:
        """Read and return an array of data for the given variable and source file coordinate."""
        var_name = data_var.internal_attrs.netcdf_var_name
        netcdf_path = f"netcdf:{coord.downloaded_path}:{var_name}"
        band = 1  # because rasterio netcdf requires selecting the band in the file path we always want band 1
        print("Opening netcdf file", netcdf_path)
        with rasterio.open(netcdf_path) as reader:
            print("Beginning read", netcdf_path)
            result: Array2D[np.float32] = reader.read(band, out_dtype=np.float32)
            print("Got Result", netcdf_path)
            result[result == data_var.internal_attrs.fill_value] = np.nan
            assert result.shape == (3600, 7200)
            return result

    @classmethod
    def operational_update_jobs(
        cls,
        final_store: zarr.abc.store.Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[NoaaNdviDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence["RegionJob[NoaaNdviDataVar, NoaaNdviAnalysisSourceFileCoord]"],
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
            a. Read existing data from final_store to determine what's already processed
            b. Optionally identify recent incomplete/non-final data for reprocessing
        2. Call get_template_fn(append_dim_end) to get the template_ds
        3. Create RegionJob instances by calling cls.get_jobs(..., filter_start=append_dim_start)

        Parameters
        ----------
        final_store : zarr.abc.store.Store
            The destination Zarr store to read existing data from and write updates to.
        tmp_store : zarr.abc.store.Store | Path
            The temporary Zarr store to write into while processing.
        get_template_fn : Callable[[DatetimeLike], xr.Dataset]
            Function to get the template_ds for the operational update.
        append_dim : AppendDim
            The dimension along which data is appended (e.g., "time").
        all_data_vars : Sequence[NoaaNdviDataVar]
            Sequence of all data variable configs for this dataset.
        reformat_job_name : str
            The name of the reformatting job, used for progress tracking.
            This is often the name of the Kubernetes job, or "local".

        Returns
        -------
        Sequence[RegionJob[NoaaNdviDataVar, NoaaNdviAnalysisSourceFileCoord]]
            RegionJob instances that need processing for operational updates.
        xr.Dataset
            The template_ds for the operational update.
        """
        # existing_ds = xr.open_zarr(final_store)
        # append_dim_start = existing_ds[append_dim].max()
        # append_dim_end = pd.Timestamp.now()
        # template_ds = get_template_fn(append_dim_end)

        # jobs = cls.get_jobs(
        #     kind="operational-update",
        #     final_store=final_store,
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
