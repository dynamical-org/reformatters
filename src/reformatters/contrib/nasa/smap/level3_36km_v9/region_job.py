from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np
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
    ArrayFloat32,
    DatetimeLike,
    Dim,
)

from .template_config import NasaSmapDataVar

log = get_logger(__name__)

FILL_VALUE = -9999.0


class NasaSmapLevel336KmV9SourceFileCoord(SourceFileCoord):
    """Coordinates of a single source file to process."""

    time: DatetimeLike

    def get_url(self) -> str:
        # URL pattern based on NSIDC: https://n5eil01u.ecs.nsidc.org/SMAP/SPL3SMP.009/YYYY.MM.DD/SMAP_L3_SM_P_YYYYMMDD_R19240_001.h5
        date_str = self.time.strftime("%Y.%m.%d")
        filename = f"SMAP_L3_SM_P_{self.time.strftime('%Y%m%d')}_R19240_001.h5"
        return f"https://n5eil01u.ecs.nsidc.org/SMAP/SPL3SMP.009/{date_str}/{filename}"

    def out_loc(
        self,
    ) -> Mapping[Dim, CoordinateValueOrRange]:
        return {"time": self.time}


class NasaSmapLevel336KmV9RegionJob(
    RegionJob[NasaSmapDataVar, NasaSmapLevel336KmV9SourceFileCoord]
):
    # Optionally, limit the number of variables downloaded together.
    # If set to a value less than len(data_vars), downloading, reading/recompressing,
    # and uploading steps will be pipelined within a region job.
    # 5 is a reasonable default if it is possible to download less than all
    # variables in a single file (e.g. you have a grib index).
    # Leave unset if you have to download a whole file to get one variable out
    # to avoid re-downloading the same file multiple times.
    #
    # max_vars_per_download_group: ClassVar[int | None] = None

    # Implement this method only if different variables must be retrieved from different urls
    #
    # # @classmethod
    # def source_groups(
    #     cls,
    #     data_vars: Sequence[NasaSmapDataVar],
    # ) -> Sequence[Sequence[NasaSmapDataVar]]:
    #     """
    #     Return groups of variables, where all variables in a group can be retrieived from the same source file.
    #     """
    #     grouped = defaultdict(list)
    #     for data_var in data_vars:
    #         grouped[data_var.internal_attrs.file_type].append(data_var)
    #     return list(grouped.values())

    # Implement this method only if specific post processing in this dataset
    # requires data from outside the region defined by self.region,
    # e.g. for deaccumulation or interpolation along append_dim in an analysis dataset.
    #
    # def get_processing_region(self) -> slice:
    #     """
    #     Return a slice of integer offsets into self.template_ds along self.append_dim that identifies
    #     the region to process. In most cases this is exactly self.region, but if additional data outside
    #     the region is required, for example for correct interpolation or deaccumulation, this method can
    #     return a modified slice (e.g. `slice(self.region.start - 1, self.region.stop + 1)`).
    #     """
    #     return self.region

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
        return http_download_to_disk(coord.get_url(), self.dataset_id)

    def read_data(
        self,
        coord: NasaSmapLevel336KmV9SourceFileCoord,
        data_var: NasaSmapDataVar,
    ) -> ArrayFloat32:
        """Read and return an array of data for the given variable and source file coordinate."""
        with h5py.File(coord.downloaded_file_path, "r") as f:
            data = f[data_var.internal_attrs.h5_path][:]
            # Convert -9999 to NaN
            data = np.where(data == FILL_VALUE, np.nan, data).astype(np.float32)
        return data

    # Implement this to apply transformations to the array (e.g. deaccumulation)
    #
    # def apply_data_transformations(
    #     self, data_array: xr.DataArray, data_var: NasaSmapDataVar
    # ) -> None:
    #     """
    #     Apply in-place data transformations to the output data array for a given data variable.

    #     This method is called after reading all data for a variable into the shared-memory array,
    #     and before writing shards to the output store. The default implementation applies binary
    #     rounding to float32 arrays if `data_var.internal_attrs.keep_mantissa_bits` is set.

    #     Subclasses may override this method to implement additional transformations such as
    #     deaccumulation, interpolation or other custom logic. All transformations should be
    #     performed in-place (don't copy `data_array`, it's large).

    #     Parameters
    #     ----------
    #     data_array : xr.DataArray
    #         The output data array to be transformed in-place.
    #     data_var : NasaSmapDataVar
    #         The data variable metadata object, which may contain transformation parameters.
    #     """
    #     super().apply_data_transformations(data_array, data_var)

    def update_template_with_results(
        self,
        process_results: Mapping[str, Sequence[NasaSmapLevel336KmV9SourceFileCoord]],
    ) -> xr.Dataset:
        """
        Update template dataset based on processing results. This method is called
        during operational updates.

        Subclasses should implement this method to apply dataset-specific adjustments
        based on the processing results. Examples include:
        - Trimming dataset along append_dim to end at the most recent
            successfully processed coordinate (timestamp)
        - Loading existing coordinate values from the primary store and updating them based on results
        - Updating metadata based on what was actually processed vs what was planned

        The default implementation trims along append_dim to end at the most recent
        successfully processed coordinate (timestamp).

        Parameters
        ----------
        process_results : Mapping[str, Sequence[NasaSmapLevel336KmV9SourceFileCoord]]
            Mapping from variable names to their source file coordinates with final processing status.

        Returns
        -------
        xr.Dataset
            Updated template dataset reflecting the actual processing results.
        """
        # The super() implementation looks like this:
        #
        # max_append_dim_processed = max(
        #     (
        #         c.out_loc()[self.append_dim]  # type: ignore[type-var]
        #         for c in chain.from_iterable(process_results.values())
        #         if c.status == SourceFileStatus.Succeeded
        #     ),
        #     default=None,
        # )
        # if max_append_dim_processed is None:
        #     # No data was processed, trim the template to stop before this job's region
        #     # This is using isel's exclusive slice end behavior
        #     return self.template_ds.isel(
        #         {self.append_dim: slice(None, self.region.start)}
        #     )
        # else:
        #     return self.template_ds.sel(
        #         {self.append_dim: slice(None, max_append_dim_processed)}
        #     )
        #
        # If you like the above behavior, skip implementing this method.
        # If you need to customize the behavior, implement this method.

        raise NotImplementedError(
            "Subclasses implement update_template_with_results() with dataset-specific logic"
        )

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
