import bz2
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from obstore.exceptions import GenericError
from rasterio.io import MemoryFile

from reformatters.common.download import http_download_to_disk
from reformatters.common.iterating import item
from reformatters.common.logging import get_logger
from reformatters.common.region_job import (
    CoordinateValueOrRange,
    RegionJob,
    SourceFileCoord,
)
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import (
    AppendDim,
    ArrayFloat32,
    DatetimeLike,
    Dim,
    Timedelta,
    Timestamp,
)

from .template_config import DwdIconEuDataVar

log = get_logger(__name__)


class DwdIconEuForecastSourceFileCoord(SourceFileCoord):
    """Coordinates of a single source file to process.

    Note that, unlike NOAA's NWPs, ICON-EU is published as one GRIB2 file per variable.
    """

    init_time: Timestamp
    lead_time: Timedelta
    variable_name_in_filename: str

    def get_url(self) -> str:
        """Return URL to .grib2.bz2 files on Dynamical.org's public archive of ICON-EU hosted on Source Co-Op."""
        init_time_str = self.init_time.strftime("%Y-%m-%dT%HZ")
        return (
            "https://source.coop/dynamical/dwd-icon-grib/icon-eu/regular-lat-lon/"
            f"{init_time_str}/{self.variable_name_in_filename}/"
        ) + self._get_basename()

    def get_fallback_url(self) -> str:
        """Return URLs to .grib2.bz2 files on DWD's HTTP server.

        Note that DWD may change their directory structure for ICON-EU. See this comment for
        details: https://github.com/dynamical-org/reformatters/issues/183#issuecomment-3365068327
        """
        # Example DWD URL:
        # https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/alb_rad/icon-eu_europe_regular-lat-lon_single-level_2025090700_000_ALB_RAD.grib2.bz2

        init_hour_str = self.init_time.strftime("%H")
        return (
            "https://opendata.dwd.de/weather/nwp/icon-eu/grib/"
            f"{init_hour_str}/{self.variable_name_in_filename}/"
        ) + self._get_basename()

    def _get_basename(self) -> str:
        """Note that this only handles single-level variables."""
        init_time_str: str = self.init_time.strftime("%Y%m%d%H")
        lead_time_hours: int = whole_hours(self.lead_time)
        return (
            f"icon-eu_europe_regular-lat-lon_single-level_{init_time_str}_{lead_time_hours:03d}_"
            f"{self.variable_name_in_filename.upper()}.grib2.bz2"
        )

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        """Return the output location for this file's data in the dataset."""
        # Map to the standard dimension names used in the template
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
        }


class DwdIconEuForecastRegionJob(
    RegionJob[DwdIconEuDataVar, DwdIconEuForecastSourceFileCoord]
):
    @classmethod
    def source_groups(
        cls,
        data_vars: Sequence[DwdIconEuDataVar],
    ) -> Sequence[Sequence[DwdIconEuDataVar]]:
        """Return a sequence of one-element sequences `[[var1], [var2], ...]`.

        In the `reformatters` API, this function can be used to return groups of variables,
        where all variables in a group can be retrieved from the same source file.

        But, in the case of ICON-EU, each grib file from DWD holds only one variable. So there's
        no ability/benefit from downloading multiple variables from the same file at the same time.
        """
        return [[var] for var in data_vars]

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[DwdIconEuDataVar],
    ) -> Sequence[DwdIconEuForecastSourceFileCoord]:
        """Return a sequence of coords, one for each source file required to process the data
        covered by processing_region_ds.

        Parameters
        ----------
        processing_region_ds : xr.Dataset
        data_var_group : Sequence[DwdIconEuDataVar]
            A sequence containing a single `DwdIconEuDataVar` (because each ICON-EU grib file
            contains a single variable).
        """
        init_times = pd.to_datetime(processing_region_ds["init_time"].values)
        lead_times = pd.to_timedelta(processing_region_ds["lead_time"].values)
        variable_name_in_filename = item(
            data_var_group
        ).internal_attrs.variable_name_in_filename

        # Sanity checks
        assert len(init_times) > 0
        assert len(lead_times) > 0

        return [
            DwdIconEuForecastSourceFileCoord(
                init_time=init_time,
                lead_time=lead_time,
                variable_name_in_filename=variable_name_in_filename,
            )
            for init_time in init_times
            for lead_time in lead_times
        ]

    def download_file(self, coord: DwdIconEuForecastSourceFileCoord) -> Path:
        """Download the file for the given coordinate and return the local path.

        Downloads the `.grib.bz2` file and returns its local `Path`.
        """
        url = coord.get_url()
        try:
            bz2_file_path = http_download_to_disk(url, self.dataset_id)
        except (FileNotFoundError, GenericError) as e:
            log.debug(f"Failed to download '{url}': {e}")
            fallback_url = coord.get_fallback_url()
            log.debug(f"Attempting to download from {fallback_url=}")
            bz2_file_path = http_download_to_disk(fallback_url, self.dataset_id)
        return bz2_file_path

    def read_data(
        self,
        coord: DwdIconEuForecastSourceFileCoord,
        data_var: DwdIconEuDataVar,
    ) -> ArrayFloat32:
        """Read and return an array of data for the given variable and source file coordinate."""
        assert coord.downloaded_path is not None  # for type check, system guarantees it

        with bz2.open(coord.downloaded_path, "rb") as f:
            grib_data = f.read()

        with MemoryFile(grib_data) as memfile, memfile.open() as reader:
            assert reader.count == 1, (
                f"Expected exactly 1 element in each ICON-EU grib file, found {reader.count=}. "
                f"{data_var.internal_attrs.variable_name_in_filename=}, "
                f"{coord.downloaded_path=}"
            )
            result: ArrayFloat32 = reader.read(indexes=1, out_dtype=np.float32)
        return result

    # Implement this to apply transformations to the array (e.g. deaccumulation)
    #
    # def apply_data_transformations(
    #     self, data_array: xr.DataArray, data_var: DwdIconEuDataVar
    # ) -> None:
    #     """
    #     Apply in-place data transformations to the output data array for a given data variable.
    #
    #     This method is called after reading all data for a variable into the shared-memory array,
    #     and before writing shards to the output store. The default implementation applies binary
    #     rounding to float32 arrays if `data_var.internal_attrs.keep_mantissa_bits` is set.
    #
    #     Subclasses may override this method to implement additional transformations such as
    #     deaccumulation, interpolation or other custom logic. All transformations should be
    #     performed in-place (don't copy `data_array`, it's large).
    #
    #     Parameters
    #     ----------
    #     data_array : xr.DataArray
    #         The output data array to be transformed in-place.
    #     data_var : DwdIconEuDataVar
    #         The data variable metadata object, which may contain transformation parameters.
    #     """
    #     super().apply_data_transformations(data_array, data_var)

    def update_template_with_results(
        self, process_results: Mapping[str, Sequence[DwdIconEuForecastSourceFileCoord]]
    ) -> xr.Dataset:
        """Update template dataset based on processing results. This method is called during
        operational updates.

        Subclasses should implement this method to apply dataset-specific adjustments
        based on the processing results. Examples include:
        - Trimming dataset along append_dim to only include successfully processed data
        - Loading existing coordinate values from the primary store and updating them based on results
        - Updating metadata based on what was actually processed vs what was planned

        The default implementation trims along append_dim to end at the most recent successfully
        processed coordinate (timestamp).

        Parameters
        ----------
        process_results : Mapping[str, Sequence[DwdIconEuForecastSourceFileCoord]]
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
        all_data_vars: Sequence[DwdIconEuDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence["RegionJob[DwdIconEuDataVar, DwdIconEuForecastSourceFileCoord]"],
        xr.Dataset,
    ]:
        """Return the sequence of RegionJob instances necessary to update the dataset from its
        current state to include the latest available data.

        Also return the template_ds, expanded along append_dim through the end of the data to
        process. The dataset returned here may extend beyond the available data at the source, in
        which case `update_template_with_results` will trim the dataset to the actual data
        processed.

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
        all_data_vars : Sequence[DwdIconEuDataVar]
            Sequence of all data variable configs for this dataset.
        reformat_job_name : str
            The name of the reformatting job, used for progress tracking.
            This is often the name of the Kubernetes job, or "local".

        Returns
        -------
        Sequence[RegionJob[DwdIconEuDataVar, DwdIconEuForecastSourceFileCoord]]
            RegionJob instances that need processing for operational updates.
        xr.Dataset
            The template_ds for the operational update.
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
