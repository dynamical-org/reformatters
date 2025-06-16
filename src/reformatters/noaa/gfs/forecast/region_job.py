import hashlib
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import xarray as xr
import zarr

from reformatters.common.download import DOWNLOAD_DIR, download_to_disk, http_store
from reformatters.common.logging import get_logger
from reformatters.common.region_job import (
    RegionJob,
)
from reformatters.common.types import (
    AppendDim,
    ArrayFloat32,
    DatetimeLike,
)
from reformatters.noaa.gfs.models import NoaaGfsSourceFileCoord
from reformatters.noaa.models import NoaaDataVar
from reformatters.noaa.noaa_utils import has_hour_0_values

# Accumulations reset every 6 hours
GFS_ACCUMULATION_RESET_HOURS = 6


def digest(data, length: int = 8) -> str:
    """Consistent, likely collision-free string digest of one or more strings."""
    message = hashlib.sha256()
    for string in data:
        message.update(string.encode())
    return message.hexdigest()[:length]


log = get_logger(__name__)


class NoaaGfsForecastSourceFileCoord(NoaaGfsSourceFileCoord):
    # We share the name attributes (init_time, lead_time) with
    # NoaaGfsSourceFileCoord, and the same.
    # The default out_loc implementation of
    # {"init_time": self.init_time, "lead_time": self.lead_time}
    # is correct for this dataset.
    pass


class NoaaGfsForecastRegionJob(RegionJob[NoaaDataVar, NoaaGfsForecastSourceFileCoord]):
    @classmethod
    def source_groups(
        cls,
        data_vars: Sequence[NoaaDataVar],
    ) -> Sequence[Sequence[NoaaDataVar]]:
        """
        Return groups of variables, where all variables in a group can be retrieived from the same source file.

        By separating variables that have hour 0 values from those that don't here,
        generate_source_file coords can skip attempting to download hour 0 data,
        reducing error log noise.
        """
        grouped = defaultdict(list)
        for data_var in data_vars:
            grouped[has_hour_0_values(data_var)].append(data_var)
        return list(grouped.values())

    def generate_source_file_coords(
        self, processing_region_ds: xr.Dataset, data_var_group: Sequence[NoaaDataVar]
    ) -> Sequence[NoaaGfsForecastSourceFileCoord]:
        """Return a sequence of coords, one for each source file required to process the data covered by processing_region_ds."""
        return [
            NoaaGfsForecastSourceFileCoord(
                init_time=pd.Timestamp(init_time),
                lead_time=pd.Timedelta(lead_time),
                data_vars=data_var_group,
            )
            for init_time, lead_time in product(
                processing_region_ds["init_time"].values,
                processing_region_ds["lead_time"].values,
            )
        ]

    def download_file(self, coord: NoaaGfsForecastSourceFileCoord) -> Path:
        """Download the file for the given coordinate and return the local path."""
        # Download grib index file
        idx_url = f"{coord.get_url()}.idx"
        idx_local_path = http_download_to_disk(idx_url, self.dataset_id)
        index_contents = idx_local_path.read_text()

        lead_hours = int(coord.lead_time.total_seconds() / 3600)

        # Parse byte ranges from index
        starts: list[int] = []
        ends: list[int] = []
        for var in coord.data_vars:
            if lead_hours == 0:
                hours_str_prefix = ""
            elif var.attrs.step_type == "instant":
                hours_str_prefix = str(lead_hours)
            else:
                diff = lead_hours % GFS_ACCUMULATION_RESET_HOURS
                reset_hour = (
                    lead_hours - diff
                    if diff != 0
                    else lead_hours - GFS_ACCUMULATION_RESET_HOURS
                )
                hours_str_prefix = f"{reset_hour}-{lead_hours}"
            var_match_str = re.escape(
                f"{var.internal_attrs.grib_element}:{var.internal_attrs.grib_index_level}:{hours_str_prefix}"
            )
            matches = re.findall(
                rf"\d+:(\d+):.+:{var_match_str}.+:(?:\n\d+:(\d+))?",
                index_contents,
            )
            assert len(matches) == 1, (
                f"Expected exactly 1 match, found {matches}, {var.name} {idx_url}"
            )
            m0, m1 = matches[0]
            start = int(m0)
            end = int(m1) if m1 else start + 10 * (2**30)
            starts.append(start)
            ends.append(end)

        data_url = coord.get_url()
        parsed_url = urlparse(data_url)
        store = http_store(f"{parsed_url.scheme}://{parsed_url.netloc}")
        filename = Path(parsed_url.path).name
        suffix = digest(f"{s}-{e}" for s, e in zip(starts, ends, strict=False))
        filename_with_suffix = f"{filename}-{suffix}"
        local_path = DOWNLOAD_DIR / self.dataset_id / parsed_url.path.removeprefix("/")
        local_path = local_path.with_name(filename_with_suffix)
        download_to_disk(
            store,
            parsed_url.path,
            local_path,
            overwrite_existing=True,
            byte_ranges=(starts, ends),
        )
        return local_path

    def read_data(
        self,
        coord: NoaaGfsSourceFileCoord,
        data_var: NoaaDataVar,
    ) -> ArrayFloat32:
        """Read and return an array of data for the given variable and source file coordinate."""
        # with rasterio.open(coord.downloaded_file_path) as reader:
        #     TODO: make a band index based on tag matching utility function
        #     matching_indexes = [
        #         i
        #         for i in range(reader.count)
        #         if (tags := reader.tags(i))["GRIB_ELEMENT"]
        #         == data_var.internal_attrs.grib_element
        #         and tags["GRIB_COMMENT"] == data_var.internal_attrs.grib_comment
        #     ]
        #     assert len(matching_indexes) == 1, f"Expected exactly 1 matching band, found {matching_indexes}. {data_var.internal_attrs.grib_element=}, {data_var.internal_attrs.grib_description=}, {coord.downloaded_file_path=}"  # fmt: skip
        #     rasterio_band_index = 1 + matching_indexes[0]  # rasterio is 1-indexed
        #     return reader.read(rasterio_band_index, dtype=np.float32)
        raise NotImplementedError(
            "Read and return data for the given variable and source file coordinate."
        )

    # Implement this to apply transformations to the array (e.g. deaccumulation)
    #
    # def apply_data_transformations(
    #     self, data_array: xr.DataArray, data_var: NoaaDataVar
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
    #     data_var : NoaaDataVar
    #         The data variable metadata object, which may contain transformation parameters.
    #     """
    #     super().apply_data_transformations(data_array, data_var)

    def update_template_with_results(
        self, process_results: Mapping[str, Sequence[NoaaGfsSourceFileCoord]]
    ) -> xr.Dataset:
        """
        Update template dataset based on processing results. This method is called
        during operational updates.

        Subclasses should implement this method to apply dataset-specific adjustments
        based on the processing results. Examples include:
        - Trimming dataset along append_dim to only include successfully processed data
        - Loading existing coordinate values from final_store and updating them based on results
        - Updating metadata based on what was actually processed vs what was planned

        The default implementation trims along append_dim to end at the most recent
        successfully processed coordinate (timestamp).

        Parameters
        ----------
        process_results : Mapping[str, Sequence[NoaaGfsSourceFileCoord]]
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
        final_store: zarr.abc.store.Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[NoaaDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence["RegionJob[NoaaDataVar, NoaaGfsForecastSourceFileCoord]"], xr.Dataset
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
        all_data_vars : Sequence[NoaaDataVar]
            Sequence of all data variable configs for this dataset.
        reformat_job_name : str
            The name of the reformatting job, used for progress tracking.
            This is often the name of the Kubernetes job, or "local".

        Returns
        -------
        Sequence[RegionJob[NoaaDataVar, NoaaGfsSourceFileCoord]]
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
        # return jobs, temoplate_ds

        raise NotImplementedError(
            "Subclasses implement operational_update_jobs() with dataset-specific logic"
        )
