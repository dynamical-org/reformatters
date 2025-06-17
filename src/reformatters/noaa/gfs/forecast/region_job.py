import warnings
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio  # type: ignore
import xarray as xr
import zarr

from reformatters.common.download import (
    http_download_to_disk,
)
from reformatters.common.iterating import digest
from reformatters.common.logging import get_logger
from reformatters.common.region_job import (
    RegionJob,
)
from reformatters.common.types import (
    AppendDim,
    ArrayFloat32,
    DatetimeLike,
)
from reformatters.noaa.gfs.gfs_common import (
    NoaaGfsSourceFileCoord,
    parse_grib_index,
)
from reformatters.noaa.models import NoaaDataVar
from reformatters.noaa.noaa_utils import has_hour_0_values

log = get_logger(__name__)


def _get_grib_element_for_reading(
    data_var: NoaaDataVar, lead_time: pd.Timedelta
) -> str:
    """Get the GRIB element name for reading, including lead time suffix if needed."""
    grib_element = data_var.internal_attrs.grib_element
    if data_var.internal_attrs.include_lead_time_suffix:
        lead_hours = int(lead_time.total_seconds() / 3600)
        lead_hours_accum = lead_hours % GFS_ACCUMULATION_RESET_HOURS
        if lead_hours_accum == 0:
            lead_hours_accum = 6
        grib_element += str(lead_hours_accum).zfill(2)
    return grib_element


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

        # Download the grib messages for the data vars in the coord using byte ranges
        starts, ends = parse_grib_index(index_contents, coord)
        vars_suffix = digest(f"{s}-{e}" for s, e in zip(starts, ends, strict=True))
        return http_download_to_disk(
            coord.get_url(),
            self.dataset_id,
            byte_ranges=(starts, ends),
            local_path_suffix=vars_suffix,
        )

    def read_data(
        self,
        coord: NoaaGfsSourceFileCoord,
        data_var: NoaaDataVar,
    ) -> ArrayFloat32:
        """Read and return an array of data for the given variable and source file coordinate."""
        grib_element = _get_grib_element_for_reading(data_var, coord.lead_time)
        # inline the grib element for reading logic here  AI!
        grib_description = data_var.internal_attrs.grib_description

        with warnings.catch_warnings(), rasterio.open(coord.downloaded_path) as reader:
            matching_bands = [
                rasterio_band_i
                for band_i in range(reader.count)
                if reader.descriptions[band_i] == grib_description
                and reader.tags(rasterio_band_i := band_i + 1)["GRIB_ELEMENT"]
                == grib_element
            ]

            assert len(matching_bands) == 1, (
                f"Expected exactly 1 matching band, found {matching_bands}. "
                f"{grib_element=}, {grib_description=}, {coord.downloaded_path=}"
            )
            rasterio_band_index = matching_bands[0]

            result: ArrayFloat32 = reader.read(
                rasterio_band_index,
                out_dtype=np.float32,
            )
            return result

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
