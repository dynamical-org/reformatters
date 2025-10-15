import itertools
from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import rasterio  # type: ignore[import-untyped]
import xarray as xr
import zarr

from reformatters.common.download import (
    http_download_to_disk,
)
from reformatters.common.iterating import digest
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

from .template_config import EcmwfIfsEnsDataVar

log = get_logger(__name__)

"""
Region = single init_time (1 along append_dim axis), full zarr slice for that init_time. Can be one var.
   - strictly about coordinates, agnostic of data vars.
   -> allows us to never insert, only append.
   - full ensemble members, full lead times


Tues afternoon:
- implement byte ranges
- region job tests?
- read_data implementation
- try run backfill-local??
"""


class EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord(SourceFileCoord):
    """Coordinates of a single source file to process.

    NOTE: All data vars & ensemble members actually live within the same ECMWF grib file,
        but all in different sections, which we want to do windowed downloads/reads on.

    Data_var_group is a sequence, but should in practice only have one element (see max_vars_per_download_group below).
    Ensemble member is a single int instead of a sequence (expanded out in generate_source_file_coords).
    """

    init_time: Timestamp
    lead_time: Timedelta
    ensemble_member: int

    # should contain one element, but leaving as sequence for flexibility
    data_var_group: Sequence[EcmwfIfsEnsDataVar]

    s3_bucket_url: ClassVar[str] = "ecmwf-forecasts"
    s3_region: ClassVar[str] = "eu-central-1"

    def _get_base_url(self) -> str:
        base_url = f"https://{self.s3_bucket_url}.s3.{self.s3_region}.amazonaws.com"

        init_time_str = self.init_time.strftime("%Y%m%d")
        init_hour_str = self.init_time.strftime("%H")  # pads 0 to be "00", as desired
        lead_time_hour_str = whole_hours(self.lead_time)

        # On 2024-02-29 and onward, the /ifs/ directory is included in the URL path.
        if self.init_time >= pd.Timestamp("2024-02-29T00:00"):
            directory_path = f"{init_time_str}/{init_hour_str}z/ifs/0p25/enfo"
        else:
            directory_path = f"{init_time_str}/{init_hour_str}z/0p25/enfo"

        filename = f"{init_time_str}{init_hour_str}0000-{lead_time_hour_str}h-enfo-ef"
        return f"{base_url}/{directory_path}/{filename}"

    def get_url(self) -> str:
        return self._get_base_url() + ".grib2"

    def get_index_url(self) -> str:
        return self._get_base_url() + ".index"

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
            "ensemble_member": self.ensemble_member,
        }


class EcmwfIfsEnsForecast15Day025DegreeRegionJob(
    RegionJob[EcmwfIfsEnsDataVar, EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord]
):
    # Optionally, limit the number of variables downloaded together.
    # If set to a value less than len(data_vars), downloading, reading/recompressing,
    # and uploading steps will be pipelined within a region job.
    # 5 is a reasonable default if it is possible to download less than all
    # variables in a single file (e.g. you have a grib index).
    # Leave unset if you have to download a whole file to get one variable out
    # to avoid re-downloading the same file multiple times.
    #
    max_vars_per_download_group: ClassVar[int] = 1

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
        data_var_group: Sequence[EcmwfIfsEnsDataVar],
    ) -> Sequence[EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord]:
        """Return a sequence of coords, one for each source file required to process the data covered by processing_region_ds."""
        return [
            EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord(
                init_time=init_time,
                lead_time=lead_time,
                data_var_group=data_var_group,
                ensemble_member=int(ensemble_member),
            )
            for init_time, lead_time, ensemble_member in itertools.product(
                processing_region_ds["init_time"].values,
                processing_region_ds["lead_time"].values,
                processing_region_ds["ensemble_member"].values,
            )
        ]

    def download_file(
        self, coord: EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord
    ) -> Path:
        """Download the file for the given coordinate and return the local path."""
        # Download grib index file
        idx_url = coord.get_index_url()
        idx_local_path = http_download_to_disk(idx_url, self.dataset_id)

        """
        max_vars_per_download_group = 1, max_download = 1
        hrrr forecast: steal vars_suffix (as just suffix). need unique request
        SourceFileCoord: include ensemble member
        own grib index parser -- pd readlines, pandas json lineparser. ask AI
        byte range: offset + offset+length in index file

        read_data: extremely similar to HRRR.
        don't need matching really, reader.read 1
        """

        def parse_index_file(index_local_path: PathLike[str]) -> pd.DataFrame:
            """
            Returns a pandas df representing the index file.
            MultiIndex of (number, param)
            Columns include:
              - used information: type (control/perturbed), param (variable short name), _offset (byte window start), _length (length of byte window)
              - unused information: levtype (sfc/pl/sol/...), levelist, domain, date, time, step, expver, class, stream
            """
            df = pd.read_json(index_local_path, lines=True)

            # Control members don't have "number" field, so fill with 0
            df["number"] = df["number"].fillna(0).astype(int)
            # Ensure that every row we filled with number=0 was indeed type "cf" (control forecast)
            assert all(df[df["number"] == 0]["type"] == "cf"), (
                "Parsed row as control member that didn't have type='cf'"
            )

            return df.set_index(["number", "param"]).sort_index()

        def get_message_byte_ranges_from_index(
            index_local_path: PathLike[str],
            data_vars: Sequence[EcmwfIfsEnsDataVar],
            ensemble_member: int,
        ) -> tuple[list[int], list[int]]:
            """
            Returns a list of byte range starts & a list of byte range ends,
            with the elements of each list in order of data vars.
            """
            byte_range_starts = []
            byte_range_ends = []
            index_file_df = parse_index_file(index_local_path)
            for data_var in data_vars:
                start, length = index_file_df.loc[
                    (ensemble_member, data_var.internal_attrs.grib_index_param),
                    ["_offset", "_length"],
                ].values[0]
                byte_range_starts.append(start)
                byte_range_ends.append(start + length)
            return byte_range_starts, byte_range_ends

        # Download the grib messages for the data vars in the coord using byte ranges
        byte_range_starts, byte_range_ends = get_message_byte_ranges_from_index(
            idx_local_path,
            coord.data_var_group,
            coord.ensemble_member,
        )
        suffix = digest(
            f"{s}-{e}" for s, e in zip(byte_range_starts, byte_range_ends, strict=True)
        )
        return http_download_to_disk(
            coord.get_url(),
            self.dataset_id,
            byte_ranges=(byte_range_starts, byte_range_ends),
            local_path_suffix=f"-{suffix}",
        )

    def read_data(
        self,
        coord: EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord,
        data_var: EcmwfIfsEnsDataVar,
    ) -> ArrayFloat32:
        """Read and return an array of data for the given variable and source file coordinate."""

        with rasterio.open(coord.downloaded_path) as reader:
            assert reader.count == 1, "Expected only one band per downloaded file"
            assert (
                reader.tags(1)["GRIB_COMMENT"] == data_var.internal_attrs.grib_comment
            )
            rasterio_band_index = 1

            result: ArrayFloat32 = reader.read(
                rasterio_band_index, out_dtype=np.float32
            )
            assert result.shape == (721, 1439), (
                f"Expected (721, 1439) shape, found {result.shape}"
            )
            return result

    # Implement this to apply transformations to the array (e.g. deaccumulation)
    #
    # def apply_data_transformations(
    #     self, data_array: xr.DataArray, data_var: EcmwfIfsEnsDataVar
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
    #     data_var : EcmwfIfsEnsDataVar
    #         The data variable metadata object, which may contain transformation parameters.
    #     """
    #     super().apply_data_transformations(data_array, data_var)

    def update_template_with_results(
        self,
        process_results: Mapping[
            str, Sequence[EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord]
        ],
    ) -> xr.Dataset:
        """
        Update template dataset based on processing results. This method is called
        during operational updates.

        Subclasses should implement this method to apply dataset-specific adjustments
        based on the processing results. Examples include:
        - Trimming dataset along append_dim to only include successfully processed data
        - Loading existing coordinate values from the primary store and updating them based on results
        - Updating metadata based on what was actually processed vs what was planned

        The default implementation trims along append_dim to end at the most recent
        successfully processed coordinate (timestamp).

        Parameters
        ----------
        process_results : Mapping[str, Sequence[EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord]]
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
        all_data_vars: Sequence[EcmwfIfsEnsDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[
            "RegionJob[EcmwfIfsEnsDataVar, EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord]"
        ],
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
        all_data_vars : Sequence[EcmwfIfsEnsDataVar]
            Sequence of all data variable configs for this dataset.
        reformat_job_name : str
            The name of the reformatting job, used for progress tracking.
            This is often the name of the Kubernetes job, or "local".

        Returns
        -------
        Sequence[RegionJob[EcmwfIfsEnsDataVar, EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord]]
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
