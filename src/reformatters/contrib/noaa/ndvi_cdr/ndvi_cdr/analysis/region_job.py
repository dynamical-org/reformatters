from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

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
    ArrayInt16,
    DatetimeLike,
    Dim,
    Timestamp,
)

from .template_config import NoaaNdviCdrDataVar

log = get_logger(__name__)


class NoaaNdviCdrAnalysisSourceFileCoord(SourceFileCoord):
    """Coordinates of a single source file to process."""

    time: Timestamp
    url: str

    def get_url(self) -> str:
        return self.url

    def out_loc(
        self,
    ) -> Mapping[Dim, CoordinateValueOrRange]:
        return {"time": self.time}


class NoaaNdviCdrAnalysisRegionJob(
    RegionJob[NoaaNdviCdrDataVar, NoaaNdviCdrAnalysisSourceFileCoord]
):
    download_parallelism: int = 2
    # We observed deadlocks when using more than 2 threads to read data into shared memory.
    read_parallelism: int = 1

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaNdviCdrDataVar],
    ) -> Sequence[NoaaNdviCdrAnalysisSourceFileCoord]:
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
            # TODO: Should we rethink this? This kind of preemptively handles the dataset truncation
            # which would otherwise happen during operational update. It means we fail if our append_dim end date
            # is too far in the future.
            raise ValueError(f"No file found for {time}")

        print("Returning source file coords")
        return [
            NoaaNdviCdrAnalysisSourceFileCoord(time=t, url=_get_url(t)) for t in times
        ]

    def download_file(self, coord: NoaaNdviCdrAnalysisSourceFileCoord) -> Path:
        """Download the file for the given coordinate and return the local path."""
        return http_download_to_disk(coord.get_url(), self.dataset_id)

    def read_data(
        self,
        coord: NoaaNdviCdrAnalysisSourceFileCoord,
        data_var: NoaaNdviCdrDataVar,
    ) -> ArrayFloat32 | ArrayInt16:
        """Read and return an array of data for the given variable and source file coordinate."""
        var_name = data_var.internal_attrs.netcdf_var_name
        scale_factor = data_var.internal_attrs.scale_factor
        add_offset = data_var.internal_attrs.add_offset
        valid_range = data_var.internal_attrs.valid_range

        dtype = data_var.encoding.dtype
        encoding_fill_value = data_var.encoding.fill_value

        netcdf_path = f"netcdf:{coord.downloaded_path}:{var_name}"
        band = 1  # because rasterio netcdf requires selecting the band in the file path we always want band 1

        with rasterio.open(netcdf_path) as reader:
            masked_result = reader.read(band, out_dtype=dtype, masked=True)
            result: Array2D[np.float32 | np.int16] = masked_result.filled(
                encoding_fill_value
            )
            assert result.shape == (3600, 7200)

            if var_name != "QA":
                assert scale_factor is not None
                assert add_offset is not None
                assert valid_range is not None
                assert (
                    np.nanmin(result) >= valid_range[0]
                    and np.nanmax(result) <= valid_range[1]
                )

                assert result.dtype == np.float32, (
                    f"Expected float32, got {result.dtype}"
                )
                result = cast(Array2D[np.float32], result)
                result *= scale_factor
                result += add_offset

            if var_name == "QA":
                assert result.dtype == np.int16
                return cast(ArrayInt16, result)
            else:
                assert result.dtype == np.float32
                return cast(ArrayFloat32, result)

    @classmethod
    def operational_update_jobs(
        cls,
        final_store: zarr.abc.store.Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[NoaaNdviCdrDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence["RegionJob[NoaaNdviCdrDataVar, NoaaNdviCdrAnalysisSourceFileCoord]"],
        xr.Dataset,
    ]:
        existing_ds = xr.open_zarr(final_store)
        append_dim_start = existing_ds[append_dim].max()
        append_dim_end = pd.Timestamp.now()
        template_ds = get_template_fn(append_dim_end)

        jobs = cls.get_jobs(
            kind="operational-update",
            final_store=final_store,
            tmp_store=tmp_store,
            template_ds=template_ds,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            filter_start=append_dim_start,
        )
        return jobs, template_ds
