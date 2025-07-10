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
from reformatters.common.iterating import item
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

from . import quality_flags
from .template_config import (
    QA_NETCDF_VAR_NAME,
    NoaaNdviCdrAnalysisTemplateConfig,
    NoaaNdviCdrDataVar,
)

QA_DATA_VAR = item(
    dv for dv in NoaaNdviCdrAnalysisTemplateConfig().data_vars if dv.name == "qa"
)


log = get_logger(__name__)

VIIRS_START_DATE = pd.Timestamp("2014-01-01")


class NoaaNdviCdrAnalysisSourceFileCoord(SourceFileCoord):
    """Coordinates of a single source file to process."""

    time: Timestamp
    url: str

    def get_url(self) -> str:
        return self.url

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        return {"time": self.time}


class NoaaNdviCdrAnalysisRegionJob(
    RegionJob[NoaaNdviCdrDataVar, NoaaNdviCdrAnalysisSourceFileCoord]
):
    # We observed deadlocks when using more than 2 threads to read data into shared memory.
    read_parallelism: int = 1

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaNdviCdrDataVar],
    ) -> Sequence[NoaaNdviCdrAnalysisSourceFileCoord]:
        """Return a sequence of coords, one for each source file required to process the data covered by processing_region_ds.

        The source file names include the processing timestamp, which means that we cannot determine the filename and URL
        just from a timestamp. When generating SourceFileCoords we need to enumerate the available files and match them to each time we are
        are attempting to process.

        The source data for this dataset is stored and available from a variety of sources. See the "Data Access" section of
        https://www.ncei.noaa.gov/products/climate-data-records/normalized-difference-vegetation-index. We are currently using
        the NOAA S3 bucket.
        """
        # set anon to True to not require AWS credentials, as this is a public bucket.
        fs = fsspec.filesystem("s3", anon=True)
        public_base_url = "https://noaa-cdr-ndvi-pds.s3.amazonaws.com/data"

        times = pd.to_datetime(processing_region_ds["time"].values)
        years = set(times.year)

        # Build a mapping from date string to URL for all files in all relevant years
        urls_by_time: dict[pd.Timestamp, str] = {}

        for year in years:
            for filepath in fs.ls(f"noaa-cdr-ndvi-pds/data/{year}"):
                # Example filename: AVHRR-Land_v005_AVH13C1_NOAA-07_19810728_c20170610011910.nc
                # We want to extract the date part (e.g., 19810728)
                try:
                    _, date_str, _ = filepath.rsplit("_", 2)
                    # Parse date string to pd.Timestamp
                    file_time = pd.Timestamp(date_str)
                    filename = Path(filepath).name
                    url = f"{public_base_url}/{year}/{filename}"
                    urls_by_time[file_time] = url
                except Exception as e:
                    log.warning(f"Skipping file {filepath} due to error: {e}")
                    continue  # skip files that don't match the expected pattern

        coords = []
        for t in times:
            if (timestamp := pd.Timestamp(t)) in urls_by_time:
                coords.append(
                    NoaaNdviCdrAnalysisSourceFileCoord(
                        time=t, url=urls_by_time[timestamp]
                    )
                )
        return coords

    def download_file(self, coord: NoaaNdviCdrAnalysisSourceFileCoord) -> Path:
        """Download the file for the given coordinate and return the local path."""
        url = coord.get_url()
        return http_download_to_disk(url, self.dataset_id)

    def read_data(
        self,
        coord: NoaaNdviCdrAnalysisSourceFileCoord,
        data_var: NoaaNdviCdrDataVar,
    ) -> ArrayFloat32 | ArrayInt16:
        """Read and return an array of data for the given variable and source file coordinate."""
        if data_var.name == "ndvi_usable":
            return self._read_usable_ndvi(coord, data_var)
        else:
            return self._read_netcdf_data(coord, data_var)

    def _read_netcdf_data(
        self, coord: NoaaNdviCdrAnalysisSourceFileCoord, data_var: NoaaNdviCdrDataVar
    ) -> ArrayFloat32 | ArrayInt16:
        """Read data from NetCDF file."""
        out_dtype = data_var.encoding.dtype
        encoding_fill_value = data_var.encoding.fill_value

        var_name = data_var.internal_attrs.netcdf_var_name
        netcdf_fill_value = data_var.internal_attrs.fill_value
        scale_factor = data_var.internal_attrs.scale_factor
        add_offset = data_var.internal_attrs.add_offset

        netcdf_path = f"netcdf:{coord.downloaded_path}:{var_name}"

        with rasterio.open(netcdf_path) as reader:
            masked_result = reader.read(1, out_dtype=out_dtype, masked=True)
            result = masked_result.filled(netcdf_fill_value)

            assert result.shape == (3600, 7200)
            assert result.dtype == np.dtype(out_dtype)

            # Set invalid values to NaN before scaling (for float data)
            if var_name != QA_NETCDF_VAR_NAME:
                result[result == netcdf_fill_value] = encoding_fill_value

                assert scale_factor is not None
                assert add_offset is not None
                result *= scale_factor
                result += add_offset

                assert result.dtype == np.float32
                return cast(ArrayFloat32, result)
            else:
                assert result.dtype == np.int16
                return cast(ArrayInt16, result)

    def _read_usable_ndvi(
        self,
        coord: NoaaNdviCdrAnalysisSourceFileCoord,
        data_var: NoaaNdviCdrDataVar,
    ) -> ArrayFloat32:
        """Read NDVI data and apply quality filtering."""
        ndvi_data = self._read_netcdf_data(coord, data_var)
        qa_data = self._read_netcdf_data(coord, QA_DATA_VAR)

        assert qa_data.dtype == np.int16
        qa_data = cast(Array2D[np.int16], qa_data)

        # Apply quality filtering based on timestamp
        if coord.time < VIIRS_START_DATE:
            bad_values_mask = quality_flags.get_avhrr_mask(qa_data)
        else:
            bad_values_mask = quality_flags.get_viirs_mask(qa_data)

        ndvi_data[bad_values_mask] = np.nan
        return cast(ArrayFloat32, ndvi_data)

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
