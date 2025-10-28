import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import numpy as np
import obstore
import pandas as pd
import rasterio  # type: ignore[import-untyped]
import requests
import xarray as xr
import zarr

from reformatters.common.download import (
    download_to_disk,
    get_local_path,
    http_store,
    s3_store,
)
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
    # Set lower than would be needed for fetching exclusively from S3
    # to accomodate the cases where we are downloading from NCEI.
    download_parallelism: int = 5

    # We observed deadlocks when using more than 2 threads to read data into shared memory.
    read_parallelism: int = 1

    s3_bucket_url: str = "s3://noaa-cdr-ndvi-pds"
    s3_region: str = "us-east-1"

    root_nc_url: str = (
        "http://ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access"
    )

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        _data_var_group: Sequence[NoaaNdviCdrDataVar],
    ) -> Sequence[NoaaNdviCdrAnalysisSourceFileCoord]:
        """Return a sequence of coords, one for each source file required to process the data covered by processing_region_ds.

        The source file names include the processing timestamp, which means that we cannot determine the filename and URL
        just from a timestamp. When generating SourceFileCoords we need to enumerate the available files and match them to each time we are
        are attempting to process.

        The source data for this dataset is stored and available from a variety of sources. See the "Data Access" section of
        https://www.ncei.noaa.gov/products/climate-data-records/normalized-difference-vegetation-index. We are currently using
        the NOAA S3 bucket.
        """
        times = pd.to_datetime(processing_region_ds["time"].values)
        years = set(times.year)

        # Build a mapping from date string to URL for all files in all relevant years
        urls_by_time: dict[pd.Timestamp, str] = {}

        for year in years:
            for filepath in self._list_source_files(year):
                # Example filename: AVHRR-Land_v005_AVH13C1_NOAA-07_19810728_c20170610011910.nc
                # We want to extract the date part (e.g., 19810728)
                try:
                    _, date_str, _ = filepath.rsplit("_", 2)

                    # Parse date string to pd.Timestamp
                    file_time = pd.Timestamp(date_str)
                    filename = Path(filepath).name

                    if self._use_ncei_to_download(file_time):
                        url = f"{self.root_nc_url}/{year}/{filename}"
                    else:
                        url = f"{self.s3_bucket_url}/data/{year}/{filename}"

                    urls_by_time[file_time] = url
                except Exception as e:  # noqa: BLE001
                    log.warning(f"Skipping file {filepath} due to error: {e}")
                    continue  # skip files that don't match the expected pattern

        return [
            NoaaNdviCdrAnalysisSourceFileCoord(time=t, url=urls_by_time[timestamp])
            for t in times
            if (timestamp := pd.Timestamp(t)) in urls_by_time
        ]

    def download_file(self, coord: NoaaNdviCdrAnalysisSourceFileCoord) -> Path:
        """Download the file for the given coordinate and return the local path."""
        url = coord.get_url()
        parsed_url = urlparse(url)

        store: obstore.store.HTTPStore | obstore.store.S3Store
        if parsed_url.netloc == "ncei.noaa.gov":
            store = http_store(f"https://{parsed_url.netloc}")
        else:
            store = s3_store(self.s3_bucket_url, self.s3_region, skip_signature=True)

        remote_path = urlparse(url).path.removeprefix("/")
        local_path = get_local_path(self.dataset_id, remote_path)

        download_to_disk(store, remote_path, local_path, overwrite_existing=True)
        log.debug(f"Downloaded {url} to {local_path}")

        return local_path

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
                # We are using a different fill value here than the data var encoding fill value
                # This is because encoding fill value was previously NaN, and so when we matched
                # matched our no data value, we set values to NaN. We have now changed the
                # encoding fill value to 0. This is to accomdate the fact that due to an Xarray bug,
                # the encoding fill value was not round tripped (it was persisted as 0 despite the
                # definition in our encoding). We have updated the encoding fill value to 0 to match
                # what was written at the time of our backfill. That change ensures that empty chunks
                # continue to be interpreted as 0. But consequently, we need to ensure that when we
                # are setting the no data value when reading the netcdf data, we continue to use NaN.
                if data_var.internal_attrs.read_data_fill_value is not None:
                    result[result == netcdf_fill_value] = (
                        data_var.internal_attrs.read_data_fill_value
                    )

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

    def _use_ncei_to_download(self, file_time: pd.Timestamp) -> bool:
        """Returns True if we should use NCEI to download the file, False otherwise."""
        two_weeks_ago = pd.Timestamp.now() - pd.Timedelta(days=14)
        is_within_last_2_weeks = two_weeks_ago <= file_time
        return is_within_last_2_weeks

    def _list_source_files(self, year: int) -> list[str]:
        # We believe NCEI will have more recent files before S3 does.
        # While this gap may only be a couple of weeks at most, we cannot enumerate
        # files by a coarser granularity than a year. The reason we check if the requested
        # year is the current or previous year is to be sure that we continue to check
        # NCEI in early January of the current year. I.e., in Jan 2026, we should check
        # NCEI for the 2025 files.
        #
        # We hardcode 2025 as the earliest year to check NCEI, since as of this writing,
        # we know S3 is up to date through June 2025 and 2024 is complate.
        # Backfills should go through S3.
        known_complete_aws_year = 2024
        current_year = pd.Timestamp.now().year
        if year > known_complete_aws_year and year in (current_year, current_year - 1):
            return self._list_ncei_source_files(year)
        else:
            return self._list_s3_source_files(year)

    def _list_s3_source_files(self, year: int) -> list[str]:
        store = s3_store(self.s3_bucket_url, self.s3_region, skip_signature=True)
        results = list(obstore.list(store, f"data/{year}", chunk_size=366))
        if len(results) == 0:
            return []

        assert len(results) == 1, (
            "Got unexpected results. Expected 1 list of no more than 366 files"
        )

        return [result["path"] for result in results[0]]

    def _list_ncei_source_files(self, year: int) -> list[str]:
        """List source files from NCEI.

        The response text from NCEI is HTML with a table enumerating available files. Example:

        <td><a href="VIIRS-Land_v001_JP113C1_NOAA-20_20250101_c20250103153010.nc">VIIRS-Land_v001_JP113C1_NOAA-20_20250101_c20250103153010.nc</a></td>
        <td align="right">2025-01-05 15:40</td>
        <td align="right">63914048</td>
        <td></td>
        </tr>
        <tr>
        <td><a href="VIIRS-Land_v001_JP113C1_NOAA-20_20250102_c20250104153009.nc">VIIRS-Land_v001_JP113C1_NOAA-20_20250102_c20250104153009.nc</a></td>
        ...
        """
        ncei_url = f"{self.root_nc_url}/{year}/"

        response = requests.get(ncei_url, timeout=15)
        response.raise_for_status()

        content = response.text
        filenames = re.findall(r"href=\"(VIIRS-Land.+nc)\"", content)
        filenames = list(set(filenames))

        # Simple check: startswith, endswith, and only one .nc present
        def is_valid_viirs_nc(fname: str) -> bool:
            return (
                fname.startswith("VIIRS-Land")
                and fname.endswith(".nc")
                and fname.count(".nc") == 1
            )

        assert all(is_valid_viirs_nc(fname) for fname in filenames), (
            "Some filenames do not conform to expected structure: "
            + str([fname for fname in filenames if not is_valid_viirs_nc(fname)])
        )

        return filenames

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: zarr.abc.store.Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[NoaaNdviCdrDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence["RegionJob[NoaaNdviCdrDataVar, NoaaNdviCdrAnalysisSourceFileCoord]"],
        xr.Dataset,
    ]:
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
