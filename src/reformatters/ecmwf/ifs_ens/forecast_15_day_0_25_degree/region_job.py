import itertools
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.common.download import (
    download_to_disk,
    get_local_path,
    http_download_to_disk,
    s3_store,
)
from reformatters.common.iterating import digest, item
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
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar
from reformatters.ecmwf.ecmwf_grib_index import get_message_byte_ranges_from_index
from reformatters.ecmwf.ecmwf_utils import all_variables_available, has_hour_0_values

log = get_logger(__name__)


MARS_OPEN_DATA_CUTOVER = pd.Timestamp("2024-04-01T00:00")

MARS_STAGING_BUCKET = "s3://us-west-2.opendata.source.coop"
MARS_STAGING_PREFIX = "dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens"
MARS_STAGING_REGION = "us-west-2"


def _mars_request_type(levtype: str, ensemble_member: int) -> str:
    """Map a level type and ensemble member to the MARS request type used in source.coop file paths."""
    if levtype == "sfc":
        if ensemble_member == 0:
            return "cf_sfc"
        return "pf_sfc_0" if ensemble_member <= 25 else "pf_sfc_1"
    return "cf_pl" if ensemble_member == 0 else "pf_pl"


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
    data_var_group: Sequence[EcmwfDataVar]

    s3_bucket_url: ClassVar[str] = "ecmwf-forecasts"
    s3_region: ClassVar[str] = "eu-central-1"

    def is_mars_source(self) -> bool:
        return self.init_time < MARS_OPEN_DATA_CUTOVER

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

    def mars_request_types(self) -> set[str]:
        """Distinct MARS request types needed for all data vars in this coord."""
        return {
            _mars_request_type(
                v.internal_attrs.grib_index_level_type, self.ensemble_member
            )
            for v in self.data_var_group
        }

    def _mars_date_str(self) -> str:
        return self.init_time.strftime("%Y-%m-%d")

    def mars_grib_s3_path(self, request_type: str) -> str:
        return f"{MARS_STAGING_PREFIX}/{self._mars_date_str()}/{request_type}.grib"

    def mars_index_s3_path(self, request_type: str) -> str:
        return f"{MARS_STAGING_PREFIX}/{self._mars_date_str()}/{request_type}.grib.idx"

    def out_loc(
        self,
    ) -> Mapping[Dim, CoordinateValueOrRange]:
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
            "ensemble_member": self.ensemble_member,
        }


class EcmwfIfsEnsForecast15Day025DegreeRegionJob(
    RegionJob[EcmwfDataVar, EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord]
):
    # Limits the number of variables downloaded together.
    # All variables are scattered throughout the grib file without any organization,
    # so it's more efficient to do separate windowed downloads & reads for each
    # variable that we can parallelize.
    max_vars_per_download_group: ClassVar[int] = 1
    max_vars_per_backfill_job: ClassVar[int] = 1

    @classmethod
    def source_groups(
        cls,
        data_vars: Sequence[EcmwfDataVar],
    ) -> Sequence[Sequence[EcmwfDataVar]]:
        """Return groups of variables, where all variables in a group can be retrieved from the same source file."""
        vars_by_key: defaultdict[tuple[object, bool], list[EcmwfDataVar]] = defaultdict(
            list
        )
        for data_var in data_vars:
            key = (data_var.internal_attrs.date_available, has_hour_0_values(data_var))
            vars_by_key[key].append(data_var)
        return list(vars_by_key.values())

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[EcmwfDataVar],
    ) -> Sequence[EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord]:
        """Returns a sequence of coords, one for each source file required to process the data covered by processing_region_ds.

        NOTE: all ensemble members are included in the same source file, but we only
            include one per SourceFileCoord because we want them to get processed separately.
        This is because the ensemble members & variables are scattered throughout the file
            rather than being clustered in one contiguous window, and we can get better
            download/read performance by treating them separately and parallelizing.
        """
        coords = []
        group_has_hour_0_values = item({has_hour_0_values(v) for v in data_var_group})
        for init_time, lead_time, ensemble_member in itertools.product(
            processing_region_ds["init_time"].values,
            processing_region_ds["lead_time"].values,
            processing_region_ds["ensemble_member"].values,
        ):
            if not all_variables_available(data_var_group, init_time):
                dates_available = {
                    v.internal_attrs.date_available for v in data_var_group
                }
                assert len(dates_available) == 1, (
                    f"Expected all variables in the group to have the same date_available, found {dates_available}"
                )
                continue

            if not group_has_hour_0_values and lead_time == np.timedelta64(0):
                continue

            coord = EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord(
                init_time=init_time,
                lead_time=lead_time,
                data_var_group=data_var_group,
                ensemble_member=int(ensemble_member),
            )
            coords.append(coord)
        return coords

    def download_file(
        self, coord: EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord
    ) -> Path:
        """Download the file for the given coordinate and return the local path."""
        if coord.is_mars_source():
            return self._download_mars_file(coord)
        return self._download_open_data_file(coord)

    def _download_open_data_file(
        self, coord: EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord
    ) -> Path:
        # Download grib index file
        idx_url = coord.get_index_url()
        idx_local_path = http_download_to_disk(idx_url, self.dataset_id)

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

    def _download_mars_file(
        self, coord: EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord
    ) -> Path:
        store = s3_store(MARS_STAGING_BUCKET, MARS_STAGING_REGION, skip_signature=True)
        step = whole_hours(coord.lead_time)

        # MARS files are organized by request type; collect byte ranges across all needed types
        all_byte_range_starts: list[int] = []
        all_byte_range_ends: list[int] = []
        # All data vars in a coord share the same request type (same level type + same member)
        request_type = item(coord.mars_request_types())

        # Download MARS index
        idx_s3_path = coord.mars_index_s3_path(request_type)
        idx_local_path = get_local_path(self.dataset_id, idx_s3_path)
        download_to_disk(store, idx_s3_path, idx_local_path, overwrite_existing=False)

        byte_range_starts, byte_range_ends = get_message_byte_ranges_from_index(
            idx_local_path,
            coord.data_var_group,
            coord.ensemble_member,
            step=step,
        )
        all_byte_range_starts.extend(byte_range_starts)
        all_byte_range_ends.extend(byte_range_ends)

        suffix = digest(
            f"{s}-{e}"
            for s, e in zip(all_byte_range_starts, all_byte_range_ends, strict=True)
        )
        grib_s3_path = coord.mars_grib_s3_path(request_type)
        local_path = get_local_path(
            self.dataset_id, grib_s3_path, local_path_suffix=f"-{suffix}"
        )
        download_to_disk(
            store,
            grib_s3_path,
            local_path,
            byte_ranges=(all_byte_range_starts, all_byte_range_ends),
            overwrite_existing=True,
        )
        return local_path

    def read_data(
        self,
        coord: EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord,
        data_var: EcmwfDataVar,
    ) -> ArrayFloat32:
        """Read and return an array of data for the given variable and source file coordinate."""

        with rasterio.open(coord.downloaded_path) as reader:
            # Expecting one band per downloaded file because we should only have one data var & one ensemble member
            assert reader.count == 1, "Expected only one band per downloaded file"
            rasterio_band_index = 1

            # MARS GRIBs have different comment/description metadata than open data,
            # so we only validate these fields for open data sources.
            if not coord.is_mars_source():
                grib_comment = reader.tags(rasterio_band_index)["GRIB_COMMENT"]
                grib_description = reader.descriptions[rasterio_band_index - 1]

                if data_var.name == "categorical_precipitation_type_surface":
                    # ECMWF occasionally adds new values in the reserved range.
                    # Check the first 6 categories that shouldn't change.
                    assert (
                        grib_comment[:100] == data_var.internal_attrs.grib_comment[:100]
                    ), f"{grib_comment=} != {data_var.internal_attrs.grib_comment=}"
                else:
                    assert grib_comment == data_var.internal_attrs.grib_comment, (
                        f"{grib_comment=} != {data_var.internal_attrs.grib_comment=}"
                    )
                assert grib_description == data_var.internal_attrs.grib_description, (
                    f"{grib_description=} != {data_var.internal_attrs.grib_description}"
                )

            result: ArrayFloat32 = reader.read(
                rasterio_band_index, out_dtype=np.float32
            )
            expected_shape = (721, 1440)
            assert result.shape == expected_shape, (
                f"Expected {expected_shape} shape, found {result.shape}"
            )

            if (
                coord.is_mars_source()
                and data_var.internal_attrs.mars_read_scale_factor is not None
            ):
                result = result * data_var.internal_attrs.mars_read_scale_factor

            return result

    def apply_data_transformations(
        self, data_array: xr.DataArray, data_var: EcmwfDataVar
    ) -> None:
        """
        Apply in-place data transformations to the output data array for a given data variable.
        Deaccumulates precipitation to rates.

        Parameters
        ----------
        data_array : xr.DataArray
            The output data array to be transformed in-place.
        data_var : EcmwfDataVar
            The data variable metadata object, which may contain transformation parameters.
        """
        if data_var.internal_attrs.scale_factor is not None:
            data_array *= data_var.internal_attrs.scale_factor

        if data_var.internal_attrs.deaccumulate_to_rate:
            reset_freq = data_var.internal_attrs.window_reset_frequency
            deaccumulation_invalid_below_threshold_rate = (
                data_var.internal_attrs.deaccumulation_invalid_below_threshold_rate
            )
            assert deaccumulation_invalid_below_threshold_rate is not None
            assert reset_freq is not None

            try:
                deaccumulate_to_rates_inplace(
                    data_array,
                    dim="lead_time",
                    reset_frequency=reset_freq,
                    invalid_below_threshold_rate=deaccumulation_invalid_below_threshold_rate,
                )
            except ValueError:
                log.exception(f"Error deaccumulating {data_var.name}")

        super().apply_data_transformations(data_array, data_var)

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[EcmwfDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[
            "RegionJob[EcmwfDataVar, EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord]"
        ],
        xr.Dataset,
    ]:
        """
        Return the sequence of RegionJob instances necessary to update the dataset
        from its current state to include the latest available data.

        Also return the template_ds, expanded along append_dim through the end of
        the data to process.

        Parameters
        ----------
        primary_store : Store
            The primary store to read existing data from and write updates to.
        tmp_store : Path
            The temporary Zarr store to write into while processing.
        get_template_fn : Callable[[DatetimeLike], xr.Dataset]
            Function to get the template_ds for the operational update.
        append_dim : AppendDim
            The dimension along which data is appended (e.g., "time").
        all_data_vars : Sequence[EcmwfDataVar]
            Sequence of all data variable configs for this dataset.
        reformat_job_name : str
            The name of the reformatting job, used for progress tracking.
            This is often the name of the Kubernetes job, or "local".

        Returns
        -------
        Sequence[RegionJob[EcmwfDataVar, EcmwfIfsEnsForecast15Day025DegreeSourceFileCoord]]
            RegionJob instances that need processing for operational updates.
        xr.Dataset
            The template_ds for the operational update.
        """
        existing_ds = xr.open_zarr(primary_store, chunks=None)
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
