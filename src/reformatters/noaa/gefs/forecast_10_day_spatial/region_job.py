from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.download import s3_download_to_disk, s3_store
from reformatters.common.logging import get_logger
from reformatters.common.region_job import RegionJob
from reformatters.common.types import AppendDim, DatetimeLike
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,
)
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_B22_TRANSITION_DATE,
    GEFS_CURRENT_ARCHIVE_START,
    GEFS_S_FILE_MAX,
    GEFSDataVar,
    GefsEnsembleSourceFileCoord,
)
from reformatters.noaa.noaa_grib_index import (
    GRIB_INDEX_UNKNOWN_END_PAD,
    grib_message_byte_ranges_from_index,
)
from reformatters.noaa.noaa_utils import has_hour_0_values

log = get_logger(__name__)

_S3_LOCATION_PREFIX = "s3://noaa-gefs-pds/"
_S3_BUCKET_REGION = "us-east-1"


class GefsForecast10DaySpatialSourceFileCoord(GefsEnsembleSourceFileCoord):
    """One s file: (init_time, ensemble_member, lead_time) and the vars it packs."""

    @property
    def gefs_file_type(self) -> Literal["s"]:
        # Virtual chunks decode the raw GRIB message, so every ref must share the
        # 0.25 degree s file grid; this dataset's leads stop where the s files do.
        assert self.init_time >= GEFS_CURRENT_ARCHIVE_START, (
            f"Only the current GEFS archive is supported, got {self.init_time}"
        )
        assert self.lead_time <= GEFS_S_FILE_MAX, (
            f"s files end at {GEFS_S_FILE_MAX}, got {self.lead_time}"
        )
        assert self.data_vars == _vars_in_s_file(
            self.data_vars, self.init_time, self.lead_time
        ), f"coord includes vars not in the s file for {self.init_time}"
        return "s"

    def get_url(self) -> str:
        """The s3:// source location; refs point here, matching the dataset's
        virtual chunk container prefix."""
        url = super().get_url()
        https_prefix = f"https://{self.primary_base_url}/"
        assert url.startswith(https_prefix), url
        return _S3_LOCATION_PREFIX + url.removeprefix(https_prefix)


class GefsForecast10DaySpatialRegionJob(
    VirtualRegionJob[GEFSDataVar, GefsForecast10DaySpatialSourceFileCoord]
):
    """RegionJob for the GEFS 10-day spatial (virtual) forecast dataset."""

    # Concurrent index file downloads. The .idx files are ~30KB latency-bound
    # requests; measured throughput vs pool width: 8 -> ~105 files/s,
    # 32 -> ~310, 64 -> ~380, 128 -> ~435 (diminishing).
    download_concurrency: ClassVar[int] = 64

    def generate_source_file_coords(
        self, processing_region_ds: xr.Dataset, data_var_group: Sequence[GEFSDataVar]
    ) -> Sequence[GefsForecast10DaySpatialSourceFileCoord]:
        coords = []
        for init_time in processing_region_ds["init_time"].values:
            for ensemble_member in processing_region_ds["ensemble_member"].values:
                for lead_time in processing_region_ds["lead_time"].values:
                    vars_in_file = _vars_in_s_file(
                        data_var_group, pd.Timestamp(init_time), pd.Timedelta(lead_time)
                    )
                    if not vars_in_file:
                        continue
                    coords.append(
                        GefsForecast10DaySpatialSourceFileCoord(
                            init_time=pd.Timestamp(init_time),
                            ensemble_member=int(ensemble_member),
                            lead_time=pd.Timedelta(lead_time),
                            data_vars=vars_in_file,
                        )
                    )
        return coords

    def discover_available(
        self, pending: list[GefsForecast10DaySpatialSourceFileCoord]
    ) -> list[tuple[GefsForecast10DaySpatialSourceFileCoord, int]]:
        return discover_available_by_obstore_listing(
            pending,
            store=s3_store(_S3_LOCATION_PREFIX, region=_S3_BUCKET_REGION),
            location_prefix=_S3_LOCATION_PREFIX,
            require_index=True,
        )

    def file_refs(
        self, coord: GefsForecast10DaySpatialSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        index_path = s3_download_to_disk(
            coord.get_index_url(), self.dataset_id, region=_S3_BUCKET_REGION
        )
        try:
            starts, ends = grib_message_byte_ranges_from_index(
                index_path, coord.data_vars, coord.init_time, coord.lead_time
            )
        finally:
            # Index files accumulate by the millions in backfills; never keep them.
            index_path.unlink()

        # The last message in the file has no end byte in the index; the listed
        # file size supplies it.
        ends = [
            file_size if end - start >= GRIB_INDEX_UNKNOWN_END_PAD else end
            for start, end in zip(starts, ends, strict=True)
        ]
        # Byte ranges past the data file mean a stale/mismatched index; skip the file.
        if any(
            end > file_size or end <= start
            for start, end in zip(starts, ends, strict=True)
        ):
            log.warning(
                f"Skipping {coord.get_url()}: index byte ranges fall outside the "
                f"{file_size}-byte data file; stale or mismatched index"
            )
            return []

        out_loc = coord.out_loc()
        location = coord.get_url()
        return [
            VirtualRef(
                data_var=var,
                out_loc=out_loc,
                location=location,
                offset=start,
                length=end - start,
            )
            for var, start, end in zip(coord.data_vars, starts, ends, strict=True)
        ]

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,  # noqa: ARG003 - the icechunk manifest, not a coordinate, tracks ingested data
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[GEFSDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[RegionJob[GEFSDataVar, GefsForecast10DaySpatialSourceFileCoord]],
        xr.DataTree,
    ]:
        """A single polling job over the recent init times (24h window, the last 4 inits).

        Polls until all expected files are ingested; filter_already_present derives
        the remaining work from the manifest. See "Operational updates" in
        docs/virtual_datasets.md.
        """
        append_dim_end = pd.Timestamp.now()
        template_ds = get_template_fn(append_dim_end)
        init_times = template_ds.to_dataset().get_index(append_dim)
        window_start = int(
            init_times.searchsorted(append_dim_end - pd.Timedelta("24h"))
        )
        job = cls(
            tmp_store=tmp_store,
            template_ds=template_ds,
            data_vars=all_data_vars,
            append_dim=append_dim,
            region=slice(window_start, len(init_times)),
            reformat_job_name=reformat_job_name,
            processing_mode="update",
        )
        return [job], template_ds


def _vars_in_s_file(
    data_vars: Sequence[GEFSDataVar], init_time: pd.Timestamp, lead_time: pd.Timedelta
) -> list[GEFSDataVar]:
    """The subset of data_vars the s file for (init_time, lead_time) contains.

    "s+b-b22" variables were only added to the s files at GEFS_B22_TRANSITION_DATE,
    and accumulated/avg variables don't exist in the 0-hour forecast.
    """
    return [
        var
        for var in data_vars
        if (
            var.internal_attrs.gefs_file_type != "s+b-b22"
            or init_time >= GEFS_B22_TRANSITION_DATE
        )
        and (lead_time != pd.Timedelta(0) or has_hour_0_values(var))
    ]
