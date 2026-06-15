from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.download import s3_download_to_disk, s3_store
from reformatters.common.logging import get_logger
from reformatters.common.region_job import (
    CoordinateValue,
    RegionJob,
    SourceFileCoord,
)
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import AppendDim, DatetimeLike, Dim, Timedelta, Timestamp
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,
)
from reformatters.ecmwf.ecmwf_config_models import (
    EcmwfDataVar,
    has_hour_0_values,
    vars_available,
)
from reformatters.ecmwf.ecmwf_grib_index import (
    grib_message_byte_ranges_from_index_by_member,
)

log = get_logger(__name__)

_S3_LOCATION_PREFIX = "s3://ecmwf-forecasts/"
_S3_BUCKET_REGION = "eu-central-1"


class IfsEnsForecast15DaySpatialSourceFileCoord(SourceFileCoord):
    """One open-data GRIB file: (init_time, lead_time) for a set of ensemble members.

    Post IFS-50r1 the control member (0) lives in the oper-fc file and perturbed
    members (1-50) in the enfo-ef file, so one (init_time, lead_time) maps to two
    files; each is its own coord with its own member set.
    """

    init_time: Timestamp
    lead_time: Timedelta
    ensemble_members: tuple[int, ...]
    data_vars: Sequence[EcmwfDataVar]

    @property
    def is_control(self) -> bool:
        return self.ensemble_members[0] == 0

    def _base_url(self) -> str:
        date_str = self.init_time.strftime("%Y%m%d")
        hour_str = self.init_time.strftime("%H")
        lead_hours = whole_hours(self.lead_time)
        stream, kind = ("oper", "oper-fc") if self.is_control else ("enfo", "enfo-ef")
        return (
            f"{_S3_LOCATION_PREFIX}{date_str}/{hour_str}z/ifs/0p25/{stream}/"
            f"{date_str}{hour_str}0000-{lead_hours}h-{kind}"
        )

    def get_url(self) -> str:
        return self._base_url() + ".grib2"

    def get_index_url(self) -> str:
        return self._base_url() + ".index"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        # The whole file commits atomically, so the first member stands in as the
        # filter's presence probe for all of them.
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
            "ensemble_member": self.ensemble_members[0],
        }


class EcmwfIfsEnsForecast15DaySpatialRegionJob(
    VirtualRegionJob[EcmwfDataVar, IfsEnsForecast15DaySpatialSourceFileCoord]
):
    """RegionJob for the ECMWF IFS ENS 15-day spatial (virtual) forecast dataset."""

    def generate_source_file_coords(
        self, processing_region_ds: xr.Dataset, data_var_group: Sequence[EcmwfDataVar]
    ) -> Sequence[IfsEnsForecast15DaySpatialSourceFileCoord]:
        members = [int(m) for m in processing_region_ds["ensemble_member"].values]
        member_groups = [
            group
            for group in (
                tuple(m for m in members if m == 0),  # control -> oper-fc
                tuple(m for m in members if m != 0),  # perturbed -> enfo-ef
            )
            if group
        ]
        coords = []
        for init_time in processing_region_ds["init_time"].values:
            for lead_time in processing_region_ds["lead_time"].values:
                vars_in_file = _vars_in_file(
                    data_var_group, pd.Timestamp(init_time), pd.Timedelta(lead_time)
                )
                if not vars_in_file:
                    continue
                coords.extend(
                    IfsEnsForecast15DaySpatialSourceFileCoord(
                        init_time=pd.Timestamp(init_time),
                        lead_time=pd.Timedelta(lead_time),
                        ensemble_members=member_group,
                        data_vars=vars_in_file,
                    )
                    for member_group in member_groups
                )
        return coords

    def discover_available(
        self, pending: list[IfsEnsForecast15DaySpatialSourceFileCoord]
    ) -> list[tuple[IfsEnsForecast15DaySpatialSourceFileCoord, int]]:
        return discover_available_by_obstore_listing(
            pending,
            store=s3_store(_S3_LOCATION_PREFIX, region=_S3_BUCKET_REGION),
            location_prefix=_S3_LOCATION_PREFIX,
            require_index=True,
        )

    def file_refs(
        self, coord: IfsEnsForecast15DaySpatialSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        index_path = s3_download_to_disk(
            coord.get_index_url(), self.dataset_id, region=_S3_BUCKET_REGION
        )
        try:
            ranges_by_member = grib_message_byte_ranges_from_index_by_member(
                index_path, coord.data_vars, coord.ensemble_members
            )
        finally:
            # Index files accumulate by the millions in backfills; never keep them.
            index_path.unlink()

        location = coord.get_url()
        refs: list[VirtualRef] = []
        for member, (starts, ends) in ranges_by_member.items():
            out_loc: Mapping[Dim, CoordinateValue] = {
                "init_time": coord.init_time,
                "lead_time": coord.lead_time,
                "ensemble_member": member,
            }
            for var, start, end in zip(coord.data_vars, starts, ends, strict=True):
                # Byte ranges past the data file mean a stale/mismatched index; skip it.
                if end > file_size or end <= start:
                    log.warning(
                        f"Skipping {coord.get_url()}: index byte ranges fall outside "
                        f"the {file_size}-byte data file; stale or mismatched index"
                    )
                    return []
                refs.append(
                    VirtualRef(
                        data_var=var,
                        out_loc=out_loc,
                        location=location,
                        offset=start,
                        length=end - start,
                    )
                )
        return refs

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,  # noqa: ARG003 - the icechunk manifest, not a coordinate, tracks ingested data
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[EcmwfDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[RegionJob[EcmwfDataVar, IfsEnsForecast15DaySpatialSourceFileCoord]],
        xr.Dataset,
    ]:
        """A single polling job over the recent init times (48h window, the last 2 daily inits).

        Polls until all expected files are ingested; filter_already_present derives
        the remaining work from the manifest. See "Operational updates" in
        docs/virtual_datasets.md.
        """
        append_dim_end = pd.Timestamp.now()
        template_ds = get_template_fn(append_dim_end)
        init_times = template_ds.get_index(append_dim)
        window_start = int(
            init_times.searchsorted(append_dim_end - pd.Timedelta("48h"))
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


def _vars_in_file(
    data_vars: Sequence[EcmwfDataVar], init_time: pd.Timestamp, lead_time: pd.Timedelta
) -> list[EcmwfDataVar]:
    """The subset of data_vars an open-data file for (init_time, lead_time) contains.

    Excludes variables not yet available at init_time and max/min variables that
    have no value at lead_time=0h.
    """
    return [
        var
        for var in data_vars
        if vars_available([var], init_time)
        and (lead_time != pd.Timedelta(0) or has_hour_0_values(var))
    ]
