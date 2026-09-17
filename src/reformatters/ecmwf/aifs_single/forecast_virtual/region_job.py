from collections.abc import Mapping, Sequence
from typing import ClassVar, Final, assert_never

import icechunk
import pandas as pd
import xarray as xr
from pydantic import PrivateAttr

from reformatters.common.config_models import ROOT
from reformatters.common.download import (
    gcs_download_to_disk,
    gcs_store,
    s3_download_to_disk,
    s3_store,
)
from reformatters.common.logging import get_logger
from reformatters.common.region_job import (
    CoordinateValue,
    InitLeadSourceFileCoord,
)
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Dim, Timedelta
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,
)
from reformatters.ecmwf.aifs_single.template_config import (
    aifs_single_stream_path,
)
from reformatters.ecmwf.ecmwf_grib_index import parse_index_file
from reformatters.ecmwf.ecmwf_utils import EcmwfOpenDataSource

from .template_config import (
    EcmwfAifsSingleVirtualDataVar,
)

log = get_logger(__name__)

GCS_LOCATION_PREFIX = "gs://ecmwf-open-data/"
S3_LOCATION_PREFIX = "s3://ecmwf-forecasts/"
S3_REGION = "eu-central-1"
# Inits before this date may reference s3://ecmwf-forecasts/ for files the GCS
# mirror lacks; inits on or after it reference GCS only, since S3 rate-limits the
# recent data readers touch most.
S3_FALLBACK_BEFORE: Final = pd.Timestamp("2025-03-01")


def aifs_single_virtual_chunk_containers() -> tuple[
    icechunk.VirtualChunkContainer, ...
]:
    """Fresh container objects per call; icechunk containers can't be shared
    pydantic defaults."""
    return (
        icechunk.VirtualChunkContainer(GCS_LOCATION_PREFIX, icechunk.gcs_store()),
        icechunk.VirtualChunkContainer(
            S3_LOCATION_PREFIX, icechunk.s3_store(region=S3_REGION)
        ),
    )


class EcmwfAifsSingleForecastVirtualSourceFileCoord(InitLeadSourceFileCoord):
    """One AIFS Single forecast file (init_time, lead_time) and the vars it packs."""

    data_vars: Sequence[EcmwfAifsSingleVirtualDataVar]
    # Private so discover_available can switch the source of the pending coord
    # itself; the write loop tracks pending coords by identity.
    _source: EcmwfOpenDataSource = PrivateAttr(default="gcs")

    @property
    def source(self) -> EcmwfOpenDataSource:
        return self._source

    def fall_back_to_s3(self) -> None:
        assert self.init_time < S3_FALLBACK_BEFORE, (
            f"{self.init_time} is on or after {S3_FALLBACK_BEFORE}; only older inits "
            "may reference S3"
        )
        self._source = "s3"

    def _get_base_url(self) -> str:
        match self.source:
            case "gcs":
                location_prefix = GCS_LOCATION_PREFIX
            case "s3":
                location_prefix = S3_LOCATION_PREFIX
            case _ as unreachable:
                assert_never(unreachable)
        stream_path = aifs_single_stream_path(self.init_time)
        init_date_str = self.init_time.strftime("%Y%m%d")
        init_hour_str = self.init_time.strftime("%H")
        return (
            f"{location_prefix}{init_date_str}/{init_hour_str}z/"
            f"{stream_path}/"
            f"{init_date_str}{init_hour_str}0000-{whole_hours(self.lead_time)}h-oper-fc"
        )

    def get_url(self) -> str:
        return self._get_base_url() + ".grib2"

    def get_index_url(self) -> str:
        return self._get_base_url() + ".index"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"init_time": self.init_time, "lead_time": self.lead_time}


class EcmwfAifsSingleForecastVirtualRegionJob(
    VirtualRegionJob[
        EcmwfAifsSingleVirtualDataVar, EcmwfAifsSingleForecastVirtualSourceFileCoord
    ]
):
    # Files publish ~init+5.5-6h, so at fire time (init+5h20m) the newest init plus
    # the two prior cycles sit 5h20m/11h20m/17h20m back; 20h covers all three so a
    # couple of missed runs still self-heal.
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("20h")

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[EcmwfAifsSingleVirtualDataVar],
    ) -> Sequence[EcmwfAifsSingleForecastVirtualSourceFileCoord]:
        init_times = pd.to_datetime(processing_region_ds["init_time"].values)
        lead_times = pd.to_timedelta(processing_region_ds["lead_time"].values)

        coords = []
        for init_time in init_times:
            available_vars = [
                var
                for var in data_var_group
                if (date := var.internal_attrs.date_available) is None
                or date <= init_time
            ]
            for lead_time in lead_times:
                vars_in_file = [
                    var
                    for var in available_vars
                    if (
                        var.has_hour_0_values()
                        if lead_time == pd.Timedelta(0)
                        else not var.internal_attrs.lead_0_only
                    )
                ]
                if not vars_in_file:
                    continue
                coords.append(
                    EcmwfAifsSingleForecastVirtualSourceFileCoord(
                        init_time=init_time,
                        lead_time=lead_time,
                        data_vars=vars_in_file,
                    )
                )
        return coords

    def discover_available(
        self, pending: list[EcmwfAifsSingleForecastVirtualSourceFileCoord]
    ) -> list[tuple[EcmwfAifsSingleForecastVirtualSourceFileCoord, int]]:
        """Files GCS can serve (data and index listed), plus, for inits before
        S3_FALLBACK_BEFORE, files S3 can serve that GCS cannot; a fallback coord
        is switched to its S3 URL in place."""
        available = discover_available_by_obstore_listing(
            pending,
            store=gcs_store(GCS_LOCATION_PREFIX),
            location_prefix=GCS_LOCATION_PREFIX,
            require_index=True,
        )
        on_gcs = {id(coord) for coord, _ in available}
        # S3 is listed through copies so a coord GCS lacks and S3 also lacks keeps
        # its GCS URL and stays pending.
        s3_probes: dict[
            int,
            tuple[
                EcmwfAifsSingleForecastVirtualSourceFileCoord,
                EcmwfAifsSingleForecastVirtualSourceFileCoord,
            ],
        ] = {}
        for coord in pending:
            if id(coord) in on_gcs or coord.init_time >= S3_FALLBACK_BEFORE:
                continue
            s3_probe = coord.model_copy()
            s3_probe.fall_back_to_s3()
            s3_probes[id(s3_probe)] = (s3_probe, coord)
        if not s3_probes:
            return available
        on_s3 = discover_available_by_obstore_listing(
            [s3_probe for s3_probe, _ in s3_probes.values()],
            store=s3_store(S3_LOCATION_PREFIX, region=S3_REGION),
            location_prefix=S3_LOCATION_PREFIX,
            require_index=True,
        )
        for s3_probe, file_size in on_s3:
            _, coord = s3_probes[id(s3_probe)]
            coord.fall_back_to_s3()
            available.append((coord, file_size))
        return available

    def file_refs(
        self,
        coord: EcmwfAifsSingleForecastVirtualSourceFileCoord,
        file_size: int,
    ) -> list[VirtualRef]:
        match coord.source:
            case "gcs":
                index_path = gcs_download_to_disk(
                    coord.get_index_url(), self.dataset_id
                )
            case "s3":
                index_path = s3_download_to_disk(
                    coord.get_index_url(), self.dataset_id, region=S3_REGION
                )
            case _ as unreachable:
                assert_never(unreachable)
        try:
            index_df = parse_index_file(index_path, ensemble=False)
        finally:
            index_path.unlink()

        lookup = self._message_lookup(coord.data_vars)
        out_loc_base = dict(coord.out_loc())
        location = coord.get_url()
        refs = []
        entries = index_df.reset_index()
        for param, levtype, levelist, raw_offset, raw_length in zip(
            entries["param"],
            entries["levtype"],
            entries["levelist"],
            entries["_offset"],
            entries["_length"],
            strict=True,
        ):
            level = None if pd.isna(levelist) else int(levelist)
            matches = lookup.get((str(param), str(levtype), level))
            if not matches:
                continue
            offset, length = int(raw_offset), int(raw_length)
            # Byte ranges past the data file mean a stale/mismatched index; skip it.
            if length <= 0 or offset + length > file_size:
                log.warning(
                    f"Skipping {location}: index byte range falls outside the "
                    f"{file_size}-byte data file; stale or mismatched index"
                )
                return []
            for var, level_label in matches:
                refs.append(
                    VirtualRef(
                        data_var=var,
                        out_loc={**out_loc_base, **level_label},
                        location=location,
                        offset=offset,
                        length=length,
                    )
                )
        return refs

    def _message_lookup(
        self, data_vars: Sequence[EcmwfAifsSingleVirtualDataVar]
    ) -> dict[
        tuple[str, str, int | None],
        list[tuple[EcmwfAifsSingleVirtualDataVar, dict[Dim, CoordinateValue]]],
    ]:
        """Map each index (param, levtype, levelist) key to the variables it fills and
        the vertical label each ref carries. A root var contributes one entry (soil vars
        carry their level in grib_index_level_value); a pressure_level var one per
        template level (levels absent from a file, e.g. q at 10 hPa, match nothing)."""
        lookup: dict[
            tuple[str, str, int | None],
            list[tuple[EcmwfAifsSingleVirtualDataVar, dict[Dim, CoordinateValue]]],
        ] = {}
        for var in data_vars:
            param = var.internal_attrs.grib_index_param
            levtype = var.internal_attrs.grib_index_level_type
            if var.group is ROOT:
                level_value = var.internal_attrs.grib_index_level_value
                level = None if pd.isna(level_value) else int(level_value)
                lookup.setdefault((param, levtype, level), []).append((var, {}))
            else:
                dim = var.group  # group name equals its dimension name
                for level in self.template_ds[var.path].get_index(dim):
                    lookup.setdefault((param, levtype, int(level)), []).append(
                        (var, {dim: int(level)})
                    )
        return lookup
