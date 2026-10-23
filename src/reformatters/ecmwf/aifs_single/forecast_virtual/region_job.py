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
from reformatters.ecmwf.ecmwf_grib_index import iter_index_file_rows
from reformatters.ecmwf.ecmwf_utils import EcmwfOpenDataSource

from .template_config import (
    AIFS_2026_UPGRADE_DATE,
    EcmwfAifsSingleVirtualDataVar,
)

log = get_logger(__name__)

GCS_LOCATION_PREFIX = "gs://ecmwf-open-data/"
S3_LOCATION_PREFIX = "s3://ecmwf-forecasts/"
S3_REGION = "eu-central-1"
# Inits before this date may reference s3://ecmwf-forecasts/ for files the GCS
# mirror lacks; inits on or after it reference GCS only.
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
        self._source = "s3"

    def _get_base_url(self) -> str:
        match self.source:
            case "gcs":
                location_prefix = GCS_LOCATION_PREFIX
            case "s3":
                assert self.init_time < S3_FALLBACK_BEFORE, (
                    f"{self.init_time} is on or after {S3_FALLBACK_BEFORE}; only "
                    "older inits may reference S3"
                )
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
    # At the init+5h42m fire, 20h covers the newest init and two prior cycles
    # so a couple of missed runs still self-heal.
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
        is switched to its S3 URL in place. `pending` must be GCS-sourced: a coord
        already switched to S3 is never offered again."""
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
        lookup = self._message_lookup(coord.data_vars)
        out_loc_base = dict(coord.out_loc())
        location = coord.get_url()
        refs = []
        found = set()
        previous_end = 0
        try:
            for row in iter_index_file_rows(index_path):
                try:
                    offset = _index_integer(row["_offset"])
                    length = _index_integer(row["_length"])
                    if (
                        offset < previous_end
                        or length <= 0
                        or offset + length > file_size
                    ):
                        raise ValueError(
                            f"byte range {offset}+{length} overlaps or precedes "
                            f"previous end {previous_end}, or falls outside {file_size}-byte file"
                        )
                except (KeyError, TypeError, ValueError) as exc:
                    log.error(
                        f"Skipping {location}: untrustworthy index byte geometry: {row!r}: {exc}"
                    )
                    return []
                previous_end = offset + length
                try:
                    param, levtype = row["param"], row["levtype"]
                    if not isinstance(param, str) or not isinstance(levtype, str):
                        raise ValueError("param and levtype must be strings")
                    raw_level = row.get("levelist")
                    level = None if raw_level is None else _index_integer(raw_level)
                    if (
                        not param
                        or not levtype
                        or (levtype in {"pl", "sol"} and level is None)
                    ):
                        raise ValueError("missing message selector")
                except (KeyError, TypeError, ValueError) as exc:
                    log.error(f"Skipping index row for {location}: {row!r}: {exc}")
                    continue
                key = (param, levtype, level)
                for var, level_label in lookup.get(key, []):
                    refs.append(
                        VirtualRef(
                            data_var=var,
                            out_loc={**out_loc_base, **level_label},
                            location=location,
                            offset=offset,
                            length=length,
                        )
                    )
                    found.add(key)
        finally:
            index_path.unlink()
        missing = {
            key
            for key in lookup.keys() - found
            if not (
                key[1] == "pl"
                and key[2] == 10
                and (key[0] == "q" or coord.init_time < AIFS_2026_UPGRADE_DATE)
            )
        }
        if missing:
            log.error(
                f"Missing {len(missing)} expected messages in {location}: {sorted(missing, key=str)}"
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


def _index_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"expected integer, got {value!r}")
    return int(value)
