from collections.abc import Iterator, Mapping, Sequence
from itertools import batched
from typing import Any, ClassVar, Literal, cast

import icechunk
import pandas as pd
import xarray as xr

from reformatters.common.config_models import ROOT, DataVar
from reformatters.common.download import gcs_download_to_disk, s3_download_to_disk
from reformatters.common.logging import get_logger
from reformatters.common.region_job import CoordinateValue
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Dim, Timedelta
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,
)
from reformatters.ecmwf.ecmwf_grib_index import parse_index_file
from reformatters.ecmwf.ecmwf_virtual_source import (
    GCS_LOCATION_PREFIX,
    S3_LOCATION_PREFIX,
    S3_REGION,
    EcmwfVirtualSourceFileCoord,
    discover_ecmwf_available,
    ecmwf_virtual_chunk_containers,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.source_file_coord import (
    IFS_CYCLE_50R1_CUTOVER,
)

from .virtual_template_config import (
    EcmwfIfsEnsVirtualDataVar,
    available_pressure_levels,
)

log = get_logger(__name__)
S3_FALLBACK_BEFORE = pd.Timestamp("2025-02-13T00")


def ifs_ens_virtual_chunk_containers(
    *, fine_resolution: bool
) -> tuple[icechunk.VirtualChunkContainer, ...]:
    return ecmwf_virtual_chunk_containers(
        s3_fallback_before=None if fine_resolution else S3_FALLBACK_BEFORE
    )


class EcmwfIfsEnsForecastVirtualSourceFileCoord(EcmwfVirtualSourceFileCoord):
    data_vars: Sequence[EcmwfIfsEnsVirtualDataVar]
    resolution: Literal["0p25", "0p1"]
    family: Literal["enfo", "oper"]
    level_family: Literal["sfc", "pl"] | None = None
    ensemble_members: tuple[int, ...]
    pressure_levels: tuple[int, ...]

    def _get_base_url(self) -> str:
        assert self.source == "gcs" or (
            self.resolution == "0p25" and self.init_time < S3_FALLBACK_BEFORE
        ), "Only older 0.25 degree inits may reference S3"
        location_prefix = (
            GCS_LOCATION_PREFIX if self.source == "gcs" else S3_LOCATION_PREFIX
        )
        file_type = "ef" if self.family == "enfo" else "fc"
        suffix = f"-{self.level_family}" if self.level_family else ""
        return (
            f"{location_prefix}{self.init_time:%Y%m%d}/{self.init_time:%H}z/"
            f"ifs/{self.resolution}/{self.family}/"
            f"{self.init_time:%Y%m%d%H}0000-{whole_hours(self.lead_time)}h-{self.family}-{file_type}{suffix}"
        )

    def get_url(self) -> str:
        return self._get_base_url() + ".grib2"

    def get_index_url(self) -> str:
        return self._get_base_url() + ".index"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"init_time": self.init_time, "lead_time": self.lead_time}

    def members_for(self, var: EcmwfIfsEnsVirtualDataVar) -> tuple[int, ...]:
        return tuple(
            m
            for m in self.ensemble_members
            if not var.internal_attrs.control_only or m == 0
        )


class EcmwfIfsEnsForecastVirtualRegionJob(
    VirtualRegionJob[
        EcmwfIfsEnsVirtualDataVar, EcmwfIfsEnsForecastVirtualSourceFileCoord
    ]
):
    resolution: ClassVar[Literal["0p25", "0p1"]] = "0p25"
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("30h")
    download_concurrency: ClassVar[int] = 8

    backfill_files_per_batch: ClassVar[int] = 100

    def process_virtual_refs(
        self,
        remaining: Sequence[EcmwfIfsEnsForecastVirtualSourceFileCoord],
    ) -> Iterator[
        Sequence[tuple[EcmwfIfsEnsForecastVirtualSourceFileCoord, Sequence[VirtualRef]]]
    ]:
        batches = (
            batched(remaining, self.backfill_files_per_batch, strict=False)
            if self.processing_mode == "backfill"
            else (remaining,)
        )
        for batch in batches:
            yield from super().process_virtual_refs(batch)

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[EcmwfIfsEnsVirtualDataVar],
    ) -> Sequence[EcmwfIfsEnsForecastVirtualSourceFileCoord]:
        coords = []
        members = tuple(int(m) for m in processing_region_ds.ensemble_member.values)
        output_levels = (
            {int(p) for p in processing_region_ds.pressure_level.values}
            if "pressure_level" in processing_region_ds.coords
            else set()
        )
        for init_time in pd.to_datetime(processing_region_ds.init_time.values):
            separate_control = init_time >= IFS_CYCLE_50R1_CUTOVER
            levels = tuple(
                p for p in available_pressure_levels(init_time) if p in output_levels
            )
            for lead_time in pd.to_timedelta(processing_region_ds.lead_time.values):
                for family in ("enfo", "oper") if separate_control else ("enfo",):
                    file_members = (
                        tuple(m for m in members if (m == 0) == (family == "oper"))
                        if separate_control
                        else members
                    )
                    if not file_members:
                        continue
                    for level_family in (
                        ("sfc", "pl") if self.resolution == "0p1" else (None,)
                    ):
                        file_vars = [
                            var
                            for var in data_var_group
                            if var.available(init_time, lead_time)
                            and (
                                not var.internal_attrs.control_only or 0 in file_members
                            )
                            and (
                                level_family is None
                                or (var.group == "pressure_level")
                                == (level_family == "pl")
                            )
                            and (var.group is ROOT or levels)
                        ]
                        if file_vars:
                            coords.append(
                                EcmwfIfsEnsForecastVirtualSourceFileCoord(
                                    init_time=init_time,
                                    lead_time=lead_time,
                                    resolution=self.resolution,
                                    family=family,
                                    level_family=level_family,
                                    data_vars=file_vars,
                                    ensemble_members=file_members,
                                    pressure_levels=levels,
                                )
                            )
        return coords

    def discover_available(
        self, pending: list[EcmwfIfsEnsForecastVirtualSourceFileCoord]
    ) -> list[tuple[EcmwfIfsEnsForecastVirtualSourceFileCoord, int]]:
        return discover_ecmwf_available(
            pending,
            s3_fallback_before=None if self.resolution == "0p1" else S3_FALLBACK_BEFORE,
            listing=discover_available_by_obstore_listing,
        )

    def representative_var(
        self, coord: EcmwfIfsEnsForecastVirtualSourceFileCoord
    ) -> EcmwfIfsEnsVirtualDataVar:
        return min(
            coord.data_vars,
            key=lambda var: (
                var.internal_attrs.control_only,
                var.group is not ROOT,
                var.attrs.step_type != "instant",
            ),
        )

    def representative_probe_loc(
        self,
        coord: EcmwfIfsEnsForecastVirtualSourceFileCoord,
        var: DataVar[Any],
    ) -> Mapping[Dim, CoordinateValue]:
        assert isinstance(var, EcmwfIfsEnsVirtualDataVar)
        members = coord.members_for(var)
        # A damaged control block must not suppress intact perturbed members.
        member = next((m for m in members if m != 0), members[0])
        return {
            **coord.out_loc(),
            "ensemble_member": member,
            **(
                {"pressure_level": coord.pressure_levels[0]}
                if var.group is not ROOT
                else {}
            ),
        }

    def message_locations(
        self,
        coord: EcmwfIfsEnsForecastVirtualSourceFileCoord,
        var: EcmwfIfsEnsVirtualDataVar,
    ) -> list[Mapping[Dim, CoordinateValue]]:
        return [
            {
                **coord.out_loc(),
                "ensemble_member": member,
                **({"pressure_level": level} if level is not None else {}),
            }
            for member in coord.members_for(var)
            for level in (coord.pressure_levels if var.group is not ROOT else (None,))
        ]

    def file_refs(
        self, coord: EcmwfIfsEnsForecastVirtualSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        index_path = (
            gcs_download_to_disk(coord.get_index_url(), self.dataset_id)
            if coord.source == "gcs"
            else s3_download_to_disk(
                coord.get_index_url(), self.dataset_id, region=S3_REGION
            )
        )
        try:
            index = parse_index_file(index_path, ensemble=True)
        finally:
            index_path.unlink()
        lookup: dict[
            tuple[int, str, str, int | None],
            tuple[EcmwfIfsEnsVirtualDataVar, Mapping[Dim, CoordinateValue]],
        ] = {}
        for var in coord.data_vars:
            param, level_type, source_level = var.source_selector(
                coord.init_time, coord.lead_time
            )
            for loc in self.message_locations(coord, var):
                level = (
                    cast("int", loc["pressure_level"])
                    if var.group is not ROOT
                    else source_level
                )
                key = (cast("int", loc["ensemble_member"]), param, level_type, level)
                assert key not in lookup, f"Duplicate source selector {key}"
                lookup[key] = var, loc
        refs = []
        entries = index.reset_index()
        for number, param, levtype, levelist, raw_offset, raw_length in zip(
            entries["number"],
            entries["param"],
            entries["levtype"],
            entries["levelist"],
            entries["_offset"],
            entries["_length"],
            strict=True,
        ):
            level = None if pd.isna(levelist) else int(levelist)
            match = lookup.get((int(number), str(param), str(levtype), level))
            if match is None:
                continue
            offset, length = int(raw_offset), int(raw_length)
            if offset < 0 or length <= 0 or offset + length > file_size:
                log.warning(
                    f"Skipping {coord.get_url()}: index byte range outside {file_size}-byte data file"
                )
                return []
            var, loc = match
            refs.append(
                VirtualRef(
                    data_var=var,
                    out_loc=loc,
                    location=coord.get_url(),
                    offset=offset,
                    length=length,
                )
            )
        return refs
