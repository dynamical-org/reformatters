from collections.abc import Iterator, Mapping, Sequence
from itertools import batched
from typing import Any, ClassVar, Literal, cast

import pandas as pd
import xarray as xr

from reformatters.common.config_models import ROOT, DataVar
from reformatters.common.download import gcs_download_to_disk
from reformatters.common.logging import get_logger
from reformatters.common.region_job import CoordinateValue
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Dim, Timedelta
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.ecmwf.ecmwf_grib_index import parse_index_file
from reformatters.ecmwf.ecmwf_virtual_source import (
    GCS_LOCATION_PREFIX,
    EcmwfVirtualSourceFileCoord,
    discover_ecmwf_available,
)

from .template_config import EcmwfAifsEnsVirtualDataVar, available_pressure_levels

log = get_logger(__name__)


def var_levels(
    var: EcmwfAifsEnsVirtualDataVar, levels: Sequence[int]
) -> tuple[int, ...]:
    return tuple(
        level
        for level in levels
        if level != 10 or var.internal_attrs.grib_index_param != "q"
    )


class EcmwfAifsEnsForecastVirtualSourceFileCoord(EcmwfVirtualSourceFileCoord):
    data_vars: Sequence[EcmwfAifsEnsVirtualDataVar]
    file_type: Literal["cf", "pf"]
    ensemble_members: tuple[int, ...]
    pressure_levels: tuple[int, ...]

    def _get_base_url(self) -> str:
        return (
            f"{GCS_LOCATION_PREFIX}{self.init_time:%Y%m%d}/{self.init_time:%H}z/"
            f"aifs-ens/0p25/enfo/"
            f"{self.init_time:%Y%m%d%H}0000-{whole_hours(self.lead_time)}h-enfo-{self.file_type}"
        )

    def get_url(self) -> str:
        return self._get_base_url() + ".grib2"

    def get_index_url(self) -> str:
        return self._get_base_url() + ".index"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"init_time": self.init_time, "lead_time": self.lead_time}

    def variable_out_loc(
        self,
        var: DataVar[Any],  # noqa: ARG002 - all variables cover the file's members
    ) -> Mapping[Dim, CoordinateValue | tuple[CoordinateValue, ...]]:
        return {**self.out_loc(), "ensemble_member": self.ensemble_members}

    def levels_for(self, var: EcmwfAifsEnsVirtualDataVar) -> tuple[int, ...]:
        return var_levels(var, self.pressure_levels)


class EcmwfAifsEnsForecastVirtualRegionJob(
    VirtualRegionJob[
        EcmwfAifsEnsVirtualDataVar, EcmwfAifsEnsForecastVirtualSourceFileCoord
    ]
):
    backfill_files_per_batch: ClassVar[int] = 400
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("30h")

    def process_virtual_refs(
        self, remaining: Sequence[EcmwfAifsEnsForecastVirtualSourceFileCoord]
    ) -> Iterator[
        Sequence[
            tuple[EcmwfAifsEnsForecastVirtualSourceFileCoord, Sequence[VirtualRef]]
        ]
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
        data_var_group: Sequence[EcmwfAifsEnsVirtualDataVar],
    ) -> Sequence[EcmwfAifsEnsForecastVirtualSourceFileCoord]:
        coords = []
        members = tuple(int(m) for m in processing_region_ds.ensemble_member.values)
        output_levels = (
            {int(p) for p in processing_region_ds.pressure_level.values}
            if "pressure_level" in processing_region_ds.coords
            else set()
        )
        for init_time in pd.to_datetime(processing_region_ds.init_time.values):
            levels = tuple(
                p for p in available_pressure_levels(init_time) if p in output_levels
            )
            for lead_time in pd.to_timedelta(processing_region_ds.lead_time.values):
                file_vars = [
                    var
                    for var in data_var_group
                    if var.available(init_time, lead_time)
                    and (var.group is ROOT or var_levels(var, levels))
                ]
                if not file_vars:
                    continue
                for file_type in ("cf", "pf"):
                    file_members = tuple(
                        m for m in members if (m == 0) == (file_type == "cf")
                    )
                    if file_members:
                        coords.append(
                            EcmwfAifsEnsForecastVirtualSourceFileCoord(
                                init_time=init_time,
                                lead_time=lead_time,
                                file_type=file_type,
                                data_vars=file_vars,
                                ensemble_members=file_members,
                                pressure_levels=levels,
                            )
                        )
        return coords

    def discover_available(
        self, pending: list[EcmwfAifsEnsForecastVirtualSourceFileCoord]
    ) -> list[tuple[EcmwfAifsEnsForecastVirtualSourceFileCoord, int]]:
        return discover_ecmwf_available(pending, s3_fallback_before=None)

    def representative_var(
        self, coord: EcmwfAifsEnsForecastVirtualSourceFileCoord
    ) -> EcmwfAifsEnsVirtualDataVar:
        return min(
            coord.data_vars,
            key=lambda var: (var.group is not ROOT, var.attrs.step_type != "instant"),
        )

    def representative_probe_loc(
        self, coord: EcmwfAifsEnsForecastVirtualSourceFileCoord, var: DataVar[Any]
    ) -> Mapping[Dim, CoordinateValue]:
        assert isinstance(var, EcmwfAifsEnsVirtualDataVar)
        return {
            **coord.out_loc(),
            "ensemble_member": coord.ensemble_members[0],
            **(
                {"pressure_level": coord.levels_for(var)[0]}
                if var.group is not ROOT
                else {}
            ),
        }

    def message_locations(
        self,
        coord: EcmwfAifsEnsForecastVirtualSourceFileCoord,
        var: EcmwfAifsEnsVirtualDataVar,
    ) -> list[Mapping[Dim, CoordinateValue]]:
        return [
            {
                **coord.out_loc(),
                "ensemble_member": member,
                **({"pressure_level": level} if level is not None else {}),
            }
            for member in coord.ensemble_members
            for level in (coord.levels_for(var) if var.group is not ROOT else (None,))
        ]

    def file_refs(
        self, coord: EcmwfAifsEnsForecastVirtualSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        index_path = gcs_download_to_disk(coord.get_index_url(), self.dataset_id)
        try:
            index = parse_index_file(index_path, ensemble=True)
        finally:
            index_path.unlink()
        entries = [
            (
                int(number),
                str(param),
                str(levtype),
                None if pd.isna(levelist) else int(levelist),
                int(offset),
                int(length),
            )
            for number, param, levtype, levelist, offset, length in index.reset_index()[
                ["number", "param", "levtype", "levelist", "_offset", "_length"]
            ].itertuples(index=False, name=None)
        ]
        if any(
            offset < 0 or length <= 0 or offset + length > file_size
            for *_, offset, length in entries
        ):
            log.warning(
                f"Skipping {coord.get_url()}: index byte range outside {file_size}-byte data file"
            )
            return []
        lookup: dict[
            tuple[int, str, str, int | None],
            tuple[EcmwfAifsEnsVirtualDataVar, Mapping[Dim, CoordinateValue]],
        ] = {}
        for var in coord.data_vars:
            attrs = var.internal_attrs
            for loc in self.message_locations(coord, var):
                if var.group is not ROOT:
                    level = cast("int", loc["pressure_level"])
                else:
                    level = (
                        None
                        if pd.isna(attrs.grib_index_level_value)
                        else int(attrs.grib_index_level_value)
                    )
                key = (
                    cast("int", loc["ensemble_member"]),
                    attrs.grib_index_param,
                    attrs.grib_index_level_type,
                    level,
                )
                assert key not in lookup, f"Duplicate source selector {key}"
                lookup[key] = var, loc
        refs = []
        matched: set[tuple[int, str, str, int | None]] = set()
        for number, param, levtype, level, offset, length in entries:
            key = (number, param, levtype, level)
            match = lookup.get(key)
            if match is None:
                continue
            assert key not in matched, f"Duplicate source selector {key}"
            matched.add(key)
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
