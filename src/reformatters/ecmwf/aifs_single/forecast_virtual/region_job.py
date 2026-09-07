import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, cast

import icechunk
import pandas as pd
import xarray as xr

from reformatters.common.config_models import ROOT
from reformatters.common.download import s3_download_to_disk, s3_store
from reformatters.common.region_job import (
    CoordinateValue,
    InitLeadSourceFileCoord,
)
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Dim, Timedelta
from reformatters.common.virtual_region_job import (
    SourceFileRejectedError,
    VirtualRef,
    VirtualRegionJob,
)
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,
)
from reformatters.ecmwf.aifs_single.template_config import (
    aifs_single_stream_path,
)

from .template_config import (
    EcmwfAifsSingleVirtualDataVar,
)

SOURCE_LOCATION_PREFIX = "s3://ecmwf-forecasts/"
SOURCE_REGION = "eu-central-1"
_IndexRow = tuple[str, str, object, int, int]


def _exact_index_integer(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    raise ValueError(f"non-integer index field {value!r}")


def _raw_index_rows(index_path: Path) -> list[dict[str, object]]:
    rows = []
    for line in index_path.read_text().splitlines():
        if not line.strip():
            continue
        row: object = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError("GRIB index row is not an object")
        typed_row = cast("dict[str, object]", row)
        for field in ("param", "levtype", "_offset", "_length"):
            typed_row[field]
        rows.append(typed_row)
    return rows


def _validated_index_rows(index_path: Path, file_size: int) -> list[_IndexRow]:
    try:
        raw_rows = _raw_index_rows(index_path)
    except (
        json.JSONDecodeError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
    ) as error:
        raise SourceFileRejectedError("empty or unparseable GRIB index") from error
    try:
        rows = [
            (
                str(row["param"]),
                str(row["levtype"]),
                row.get("levelist"),
                _exact_index_integer(row["_offset"]),
                _exact_index_integer(row["_length"]),
            )
            for row in raw_rows
        ]
    except (TypeError, ValueError) as error:
        raise SourceFileRejectedError("invalid GRIB index row fields") from error
    if not rows:
        raise SourceFileRejectedError("empty or unparseable GRIB index")
    if any(
        not 0 <= offset < offset + length <= file_size for *_, offset, length in rows
    ):
        raise SourceFileRejectedError(
            f"index byte range falls outside the {file_size}-byte data file; "
            "stale or mismatched index"
        )
    return rows


def _matching_index_level(
    raw_level: object, candidates: set[int | None]
) -> tuple[bool, int | None]:
    if raw_level is None:
        return None in candidates, None
    try:
        level = _exact_index_integer(raw_level)
    except TypeError, ValueError:
        if any(raw_level == candidate for candidate in candidates):
            raise
        return False, None
    return level in candidates, level


def aifs_single_virtual_chunk_containers() -> tuple[
    icechunk.VirtualChunkContainer, ...
]:
    """Fresh container objects per call; icechunk containers can't be shared
    pydantic defaults."""
    return (
        icechunk.VirtualChunkContainer(
            SOURCE_LOCATION_PREFIX, icechunk.s3_store(region=SOURCE_REGION)
        ),
    )


class EcmwfAifsSingleForecastVirtualSourceFileCoord(InitLeadSourceFileCoord):
    """One AIFS Single forecast file (init_time, lead_time) and the vars it packs."""

    data_vars: Sequence[EcmwfAifsSingleVirtualDataVar]

    def _get_base_url(self) -> str:
        stream_path = aifs_single_stream_path(self.init_time)
        init_date_str = self.init_time.strftime("%Y%m%d")
        init_hour_str = self.init_time.strftime("%H")
        return (
            f"{SOURCE_LOCATION_PREFIX}{init_date_str}/{init_hour_str}z/"
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
        return discover_available_by_obstore_listing(
            pending,
            store=s3_store(SOURCE_LOCATION_PREFIX, region=SOURCE_REGION),
            location_prefix=SOURCE_LOCATION_PREFIX,
            require_index=True,
        )

    def file_refs(
        self,
        coord: EcmwfAifsSingleForecastVirtualSourceFileCoord,
        file_size: int,
    ) -> list[VirtualRef]:
        index_path = s3_download_to_disk(
            coord.get_index_url(), self.dataset_id, region=SOURCE_REGION
        )
        try:
            index_rows = _validated_index_rows(index_path, file_size)
        finally:
            index_path.unlink()

        lookup = self._message_lookup(coord.data_vars)
        requested_levels: dict[tuple[str, str], set[int | None]] = {}
        for param, levtype, level in lookup:
            requested_levels.setdefault((param, levtype), set()).add(level)
        out_loc_base = dict(coord.out_loc())
        location = coord.get_url()
        refs = []
        for param, levtype, raw_level, offset, length in index_rows:
            level_candidates = requested_levels.get((param, levtype))
            if level_candidates is None:
                continue
            try:
                level_matches, level = _matching_index_level(
                    raw_level, level_candidates
                )
            except (TypeError, ValueError) as error:
                raise SourceFileRejectedError(
                    "invalid GRIB index row fields"
                ) from error
            if not level_matches:
                continue
            matches = lookup.get((param, levtype, level))
            if not matches:
                continue
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
