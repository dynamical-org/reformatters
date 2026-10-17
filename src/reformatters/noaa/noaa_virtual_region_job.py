from collections.abc import Sequence
from itertools import pairwise
from typing import Any, ClassVar, Generic, TypeVar

from httpx import HTTPError
from obstore.exceptions import BaseError

from reformatters.common.config_models import ROOT, DataVar
from reformatters.common.download import s3_download_to_disk, s3_read_bytes, s3_store
from reformatters.common.grib import GRIB_SECTION_0_BYTES, grib2_message_length
from reformatters.common.logging import get_logger
from reformatters.common.region_job import CoordinateValue, InitLeadSourceFileCoord
from reformatters.common.types import Dim, Timedelta
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,
)
from reformatters.noaa.models import NoaaInternalAttrs
from reformatters.noaa.noaa_grib_index import (
    grib_index_window_str,
    parse_grib_index_lines,
)

log = get_logger(__name__)

NOAA_DATA_VAR = TypeVar("NOAA_DATA_VAR", bound=DataVar[NoaaInternalAttrs])


class NoaaVirtualSourceFileCoord(InitLeadSourceFileCoord, Generic[NOAA_DATA_VAR]):
    """One NOAA GRIB file: the forecast step it holds and the variables it packs.

    `get_url()` must return the `s3://` location refs point at, matching the dataset's
    virtual chunk container prefix.
    """

    data_vars: Sequence[NOAA_DATA_VAR]

    def get_index_url(self) -> str:
        return self.get_url() + ".idx"

    def message_lead_times(self) -> Sequence[Timedelta]:
        return (self.lead_time,)

    def index_selectors(self, selectors: tuple[str, ...]) -> tuple[str, ...] | None:
        """The matching identity, or None to skip a malformed source message."""
        return selectors


NOAA_VIRTUAL_COORD = TypeVar(
    "NOAA_VIRTUAL_COORD", bound=NoaaVirtualSourceFileCoord[Any]
)


class NoaaVirtualRegionJob(
    VirtualRegionJob[NOAA_DATA_VAR, NOAA_VIRTUAL_COORD],
    Generic[NOAA_DATA_VAR, NOAA_VIRTUAL_COORD],
):
    """Source-file discovery on a NODD S3 bucket and ref building from the GRIB
    indexes beside those files, shared by the NOAA virtual datasets.

    A subclass sets `source_location_prefix` / `source_bucket_region` and adds
    `generate_source_file_coords` and `operational_update_window`.
    """

    # The s3:// bucket URL prefix source files live under, and the bucket's region.
    source_location_prefix: ClassVar[str]
    source_bucket_region: ClassVar[str]

    def discover_available(
        self, pending: list[NOAA_VIRTUAL_COORD]
    ) -> list[tuple[NOAA_VIRTUAL_COORD, int]]:
        return discover_available_by_obstore_listing(
            pending,
            store=s3_store(
                self.source_location_prefix, region=self.source_bucket_region
            ),
            location_prefix=self.source_location_prefix,
            require_index=True,
        )

    def owns_index_message(
        self,
        coord: NOAA_VIRTUAL_COORD,  # noqa: ARG002 - overrides key the decision on it
        element: str,  # noqa: ARG002
        level: str,  # noqa: ARG002
    ) -> bool:
        """Whether `coord`'s file is the one that supplies this index message.

        Default: every message it carries. Override where a model publishes a message
        in more than one of its products and exactly one must supply the chunk, so a
        chunk's reference names a stable (file, offset).
        """
        return True

    def file_refs(self, coord: NOAA_VIRTUAL_COORD, file_size: int) -> list[VirtualRef]:
        index_path = s3_download_to_disk(
            coord.get_index_url(), self.dataset_id, region=self.source_bucket_region
        )
        try:
            index_lines = parse_grib_index_lines(index_path, tolerate_malformed=True)
        finally:
            index_path.unlink()

        if not index_lines:
            log.error(f"Skipping {coord.get_url()}: empty or unparseable grib index")
            return []

        lookup = {}
        lead_times = coord.message_lead_times()
        for lead_time in lead_times:
            for key, matches in self._message_lookup(
                coord.data_vars, lead_time.total_seconds() / 3600
            ).items():
                lookup.setdefault(key, []).extend(
                    (
                        var,
                        {
                            **labels,
                            **({"lead_time": lead_time} if len(lead_times) > 1 else {}),
                        },
                    )
                    for var, labels in matches
                )
        lengths = self._message_lengths(
            coord, [start for start, *_ in index_lines], file_size
        )
        if not lengths:
            return []
        location = coord.get_url()
        out_loc_base = dict(coord.out_loc())
        refs = []
        filled: set[tuple[str, tuple[tuple[Dim, CoordinateValue], ...]]] = set()
        for start, element, level, window, selectors in index_lines:
            length = lengths.get(start)
            if length is None or not element:
                continue
            if not self.owns_index_message(coord, element, level):
                continue
            exact_selectors = coord.index_selectors(selectors)
            if exact_selectors is None:
                continue
            matches = [
                *lookup.get((element, level, window, exact_selectors), []),
                *lookup.get((element, level, window, None), []),
            ]
            if not matches:
                continue
            for var, level_label in matches:
                out_loc = {**out_loc_base, **level_label}
                # A source may publish byte-distinct copies of a field; the first
                # message wins, as it does when reading materialized.
                # One message filling several variables is a different case and stands.
                cell = (var.path, tuple(sorted(out_loc.items())))
                if cell in filled:
                    continue
                filled.add(cell)
                refs.append(
                    VirtualRef(
                        data_var=var,
                        out_loc=out_loc,
                        location=location,
                        offset=start,
                        length=length,
                    )
                )
        self._check_refs_complete(coord, refs)
        return refs

    def _message_lengths(
        self, coord: NOAA_VIRTUAL_COORD, starts: Sequence[int], file_size: int
    ) -> dict[int, int]:
        location = coord.get_url()
        # A message that changed size shifts every offset after it, so the last entry is
        # where any staleness shows: bounds alone cannot see it. Not an equality against
        # the file end - a healthy index may omit trailing messages the object still has.
        declared_length = self.grib_message_length_at(coord, starts[-1], file_size)
        if declared_length is None or not 0 < declared_length <= file_size - starts[-1]:
            log.error(
                f"Skipping {location}: the index's last offset {starts[-1]} does not "
                f"begin a GRIB message that fits the {file_size}-byte data file; "
                f"stale or mismatched index"
            )
            return {}

        valid_starts = [start for start in starts if 0 <= start < file_size]
        if any(a >= b for a, b in pairwise(valid_starts)):
            log.error(
                f"Skipping {location}: non-increasing index offsets; "
                "stale or mismatched index"
            )
            return {}

        lengths = {}
        # Each message's end byte is the next message's start; the last is the file end.
        for start, end in pairwise([*starts, file_size]):
            if not 0 <= start < file_size:
                log.error(
                    f"{location}: skipping index row at offset {start} outside the "
                    f"{file_size}-byte data file"
                )
                continue
            length = end - start
            if not 0 < length <= file_size - start:
                try:
                    length = self.grib_message_length_at(coord, start, file_size)
                except OSError, HTTPError, BaseError:
                    log.exception(
                        f"{location}: skipping message at {start}: failed to read "
                        "its length beside an out-of-range index offset"
                    )
                    continue
                next_start = next((s for s in valid_starts if s > start), file_size)
                if length is None or not 0 < length <= next_start - start:
                    log.error(
                        f"{location}: skipping message at {start}: cannot recover "
                        "its length beside an out-of-range index offset"
                    )
                    continue
            lengths[start] = length
        return lengths

    def _check_refs_complete(
        self, coord: NOAA_VIRTUAL_COORD, refs: list[VirtualRef]
    ) -> None:
        expected = set()
        for lead in coord.message_lead_times():
            for (_, level, _, _), matches in self._message_lookup(
                coord.data_vars, lead.total_seconds() / 3600
            ).items():
                for var, labels in matches:
                    if not self.expects_index_message(coord, var, level):
                        continue
                    loc = {**coord.out_loc(), **labels}
                    if len(coord.message_lead_times()) > 1:
                        loc["lead_time"] = lead
                    expected.add((var.path, tuple(sorted(loc.items()))))
        actual = [
            (ref.data_var.path, tuple(sorted(ref.out_loc.items()))) for ref in refs
        ]
        assert len(actual) == len(set(actual)), (
            f"{coord.get_url()}: duplicate output references"
        )
        if missing := expected - set(actual):
            log.error(
                f"{coord.get_url()}: GRIB messages absent from index: "
                f"{sorted(missing, key=str)}; ingesting available messages"
            )

    def expects_index_message(
        self, coord: NOAA_VIRTUAL_COORD, var: NOAA_DATA_VAR, level: str
    ) -> bool:
        """Whether this source product publishes an owned message at this level."""
        return self.owns_index_message(coord, var.internal_attrs.grib_element, level)

    def grib_message_length_at(
        self, coord: NOAA_VIRTUAL_COORD, offset: int, file_size: int
    ) -> int | None:
        """The edition 2 message length the data file declares at `offset`, read from the
        object itself and so independent of the index, or None if no such message starts
        there. A stale index can put `offset` past the end, which a ranged GET rejects.
        """
        if offset < 0 or offset + GRIB_SECTION_0_BYTES > file_size:
            return None
        header = s3_read_bytes(
            coord.get_url(),
            region=self.source_bucket_region,
            start=offset,
            end=offset + GRIB_SECTION_0_BYTES,
        )
        return grib2_message_length(header)

    def _message_lookup(
        self, data_vars: Sequence[NOAA_DATA_VAR], lead_hours: float
    ) -> dict[
        tuple[str, str, str, tuple[str, ...] | None],
        list[tuple[NOAA_DATA_VAR, dict[Dim, CoordinateValue]]],
    ]:
        """Map each (element, idx level string, idx window string, selectors) to the
        variables it fills and the vertical label each ref carries. A root var contributes
        one entry; a vertical-group var one per level (its grib_index_level is a "{level:g} ..."
        format string). The mapping is one-to-many: a run-total and a per-hour
        accumulation variant render the same window string at the reset lead, and both
        must be filled from the single matching message.
        """
        lookup: dict[
            tuple[str, str, str, tuple[str, ...] | None],
            list[tuple[NOAA_DATA_VAR, dict[Dim, CoordinateValue]]],
        ] = {}
        for var in data_vars:
            window = grib_index_window_str(var, lead_hours)
            # Index element spellings vary by era and by the wgrib2 build that made the
            # index (e.g. TCOLWold, CLWMR, raw "var discipline=..." strings), so match
            # grib_element_alternatives too; a file carries only one spelling.
            elements = (
                var.internal_attrs.grib_element,
                *var.internal_attrs.grib_element_alternatives,
            )
            for element in elements:
                if var.group is ROOT:
                    key = (
                        element,
                        var.internal_attrs.grib_index_level,
                        window,
                        var.internal_attrs.grib_index_selectors,
                    )
                    lookup.setdefault(key, []).append((var, {}))
                else:
                    dim = var.group  # group name equals its dimension name
                    level_format = var.internal_attrs.grib_index_level
                    for level in self.template_ds[var.path].get_index(dim):
                        key = (
                            element,
                            level_format.format(level=level),
                            window,
                            var.internal_attrs.grib_index_selectors,
                        )
                        lookup.setdefault(key, []).append((var, {dim: level}))
        return lookup
