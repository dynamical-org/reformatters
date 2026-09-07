import struct
from collections.abc import Sequence
from typing import Any, ClassVar, Generic, TypeVar

from reformatters.common.config_models import ROOT, DataVar
from reformatters.common.download import s3_download_to_disk, s3_read_bytes, s3_store
from reformatters.common.logging import get_logger
from reformatters.common.region_job import CoordinateValue, InitLeadSourceFileCoord
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Dim
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

# GRIB2 section 0: b"GRIB", 2 reserved bytes, discipline, edition, then the message's
# total length as a big endian u64.
GRIB_SECTION_0_BYTES = 16


NOAA_DATA_VAR = TypeVar("NOAA_DATA_VAR", bound=DataVar[NoaaInternalAttrs])


class NoaaVirtualSourceFileCoord(InitLeadSourceFileCoord, Generic[NOAA_DATA_VAR]):
    """One NOAA GRIB file: the forecast step it holds and the variables it packs.

    `get_url()` must return the `s3://` location refs point at, matching the dataset's
    virtual chunk container prefix.
    """

    data_vars: Sequence[NOAA_DATA_VAR]

    def get_index_url(self) -> str:
        return self.get_url() + ".idx"


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
            index_lines = parse_grib_index_lines(index_path)
        finally:
            index_path.unlink()

        if not index_lines:
            log.warning(f"Skipping {coord.get_url()}: empty or unparseable grib index")
            return []

        lookup = self._message_lookup(coord.data_vars, whole_hours(coord.lead_time))
        # Each message's end byte is the next message's start; the last is the file end.
        starts = [start for start, *_ in index_lines]
        ends = [*starts[1:], file_size]

        location = coord.get_url()
        # A message that changed size shifts every offset after it, so the last entry is
        # where any staleness shows: bounds alone cannot see it. Not an equality against
        # the file end - a healthy index may omit trailing messages the object still has.
        declared_length = self.grib_message_length_at(coord, starts[-1], file_size)
        if declared_length is None or declared_length > file_size - starts[-1]:
            log.error(
                f"Skipping {location}: the index's last offset {starts[-1]} does not "
                f"begin a GRIB message that fits the {file_size}-byte data file; "
                f"stale or mismatched index"
            )
            return []

        out_loc_base = dict(coord.out_loc())
        refs = []
        filled: set[tuple[str, tuple[tuple[Dim, CoordinateValue], ...]]] = set()
        for (start, element, level, window), end in zip(index_lines, ends, strict=True):
            # Byte ranges past the data file mean a stale/mismatched index; skip it.
            # Checked for every message, matched or not: the whole file is discarded,
            # so a corrupt range anywhere in the index condemns all of it.
            if end > file_size or end <= start:
                log.error(
                    f"Skipping {location}: index byte ranges fall outside the "
                    f"{file_size}-byte data file; stale or mismatched index"
                )
                return []
            if not self.owns_index_message(coord, element, level):
                continue
            matches = lookup.get((element, level, window))
            if not matches:
                continue
            for var, level_label in matches:
                out_loc = {**out_loc_base, **level_label}
                # A source may publish the same field twice (byte-distinct, identical
                # values); the first message wins, as it does when reading materialized.
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
                        length=end - start,
                    )
                )
        self._check_refs_complete(coord, refs)
        return refs

    def _check_refs_complete(
        self, coord: NOAA_VIRTUAL_COORD, refs: list[VirtualRef]
    ) -> None:
        """Hook for a subclass to reject a file whose index matched only some of the
        coord's variables. Reached only once the index parsed and every byte range
        checked out, so an empty `refs` here means nothing matched, not a skipped file.
        """

    def grib_message_length_at(
        self, coord: NOAA_VIRTUAL_COORD, offset: int, file_size: int
    ) -> int | None:
        """The edition 2 message length the data file declares at `offset`, read from the
        object itself and so independent of the index, or None if no such message starts
        there. A stale index can put `offset` past the end, which a ranged GET rejects.
        """
        if offset + GRIB_SECTION_0_BYTES > file_size:
            return None
        header = s3_read_bytes(
            coord.get_url(),
            region=self.source_bucket_region,
            start=offset,
            end=offset + GRIB_SECTION_0_BYTES,
        )
        if header[:4] != b"GRIB" or header[7] != 2:
            return None
        (length,) = struct.unpack(">Q", header[8:GRIB_SECTION_0_BYTES])
        return length

    def _message_lookup(
        self, data_vars: Sequence[NOAA_DATA_VAR], lead_hours: int
    ) -> dict[
        tuple[str, str, str], list[tuple[NOAA_DATA_VAR, dict[Dim, CoordinateValue]]]
    ]:
        """Map each (element, idx level string, idx window string) to the variables it
        fills and the vertical label each ref carries. A root var contributes one entry;
        a vertical-group var one per level (its grib_index_level is a "{level:g} ..."
        format string). The mapping is one-to-many: a run-total and a per-hour
        accumulation variant render the same window string at the reset lead, and both
        must be filled from the single matching message.
        """
        lookup: dict[
            tuple[str, str, str],
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
                    key = (element, var.internal_attrs.grib_index_level, window)
                    lookup.setdefault(key, []).append((var, {}))
                else:
                    dim = var.group  # group name equals its dimension name
                    level_format = var.internal_attrs.grib_index_level
                    for level in self.template_ds[var.path].get_index(dim):
                        key = (element, level_format.format(level=level), window)
                        lookup.setdefault(key, []).append((var, {dim: level}))
        return lookup
