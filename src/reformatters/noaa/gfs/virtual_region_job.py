from collections.abc import Sequence
from typing import Generic, TypeVar

import icechunk

from reformatters.common.config_models import ROOT
from reformatters.common.download import s3_download_to_disk, s3_store
from reformatters.common.logging import get_logger
from reformatters.common.region_job import CoordinateValue
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Dim
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,
)
from reformatters.noaa.gfs.region_job import (
    DownloadSource,
    NoaaGfsFileType,
    NoaaGfsSourceFileCoord,
)
from reformatters.noaa.models import NoaaDataVar
from reformatters.noaa.noaa_grib_index import _lead_time_str, parse_grib_index_lines

log = get_logger(__name__)

S3_LOCATION_PREFIX = "s3://noaa-gfs-bdp-pds/"
S3_BUCKET_REGION = "us-east-1"
_S3_HTTPS_PREFIX = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"

# Every message of both products is curated, so a step is complete only once both
# files are read.
GFS_FILE_TYPES: tuple[NoaaGfsFileType, ...] = ("pgrb2", "pgrb2b")

# 41 messages that pgrb2b repeats byte for byte from pgrb2. Building refs from both
# would write two byte ranges for one array position, so a pgrb2b index skips these and
# the array is filled from pgrb2. All 41 are instantaneous and uniquely identified by
# (element, idx level string) at every lead; test_pgrb2_preferred_messages pins the set
# against real indexes.
PGRB2_PREFERRED_MESSAGES: frozenset[tuple[str, str]] = frozenset(
    [
        *(
            (element, f"{level:g} mb")
            for element in ("HGT", "TMP", "RH", "UGRD", "VGRD", "ABSV", "O3MR")
            for level in (1, 2, 3, 5, 7)
        ),
        ("CNWAT", "surface"),
        ("ICETK", "surface"),
        ("SOILL", "0-0.1 m below ground"),
        ("SOILL", "0.1-0.4 m below ground"),
        ("SOILL", "0.4-1 m below ground"),
        ("SOILL", "1-2 m below ground"),
    ]
)


def gfs_virtual_chunk_containers() -> tuple[icechunk.VirtualChunkContainer, ...]:
    """Fresh container objects per call; icechunk containers can't be shared
    pydantic defaults."""
    return (
        icechunk.VirtualChunkContainer(
            S3_LOCATION_PREFIX, icechunk.s3_store(region=S3_BUCKET_REGION)
        ),
    )


class NoaaGfsVirtualSourceFileCoord(NoaaGfsSourceFileCoord):
    """One GFS product file (init_time, lead_time, file_type) and the vars it may fill."""

    def get_url(self, source: DownloadSource = "s3") -> str:
        """The s3:// source location refs point at, matching the virtual chunk container."""
        url = super().get_url(source=source)
        assert url.startswith(_S3_HTTPS_PREFIX), url
        return S3_LOCATION_PREFIX + url.removeprefix(_S3_HTTPS_PREFIX)

    def get_index_url(self) -> str:
        return self.get_url() + ".idx"


GFS_VIRTUAL_COORD = TypeVar("GFS_VIRTUAL_COORD", bound=NoaaGfsVirtualSourceFileCoord)


class NoaaGfsVirtualRegionJob(
    VirtualRegionJob[NoaaDataVar, GFS_VIRTUAL_COORD],
    Generic[GFS_VIRTUAL_COORD],
):
    """Source-file discovery on NODD S3 and ref building from GRIB indexes, shared by
    the GFS virtual datasets. A subclass adds generate_source_file_coords and
    operational_update_window."""

    def discover_available(
        self, pending: list[GFS_VIRTUAL_COORD]
    ) -> list[tuple[GFS_VIRTUAL_COORD, int]]:
        return discover_available_by_obstore_listing(
            pending,
            store=s3_store(S3_LOCATION_PREFIX, region=S3_BUCKET_REGION),
            location_prefix=S3_LOCATION_PREFIX,
            require_index=True,
        )

    def file_refs(self, coord: GFS_VIRTUAL_COORD, file_size: int) -> list[VirtualRef]:
        index_path = s3_download_to_disk(
            coord.get_index_url(), self.dataset_id, region=S3_BUCKET_REGION
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

        prefer_pgrb2 = coord.file_type == "pgrb2b"
        out_loc_base = dict(coord.out_loc())
        location = coord.get_url()
        refs = []
        for (start, element, level, window), end in zip(index_lines, ends, strict=True):
            if prefer_pgrb2 and (element, level) in PGRB2_PREFERRED_MESSAGES:
                continue
            matches = lookup.get((element, level, window))
            if not matches:
                continue
            # Byte ranges past the data file mean a stale/mismatched index; skip it.
            if end > file_size or end <= start:
                log.warning(
                    f"Skipping {location}: index byte ranges fall outside the "
                    f"{file_size}-byte data file; stale or mismatched index"
                )
                return []
            for var, level_label in matches:
                refs.append(
                    VirtualRef(
                        data_var=var,
                        out_loc={**out_loc_base, **level_label},
                        location=location,
                        offset=start,
                        length=end - start,
                    )
                )
        return refs

    def _message_lookup(
        self, data_vars: Sequence[NoaaDataVar], lead_hours: int
    ) -> dict[
        tuple[str, str, str], list[tuple[NoaaDataVar, dict[Dim, CoordinateValue]]]
    ]:
        """Map each (element, idx level string, idx window string) to the variables it
        fills and the vertical label each ref carries. A root var contributes one entry;
        a vertical-group var one per level (its grib_index_level is a "{level:g} mb"
        format string). The mapping is one-to-many: at leads 1-6 the run-total and
        6 hour bucket variants of APCP/ACPCP render the same window string, and both
        must be filled from the single matching message."""
        lookup: dict[
            tuple[str, str, str], list[tuple[NoaaDataVar, dict[Dim, CoordinateValue]]]
        ] = {}
        for var in data_vars:
            window = _lead_time_str(var, lead_hours)
            # CLWMR was respelled CLMR in 2023; a file carries only one spelling.
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
