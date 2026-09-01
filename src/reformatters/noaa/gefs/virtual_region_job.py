from collections.abc import Sequence
from typing import Generic, TypeVar

import icechunk

from reformatters.common.download import s3_download_to_disk, s3_store
from reformatters.common.logging import get_logger
from reformatters.common.region_job import SourceFileCoord
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Timedelta, Timestamp
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,
)
from reformatters.noaa.gefs.gefs_config_models import (
    FILE_RESOLUTIONS,
    GEFSSourceFileType,
    NoaaGefsVirtualDataVar,
)
from reformatters.noaa.noaa_grib_index import _lead_time_str, parse_grib_index_lines

log = get_logger(__name__)

S3_LOCATION_PREFIX = "s3://noaa-gefs-pds/"
S3_BUCKET_REGION = "us-east-1"


def gefs_virtual_chunk_containers() -> tuple[icechunk.VirtualChunkContainer, ...]:
    """Fresh container objects per call; icechunk containers can't be shared
    pydantic defaults."""
    return (
        icechunk.VirtualChunkContainer(
            S3_LOCATION_PREFIX, icechunk.s3_store(region=S3_BUCKET_REGION)
        ),
    )


class NoaaGefsVirtualSourceFileCoord(SourceFileCoord):
    """One GEFS product file (init_time, lead_time, member, file type) and its vars."""

    init_time: Timestamp
    lead_time: Timedelta
    ensemble_member: int
    source_file_type: GEFSSourceFileType
    data_vars: Sequence[NoaaGefsVirtualDataVar]

    def get_url(self) -> str:
        """The s3:// location refs point at, matching the virtual chunk container."""
        member = f"{'c' if self.ensemble_member == 0 else 'p'}{self.ensemble_member:02}"
        file_type = self.source_file_type
        resolution = FILE_RESOLUTIONS[file_type]
        return (
            f"{S3_LOCATION_PREFIX}"
            f"gefs.{self.init_time:%Y%m%d}/{self.init_time:%H}/atmos/"
            f"pgrb2{file_type}{resolution.strip('0')}/"
            f"ge{member}.t{self.init_time:%H}z.pgrb2{file_type}.{resolution}"
            f".f{whole_hours(self.lead_time):03d}"
        )

    def get_index_url(self) -> str:
        return self.get_url() + ".idx"


GEFS_VIRTUAL_COORD = TypeVar("GEFS_VIRTUAL_COORD", bound=NoaaGefsVirtualSourceFileCoord)


class NoaaGefsVirtualRegionJob(
    VirtualRegionJob[NoaaGefsVirtualDataVar, GEFS_VIRTUAL_COORD],
    Generic[GEFS_VIRTUAL_COORD],
):
    """Source-file discovery on NODD S3 and ref building from GRIB indexes, shared by
    the GEFS virtual datasets. A subclass adds generate_source_file_coords and
    operational_update_window."""

    def discover_available(
        self, pending: list[GEFS_VIRTUAL_COORD]
    ) -> list[tuple[GEFS_VIRTUAL_COORD, int]]:
        return discover_available_by_obstore_listing(
            pending,
            store=s3_store(S3_LOCATION_PREFIX, region=S3_BUCKET_REGION),
            location_prefix=S3_LOCATION_PREFIX,
            require_index=True,
        )

    def file_refs(self, coord: GEFS_VIRTUAL_COORD, file_size: int) -> list[VirtualRef]:
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

        out_loc = dict(coord.out_loc())
        location = coord.get_url()
        refs = []
        for (start, element, level, window), end in zip(index_lines, ends, strict=True):
            matches = lookup.pop((element, level, window), None)
            if not matches:
                continue
            # Byte ranges past the data file mean a stale/mismatched index; skip it.
            if end > file_size or end <= start:
                log.warning(
                    f"Skipping {location}: index byte ranges fall outside the "
                    f"{file_size}-byte data file; stale or mismatched index"
                )
                return []
            refs.extend(
                VirtualRef(
                    data_var=var,
                    out_loc=out_loc,
                    location=location,
                    offset=start,
                    length=end - start,
                )
                for var in matches
            )

        # A variable with no matching message would otherwise be committed as a silent
        # NaN column: the file counts as ingested through its representative variable,
        # so nothing ever retries it.
        assert not lookup, (
            f"{location} has no message for {sorted({v.name for m in lookup.values() for v in m})}; "
            "the source era is not modelled by this catalog"
        )
        return refs

    def _message_lookup(
        self, data_vars: Sequence[NoaaGefsVirtualDataVar], lead_hours: int
    ) -> dict[tuple[str, str, str], list[NoaaGefsVirtualDataVar]]:
        """Map each (element, idx level string, idx window string) to the variables it
        fills. Two variables share a key where one message serves both."""
        lookup: dict[tuple[str, str, str], list[NoaaGefsVirtualDataVar]] = {}
        for var in data_vars:
            window = _lead_time_str(var, lead_hours)
            # Index element spellings vary by era and by the wgrib2 build that made the
            # index, so match grib_element_alternatives too; a file carries only one.
            for element in (
                var.internal_attrs.grib_element,
                *var.internal_attrs.grib_element_alternatives,
            ):
                key = (element, var.internal_attrs.grib_index_level, window)
                lookup.setdefault(key, []).append(var)
        return lookup

    def representative_var(self, coord: GEFS_VIRTUAL_COORD) -> NoaaGefsVirtualDataVar:
        """Probe file presence through a variable the archive published in every era,
        preferring an instant one so the probe lands where data exists at every step.

        Probing a variable the source added partway through would leave every older file
        permanently un-ingestable. A run filtered to only such variables has no other
        choice, and those files do carry them, so it falls back rather than refusing.
        """
        candidates = [
            var for var in coord.data_vars if var.internal_attrs.available_from is None
        ] or list(coord.data_vars)
        return next(
            (var for var in candidates if var.attrs.step_type == "instant"),
            candidates[0],
        )
