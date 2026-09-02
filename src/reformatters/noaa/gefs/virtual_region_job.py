from typing import ClassVar, Generic, TypeVar

import icechunk

from reformatters.common.logging import get_logger
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Timestamp
from reformatters.common.virtual_region_job import VirtualRef
from reformatters.noaa.gefs.gefs_config_models import (
    FILE_RESOLUTIONS,
    GEFSSourceFileType,
    NoaaGefsVirtualDataVar,
)
from reformatters.noaa.noaa_virtual_region_job import (
    NoaaVirtualRegionJob,
    NoaaVirtualSourceFileCoord,
)

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


class NoaaGefsVirtualSourceFileCoord(
    NoaaVirtualSourceFileCoord[NoaaGefsVirtualDataVar]
):
    """One GEFS product file (init_time, lead_time, member, file type) and its vars."""

    init_time: Timestamp
    ensemble_member: int
    source_file_type: GEFSSourceFileType

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


GEFS_VIRTUAL_COORD = TypeVar("GEFS_VIRTUAL_COORD", bound=NoaaGefsVirtualSourceFileCoord)


class NoaaGefsVirtualRegionJob(
    NoaaVirtualRegionJob[NoaaGefsVirtualDataVar, GEFS_VIRTUAL_COORD],
    Generic[GEFS_VIRTUAL_COORD],
):
    """Ref building for the GEFS virtual datasets. A subclass adds
    generate_source_file_coords and operational_update_window."""

    source_location_prefix: ClassVar[str] = S3_LOCATION_PREFIX
    source_bucket_region: ClassVar[str] = S3_BUCKET_REGION

    def _check_refs_complete(
        self, coord: GEFS_VIRTUAL_COORD, refs: list[VirtualRef]
    ) -> None:
        """A variable with no matching message would otherwise be committed as a silent
        NaN column: the file counts as ingested through its representative variable, so
        nothing ever retries it."""
        filled = {ref.data_var.path for ref in refs}
        unmatched = sorted(
            var.name for var in coord.data_vars if var.path not in filled
        )
        assert not unmatched, (
            f"{coord.get_url()} has no message for {unmatched}; "
            "the source era is not modelled by this catalog"
        )

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
