from typing import ClassVar, Generic, TypeVar

import icechunk

from reformatters.common.config_models import ROOT
from reformatters.noaa.gfs.region_job import (
    DownloadSource,
    NoaaGfsFileType,
    NoaaGfsSourceFileCoord,
)
from reformatters.noaa.models import NoaaDataVar
from reformatters.noaa.noaa_virtual_region_job import (
    NoaaVirtualRegionJob,
    NoaaVirtualSourceFileCoord,
)

S3_LOCATION_PREFIX = "s3://noaa-gfs-bdp-pds/"
S3_BUCKET_REGION = "us-east-1"
_S3_HTTPS_PREFIX = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"

# Every message of both products is curated, so a step is complete only once both
# files are read.
GFS_FILE_TYPES: tuple[NoaaGfsFileType, ...] = ("pgrb2", "pgrb2b")

# 41 messages that pgrb2b repeats byte for byte from pgrb2. Building refs from both
# would write two byte ranges for one array position, so a pgrb2b index skips these and
# the array is filled from pgrb2. All 41 are instantaneous and uniquely identified by
# (element, idx level string) at every lead.
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


# The idx level strings, and the two surface elements, that only pgrb2b publishes. Every
# other root variable comes from pgrb2. Isobaric variables are carried by both products
# except SPFH and O3MR, whose pgrb2b copies are all in PGRB2_PREFERRED_MESSAGES.
_PGRB2B_ONLY_ROOT_LEVELS: frozenset[str] = frozenset(
    [
        *(
            f"PV={sign}{value}e-0{exponent} (Km^2/kg/s) surface"
            for sign in ("", "-")
            for value, exponent in (("5", "7"), ("1", "6"), ("1.5", "6"))
        ),
        *(
            f"{top}-{bottom} mb above ground"
            for top, bottom in ((60, 30), (90, 60), (120, 90), (150, 120), (180, 150))
        ),
        *(f"{height} m above mean sea level" for height in (305, 457, 610, 914, 4572)),
    ]
)
_PGRB2B_ONLY_SURFACE_ELEMENTS: frozenset[str] = frozenset({"DUVB", "CDUVB"})
_PGRB2_ONLY_ISOBARIC_ELEMENTS: frozenset[str] = frozenset({"SPFH", "O3MR"})


def carried_by(var: NoaaDataVar, file_type: NoaaGfsFileType) -> bool:
    """Whether `file_type` publishes any message this variable is built from.

    A file is read only for the variables it carries, and the chunk whose presence marks
    it ingested has to be one of them, so a job filtered to a subset of the catalog needs
    this rather than offering every variable to both products.
    """
    if var.group is not ROOT:
        return (
            file_type == "pgrb2"
            or var.internal_attrs.grib_element not in _PGRB2_ONLY_ISOBARIC_ELEMENTS
        )
    pgrb2b_only = (
        var.internal_attrs.grib_index_level in _PGRB2B_ONLY_ROOT_LEVELS
        or var.internal_attrs.grib_element in _PGRB2B_ONLY_SURFACE_ELEMENTS
    )
    return pgrb2b_only == (file_type == "pgrb2b")


def gfs_virtual_chunk_containers() -> tuple[icechunk.VirtualChunkContainer, ...]:
    """Fresh container objects per call; icechunk containers can't be shared
    pydantic defaults."""
    return (
        icechunk.VirtualChunkContainer(
            S3_LOCATION_PREFIX, icechunk.s3_store(region=S3_BUCKET_REGION)
        ),
    )


class NoaaGfsVirtualSourceFileCoord(
    NoaaGfsSourceFileCoord, NoaaVirtualSourceFileCoord[NoaaDataVar]
):
    """One GFS product file (init_time, lead_time, file_type) and the vars it may fill."""

    def get_url(self, source: DownloadSource = "s3") -> str:
        """The s3:// source location refs point at, matching the virtual chunk container."""
        url = super().get_url(source=source)
        assert url.startswith(_S3_HTTPS_PREFIX), url
        return S3_LOCATION_PREFIX + url.removeprefix(_S3_HTTPS_PREFIX)


GFS_VIRTUAL_COORD = TypeVar("GFS_VIRTUAL_COORD", bound=NoaaGfsVirtualSourceFileCoord)


class NoaaGfsVirtualRegionJob(
    NoaaVirtualRegionJob[NoaaDataVar, GFS_VIRTUAL_COORD],
    Generic[GFS_VIRTUAL_COORD],
):
    """The GFS NODD bucket, where pgrb2b repeats 41 messages pgrb2 owns. A subclass
    adds generate_source_file_coords and operational_update_window."""

    source_location_prefix: ClassVar[str] = S3_LOCATION_PREFIX
    source_bucket_region: ClassVar[str] = S3_BUCKET_REGION

    def owns_index_message(
        self, coord: GFS_VIRTUAL_COORD, element: str, level: str
    ) -> bool:
        if coord.file_type == "pgrb2":
            return True
        return (element, level) not in PGRB2_PREFERRED_MESSAGES
