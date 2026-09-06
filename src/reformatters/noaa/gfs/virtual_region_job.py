from collections.abc import Mapping
from typing import Any, ClassVar, Generic, TypeVar

import icechunk

from reformatters.common.config_models import ROOT, DataVar
from reformatters.common.region_job import CoordinateValue
from reformatters.common.types import Dim
from reformatters.noaa.gfs.region_job import (
    NODD_BUCKET,
    NODD_BUCKET_REGION,
    NODD_HTTPS_PREFIX,
    DownloadSource,
    NoaaGfsFileType,
    NoaaGfsSourceFileCoord,
)
from reformatters.noaa.models import NoaaDataVar
from reformatters.noaa.noaa_virtual_region_job import (
    NoaaVirtualRegionJob,
    NoaaVirtualSourceFileCoord,
)

# Both products are curated in full, so a step's variables span both files.
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
    ]
)
_PGRB2B_ONLY_SURFACE_ELEMENTS: frozenset[str] = frozenset({"DUVB", "CDUVB"})
_PGRB2_ONLY_ISOBARIC_ELEMENTS: frozenset[str] = frozenset({"SPFH", "O3MR"})


def carried_by(var: NoaaDataVar, file_type: NoaaGfsFileType) -> bool:
    """Whether `file_type` publishes any message this variable is built from."""
    if var.group == "height_above_mean_sea_level":
        # Both products publish this family, at disjoint heights.
        return True
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
            f"s3://{NODD_BUCKET}/", icechunk.s3_store(region=NODD_BUCKET_REGION)
        ),
    )


class NoaaGfsVirtualSourceFileCoord(
    NoaaGfsSourceFileCoord, NoaaVirtualSourceFileCoord[NoaaDataVar]
):
    """One GFS product file (init_time, lead_time, file_type) and the vars it may fill."""

    def get_url(self, source: DownloadSource = "s3") -> str:
        """The s3:// source location refs point at, matching the virtual chunk container."""
        url = super().get_url(source=source)
        assert url.startswith(NODD_HTTPS_PREFIX), url
        return f"s3://{NODD_BUCKET}/" + url.removeprefix(NODD_HTTPS_PREFIX)


# The variables a source file's ingestion is probed by, in preference order. Each is
# published in every era of the archive and carried only by its own product. A coord
# holding only the variables without hour 0 values falls through to the second entry.
_REPRESENTATIVE_VARS: dict[NoaaGfsFileType, tuple[str, ...]] = {
    "pgrb2": ("temperature_2m", "total_precipitation_surface"),
    "pgrb2b": ("geopotential_height_0p5pvu", "uv_b_downward_solar_flux_surface"),
}

# A vertical level each product publishes for every variable of the group it carries.
# The products split every vertical coordinate, so a group's first level is a chunk the
# other product never fills and probing it would re-ingest that file forever. One
# constant per (group, product) holds because each product's level inventory is uniform
# across the group's elements.
_PROBE_VERTICAL_LEVEL: dict[Dim, dict[NoaaGfsFileType, float]] = {
    "pressure_level": {"pgrb2": 1000.0, "pgrb2b": 875.0},
    "height_above_mean_sea_level": {"pgrb2": 1829.0, "pgrb2b": 305.0},
}


GFS_VIRTUAL_COORD = TypeVar("GFS_VIRTUAL_COORD", bound=NoaaGfsVirtualSourceFileCoord)


class NoaaGfsVirtualRegionJob(
    NoaaVirtualRegionJob[NoaaDataVar, GFS_VIRTUAL_COORD],
    Generic[GFS_VIRTUAL_COORD],
):
    """The GFS NODD bucket, where pgrb2b repeats 41 messages pgrb2 owns. A subclass
    adds generate_source_file_coords and operational_update_window."""

    source_location_prefix: ClassVar[str] = f"s3://{NODD_BUCKET}/"
    source_bucket_region: ClassVar[str] = NODD_BUCKET_REGION

    def owns_index_message(
        self, coord: GFS_VIRTUAL_COORD, element: str, level: str
    ) -> bool:
        if coord.file_type == "pgrb2":
            return True
        return (element, level) not in PGRB2_PREFERRED_MESSAGES

    def representative_var(self, coord: GFS_VIRTUAL_COORD) -> NoaaDataVar:
        """A variable this file fills, preferring one whose chunk needs no level pick."""
        by_name = {var.name: var for var in coord.data_vars}
        candidates = [
            *(
                by_name[name]
                for name in _REPRESENTATIVE_VARS[coord.file_type]
                if name in by_name
            ),
            *(var for var in coord.data_vars if var.group is ROOT),
        ]
        return next(iter(candidates), super().representative_var(coord))

    def representative_probe_loc(
        self, coord: GFS_VIRTUAL_COORD, var: DataVar[Any]
    ) -> Mapping[Dim, CoordinateValue]:
        loc = dict(super().representative_probe_loc(coord, var))
        for dim, by_product in _PROBE_VERTICAL_LEVEL.items():
            if dim in loc:
                loc[dim] = by_product[coord.file_type]
        return loc
