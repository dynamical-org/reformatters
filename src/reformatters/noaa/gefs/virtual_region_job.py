from collections.abc import Mapping, Sequence
from typing import ClassVar, Generic, TypeVar

import icechunk
import pandas as pd
import xarray as xr

from reformatters.common.region_job import CoordinateValue
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Dim
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
    """The GEFS source bucket and the checks specific to its archive. A subclass adds
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
        preferring an instant one so the probe lands where data exists at every step."""
        candidates = [
            var for var in coord.data_vars if var.internal_attrs.available_from is None
        ] or list(coord.data_vars)  # a run of only later-era vars must probe one
        return next(
            (var for var in candidates if var.attrs.step_type == "instant"),
            candidates[0],
        )


class NoaaGefsForecastVirtualSourceFileCoord(NoaaGefsVirtualSourceFileCoord):
    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
            "ensemble_member": self.ensemble_member,
        }


class NoaaGefsForecastVirtualRegionJob(
    NoaaGefsVirtualRegionJob[NoaaGefsForecastVirtualSourceFileCoord]
):
    """RegionJob shared by the GEFS virtual forecast datasets."""

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaGefsVirtualDataVar],
    ) -> Sequence[NoaaGefsForecastVirtualSourceFileCoord]:
        """One coord per (init time, ensemble member, lead time, source file), holding
        the variables that file carries.

        A variable contributes no coord where the source has no message for it -- before
        its `available_from` cycle, or at lead 0 for a windowed quantity, whose lead 0
        window is zero length -- so the store holds no ref there and readers get NaN.
        """
        init_times = pd.to_datetime(processing_region_ds["init_time"].values)
        lead_times = pd.to_timedelta(processing_region_ds["lead_time"].values)
        ensemble_members = [
            int(member) for member in processing_region_ds["ensemble_member"].values
        ]

        coords = []
        for init_time in init_times:
            available_vars = [
                var
                for var in data_var_group
                if (available_from := var.internal_attrs.available_from) is None
                or init_time >= available_from
            ]
            for lead_time in lead_times:
                grouped: dict[GEFSSourceFileType, list[NoaaGefsVirtualDataVar]] = {}
                for var in available_vars:
                    if lead_time == pd.Timedelta(0) and not var.has_hour_0_values():
                        continue
                    grouped.setdefault(var.internal_attrs.source_file_type, []).append(
                        var
                    )
                for source_file_type, data_vars in grouped.items():
                    coords.extend(
                        NoaaGefsForecastVirtualSourceFileCoord(
                            init_time=init_time,
                            lead_time=lead_time,
                            ensemble_member=ensemble_member,
                            source_file_type=source_file_type,
                            data_vars=data_vars,
                        )
                        for ensemble_member in ensemble_members
                    )
        return coords
