from collections.abc import Mapping, Sequence
from typing import ClassVar

import pandas as pd
import xarray as xr

from reformatters.common.download import s3_read_bytes, s3_store
from reformatters.common.region_job import CoordinateValue, SourceFileCoord
from reformatters.common.types import Dim, Timedelta, Timestamp
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,
)
from reformatters.noaa.rtma.grib_header import scan_analysis_messages
from reformatters.noaa.rtma.rtma_config_models import RtmaDataVar

SOURCE_PREFIX = "s3://noaa-rtma-pds/"
SOURCE_REGION = "us-east-1"


class NoaaRtmaAnalysisVirtualSourceFileCoord(SourceFileCoord):
    time: Timestamp
    data_vars: Sequence[RtmaDataVar]

    def get_url(self) -> str:
        return f"{SOURCE_PREFIX}rtma2p5.{self.time:%Y%m%d}/rtma2p5.t{self.time:%H}z.2dvaranl_ndfd.grb2_wexp"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"time": self.time}


class NoaaRtmaAnalysisVirtualRegionJob(
    VirtualRegionJob[RtmaDataVar, NoaaRtmaAnalysisVirtualSourceFileCoord]
):
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("24h")
    source_prefix: ClassVar[str] = SOURCE_PREFIX
    source_file_coord_class: ClassVar[type[NoaaRtmaAnalysisVirtualSourceFileCoord]] = (
        NoaaRtmaAnalysisVirtualSourceFileCoord
    )

    def generate_source_file_coords(
        self, processing_region_ds: xr.Dataset, data_var_group: Sequence[RtmaDataVar]
    ) -> Sequence[NoaaRtmaAnalysisVirtualSourceFileCoord]:
        coords = []
        for time in pd.to_datetime(processing_region_ds["time"].values):
            available = [
                var
                for var in data_var_group
                if var.internal_attrs.available_from is None
                or time >= var.internal_attrs.available_from
            ]
            if available:
                coords.append(
                    self.source_file_coord_class(time=time, data_vars=available)
                )
        return coords

    def discover_available(
        self, pending: list[NoaaRtmaAnalysisVirtualSourceFileCoord]
    ) -> list[tuple[NoaaRtmaAnalysisVirtualSourceFileCoord, int]]:
        return discover_available_by_obstore_listing(
            pending,
            store=s3_store(self.source_prefix, region=SOURCE_REGION),
            location_prefix=self.source_prefix,
            require_index=False,
        )

    def file_refs(
        self, coord: NoaaRtmaAnalysisVirtualSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        variables = {var.internal_attrs.grib_identity: var for var in coord.data_vars}
        assert len(variables) == len(coord.data_vars), (
            "Duplicate selected GRIB identities"
        )

        def read_range(start: int, end: int) -> bytes:
            return s3_read_bytes(
                coord.get_url(), region=SOURCE_REGION, start=start, end=end
            )

        messages = scan_analysis_messages(
            read_range,
            file_size,
            coord.time,
            set(variables),
            bitmapped=frozenset(
                identity
                for identity, var in variables.items()
                if var.internal_attrs.requires_bitmap
            ),
        )
        refs = [
            VirtualRef(
                data_var=variables[message.identity],
                out_loc=coord.out_loc(),
                location=coord.get_url(),
                offset=message.offset,
                length=message.length,
            )
            for message in messages
        ]
        # Partial coverage would let the representative chunk suppress retries
        # for missing fields in the same source file.
        missing = {var.path for var in coord.data_vars} - {
            ref.data_var.path for ref in refs
        }
        assert not missing, (
            f"Missing selected fields in {coord.get_url()}: {sorted(missing)}"
        )
        return refs
