import icechunk
import pandas as pd
from pydantic import PrivateAttr

from reformatters.common.download import gcs_store, s3_store
from reformatters.common.region_job import InitLeadSourceFileCoord
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,
)
from reformatters.ecmwf.ecmwf_utils import EcmwfOpenDataSource

GCS_LOCATION_PREFIX = "gs://ecmwf-open-data/"
S3_LOCATION_PREFIX = "s3://ecmwf-forecasts/"
S3_REGION = "eu-central-1"


class EcmwfVirtualSourceFileCoord(InitLeadSourceFileCoord):
    # Discovery switches pending coordinates in place; the write loop tracks identity.
    _source: EcmwfOpenDataSource = PrivateAttr(default="gcs")

    @property
    def source(self) -> EcmwfOpenDataSource:
        return self._source

    def fall_back_to_s3(self) -> None:
        self._source = "s3"


def ecmwf_virtual_chunk_containers(
    *, s3_fallback_before: pd.Timestamp | None
) -> tuple[icechunk.VirtualChunkContainer, ...]:
    containers = (
        icechunk.VirtualChunkContainer(GCS_LOCATION_PREFIX, icechunk.gcs_store()),
    )
    if s3_fallback_before is not None:
        containers += (
            icechunk.VirtualChunkContainer(
                S3_LOCATION_PREFIX, icechunk.s3_store(region=S3_REGION)
            ),
        )
    return containers


def discover_ecmwf_available[COORD: EcmwfVirtualSourceFileCoord](
    pending: list[COORD],
    *,
    s3_fallback_before: pd.Timestamp | None,
) -> list[tuple[COORD, int]]:
    assert all(coord.source == "gcs" for coord in pending)
    available = discover_available_by_obstore_listing(
        pending,
        store=gcs_store(GCS_LOCATION_PREFIX),
        location_prefix=GCS_LOCATION_PREFIX,
        require_index=True,
    )
    if s3_fallback_before is None:
        return available
    on_gcs = {id(coord) for coord, _ in available}
    # S3 is listed through copies so a coord GCS lacks and S3 also lacks keeps
    # its GCS URL and stays pending.
    probes = []
    originals: dict[int, COORD] = {}
    for coord in pending:
        if id(coord) in on_gcs or coord.init_time >= s3_fallback_before:
            continue
        probe = coord.model_copy()
        probe.fall_back_to_s3()
        probes.append(probe)
        originals[id(probe)] = coord
    if probes:
        on_s3 = discover_available_by_obstore_listing(
            probes,
            store=s3_store(S3_LOCATION_PREFIX, region=S3_REGION),
            location_prefix=S3_LOCATION_PREFIX,
            require_index=True,
        )
        for probe, size in on_s3:
            coord = originals[id(probe)]
            coord.fall_back_to_s3()
            available.append((coord, size))
    return available
