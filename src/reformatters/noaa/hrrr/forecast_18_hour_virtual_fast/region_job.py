from pathlib import Path
from typing import ClassVar

import icechunk
import obstore
import obstore.store
import pandas as pd
import pydantic

from reformatters.common.download import download_to_disk, get_local_path
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Timedelta
from reformatters.common.virtual_region_job import VirtualRef
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,
)
from reformatters.noaa.hrrr.nomads_mirror import (
    MIRROR_LOCATION_PREFIX,
    mirror_key,
    mirror_store,
)
from reformatters.noaa.hrrr.region_job import DownloadSource
from reformatters.noaa.hrrr.virtual_region_job import (
    NoaaHrrrForecastVirtualRegionJob,
    NoaaHrrrForecastVirtualSourceFileCoord,
)

from .template_config import RETENTION


def hrrr_18_hour_virtual_fast_chunk_containers() -> tuple[
    icechunk.VirtualChunkContainer, ...
]:
    """The NOMADS mirror's public domain."""
    return (
        icechunk.VirtualChunkContainer(MIRROR_LOCATION_PREFIX, icechunk.http_store()),
    )


class NoaaHrrrForecast18HourVirtualFastSourceFileCoord(
    NoaaHrrrForecastVirtualSourceFileCoord
):
    """An HRRR file in the NOMADS mirror, under the key NODD gives it."""

    def get_url(self, source: DownloadSource = "s3") -> str:  # noqa: ARG002
        return MIRROR_LOCATION_PREFIX + mirror_key(self)


class NoaaHrrrForecast18HourVirtualFastRegionJob(NoaaHrrrForecastVirtualRegionJob):
    """Reads only the NOMADS mirror, which expires its files, so the store keeps a
    moving window of recent inits. See "NOMADS mirror" in docs/virtual_datasets.md."""

    operational_update_window: ClassVar[Timedelta] = RETENTION
    drops_before_template_start: ClassVar[bool] = True
    source_file_coord_class: ClassVar[
        type[NoaaHrrrForecast18HourVirtualFastSourceFileCoord]
    ] = NoaaHrrrForecast18HourVirtualFastSourceFileCoord

    # The first sweep of a run offers every missing file in the window; later sweeps
    # poll only for inits this recent, so a file that never arrives costs no listings.
    poll_window: ClassVar[Timedelta] = pd.Timedelta("6h")

    _sweeps: int = pydantic.PrivateAttr(default=0)
    _mirror_store: obstore.store.ObjectStore | None = pydantic.PrivateAttr(default=None)

    def mirror_store(self) -> obstore.store.ObjectStore:
        store = self._mirror_store
        if store is None:
            store = mirror_store()
            self._mirror_store = store  # ty: ignore[invalid-assignment] - private cache
        return store

    def discover_available(
        self, pending: list[NoaaHrrrForecastVirtualSourceFileCoord]
    ) -> list[tuple[NoaaHrrrForecastVirtualSourceFileCoord, int]]:
        self._sweeps += 1
        if self._sweeps > 1:
            newest = self.template_ds.to_dataset().get_index(self.append_dim)[-1]
            pending = [c for c in pending if c.init_time > newest - self.poll_window]
        return discover_available_by_obstore_listing(
            pending,
            store=self.mirror_store(),
            location_prefix=MIRROR_LOCATION_PREFIX,
            require_index=True,
        )

    def download_index(self, coord: NoaaHrrrForecastVirtualSourceFileCoord) -> Path:
        key = mirror_key(coord) + ".idx"
        local_path = get_local_path(self.dataset_id, key)
        download_to_disk(self.mirror_store(), key, local_path)
        return local_path

    def read_data_bytes(
        self, coord: NoaaHrrrForecastVirtualSourceFileCoord, start: int, end: int
    ) -> bytes:
        return bytes(
            obstore.get_range(
                self.mirror_store(), mirror_key(coord), start=start, end=end
            )
        )

    def _check_refs_complete(
        self, coord: NoaaHrrrForecastVirtualSourceFileCoord, refs: list[VirtualRef]
    ) -> None:
        """Skip a file whose index lacks any message the template expects of it: a
        pair mirrored before NOMADS finished writing it would otherwise be published
        partial and then count as ingested."""
        lookup = self._message_lookup(coord.data_vars, whole_hours(coord.lead_time))
        expected = {
            (var.path, tuple(sorted(level.items())))
            for entries in lookup.values()
            for var, level in entries
        }
        filled = {
            (
                ref.data_var.path,
                tuple(
                    sorted(
                        (dim, value)
                        for dim, value in ref.out_loc.items()
                        if dim not in ("init_time", "lead_time")
                    )
                ),
            )
            for ref in refs
        }
        missing = sorted(expected - filled)
        if missing:
            raise ValueError(
                f"{coord.get_url()} lacks {len(missing)} of {len(expected)} expected "
                f"messages, e.g. {missing[:3]}; not ingesting a partial file"
            )
