import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar, Literal

import obstore
import pandas as pd
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.download import s3_download_to_disk, s3_store
from reformatters.common.logging import get_logger
from reformatters.common.region_job import RegionJob
from reformatters.common.types import AppendDim, DatetimeLike, Timedelta
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_B22_TRANSITION_DATE,
    GEFS_CURRENT_ARCHIVE_START,
    GEFS_S_FILE_MAX,
    GEFSDataVar,
    GefsEnsembleSourceFileCoord,
)
from reformatters.noaa.noaa_grib_index import (
    GRIB_INDEX_UNKNOWN_END_PAD,
    grib_message_byte_ranges_from_index,
)
from reformatters.noaa.noaa_utils import has_hour_0_values

log = get_logger(__name__)

_S3_LOCATION_PREFIX = "s3://noaa-gefs-pds/"
_S3_BUCKET_REGION = "us-east-1"


def _vars_in_s_file(
    data_vars: Sequence[GEFSDataVar], init_time: pd.Timestamp, lead_time: pd.Timedelta
) -> list[GEFSDataVar]:
    """The subset of data_vars the s file for (init_time, lead_time) contains.

    "s+b-b22" variables were only added to the s files at GEFS_B22_TRANSITION_DATE,
    and accumulated/avg variables don't exist in the 0-hour forecast.
    """
    return [
        var
        for var in data_vars
        if (
            var.internal_attrs.gefs_file_type != "s+b-b22"
            or init_time >= GEFS_B22_TRANSITION_DATE
        )
        and (lead_time != pd.Timedelta(0) or has_hour_0_values(var))
    ]


class GefsForecast10DaySpatialSourceFileCoord(GefsEnsembleSourceFileCoord):
    """One s file: (init_time, ensemble_member, lead_time) and the vars it packs."""

    @property
    def gefs_file_type(self) -> Literal["s"]:
        # Virtual chunks decode the raw GRIB message, so every ref must share the
        # 0.25 degree s file grid; this dataset's leads stop where the s files do.
        assert self.init_time >= GEFS_CURRENT_ARCHIVE_START, (
            f"Only the current GEFS archive is supported, got {self.init_time}"
        )
        assert self.lead_time <= GEFS_S_FILE_MAX, (
            f"s files end at {GEFS_S_FILE_MAX}, got {self.lead_time}"
        )
        assert self.data_vars == _vars_in_s_file(
            self.data_vars, self.init_time, self.lead_time
        ), f"coord includes vars not in the s file for {self.init_time}"
        return "s"

    def get_url(self) -> str:
        """The s3:// source location; refs point here, matching the dataset's
        virtual chunk container prefix."""
        url = super().get_url()
        https_prefix = f"https://{self.primary_base_url}/"
        assert url.startswith(https_prefix), url
        return _S3_LOCATION_PREFIX + url.removeprefix(https_prefix)


class GefsForecast10DaySpatialRegionJob(
    VirtualRegionJob[GEFSDataVar, GefsForecast10DaySpatialSourceFileCoord]
):
    """RegionJob for the GEFS 10-day spatial (virtual) forecast dataset."""

    # Reprocess this span of recent init times each operational update. Each fire
    # re-sweeps the window, catching stragglers earlier fires gave up on.
    operational_window: ClassVar[Timedelta] = pd.Timedelta("24h")
    # When polling, pace bucket listings to at most one sweep per tick.
    tick_interval: ClassVar[Timedelta] = pd.Timedelta("1s")
    # Concurrent index file downloads. The .idx files are ~30KB latency-bound
    # requests; measured throughput vs pool width: 8 -> ~105 files/s,
    # 32 -> ~310, 64 -> ~380, 128 -> ~435 (diminishing).
    download_concurrency: ClassVar[int] = 64

    def generate_source_file_coords(
        self, processing_region_ds: xr.Dataset, data_var_group: Sequence[GEFSDataVar]
    ) -> Sequence[GefsForecast10DaySpatialSourceFileCoord]:
        coords = []
        for init_time in processing_region_ds["init_time"].values:
            for ensemble_member in processing_region_ds["ensemble_member"].values:
                for lead_time in processing_region_ds["lead_time"].values:
                    vars_in_file = _vars_in_s_file(
                        data_var_group, pd.Timestamp(init_time), pd.Timedelta(lead_time)
                    )
                    if not vars_in_file:
                        continue
                    coords.append(
                        GefsForecast10DaySpatialSourceFileCoord(
                            init_time=pd.Timestamp(init_time),
                            ensemble_member=int(ensemble_member),
                            lead_time=pd.Timedelta(lead_time),
                            data_vars=vars_in_file,
                        )
                    )
        return coords

    def process_virtual_refs(
        self, remaining: Sequence[GefsForecast10DaySpatialSourceFileCoord]
    ) -> Iterator[
        Sequence[tuple[GefsForecast10DaySpatialSourceFileCoord, Sequence[VirtualRef]]]
    ]:
        """Each tick: list the bucket, fetch newly available files' indexes, yield their refs.

        One yield (= one commit) per tick, containing every file that became
        available since the last tick.
        """
        pending = {
            coord.get_url().removeprefix(_S3_LOCATION_PREFIX): coord
            for coord in remaining
        }
        with ThreadPoolExecutor(self.download_concurrency) as pool:
            while pending:
                tick_start = time.monotonic()
                available = self._discover_available(pending)
                if available:
                    coords = [pending[key] for key in available]
                    refs_per_file = pool.map(
                        self._file_refs_or_skip, coords, available.values()
                    )
                    # Files with no refs were skipped (unreadable source); drop them
                    # so a batch never carries a file with no chunks.
                    batch = [
                        (coord, refs)
                        for coord, refs in zip(coords, refs_per_file, strict=True)
                        if refs
                    ]
                    for key in available:
                        del pending[key]
                    skipped = len(coords) - len(batch)
                    log.info(
                        f"Ingesting {len(batch)} files"
                        f"{f' ({skipped} skipped)' if skipped else ''}, "
                        f"{len(pending)} still pending"
                    )
                    if batch:
                        yield batch
                if self.processing_mode == "backfill":
                    if pending:
                        log.info(
                            f"{len(pending)} source files not present, skipping (first: {next(iter(pending))})"
                        )
                    return
                if pending:
                    elapsed = time.monotonic() - tick_start
                    time.sleep(max(0.0, self.tick_interval.total_seconds() - elapsed))

    def _discover_available(
        self, pending: Mapping[str, GefsForecast10DaySpatialSourceFileCoord]
    ) -> dict[str, int]:
        """The pending files retrievable now, mapped to their data file's size in bytes.

        A file is available once the bucket lists both its data and index objects
        (the .idx lands a few seconds after the .grib2, and refs need both).
        """
        listed: dict[str, int] = {}
        for prefix in sorted({key.rsplit("/", 1)[0] + "/" for key in pending}):
            listed |= _list_objects(prefix)
        return {
            key: listed[key]
            for key in pending
            if key in listed and f"{key}.idx" in listed
        }

    def _file_refs_or_skip(
        self, coord: GefsForecast10DaySpatialSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        # Skip a file we can't turn into refs (a decode surprise, a transient read
        # error) rather than sink the whole job; its chunks stay fill and validation
        # surfaces the gap. AssertionError is our own invariant, so it propagates.
        # PR 7 lifts this onto VirtualRegionJob; see docs/plans/virtual_icechunk_datasets.md.
        try:
            return self._file_refs(coord, file_size)
        except AssertionError:
            raise
        except Exception:
            log.exception(f"Skipping {coord.get_url()}: could not build virtual refs")
            return []

    def _file_refs(
        self, coord: GefsForecast10DaySpatialSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        index_path = s3_download_to_disk(
            coord.get_index_url(), self.dataset_id, region=_S3_BUCKET_REGION
        )
        try:
            starts, ends = grib_message_byte_ranges_from_index(
                index_path, coord.data_vars, coord.init_time, coord.lead_time
            )
        finally:
            # Index files accumulate by the millions in backfills; never keep them.
            index_path.unlink()

        # The last message in the file has no end byte in the index; the listed
        # file size supplies it.
        ends = [
            file_size if end - start >= GRIB_INDEX_UNKNOWN_END_PAD else end
            for start, end in zip(starts, ends, strict=True)
        ]
        # A matching index never references bytes past the data file. When it does, the
        # source was re-published without regenerating the index and the offsets are
        # unusable, so skip the whole file (its chunks stay fill). Returning no refs is
        # the caller's signal to drop the file from the batch.
        if any(
            end > file_size or end <= start
            for start, end in zip(starts, ends, strict=True)
        ):
            log.warning(
                f"Skipping {coord.get_url()}: index byte ranges fall outside the "
                f"{file_size}-byte data file; stale or mismatched index"
            )
            return []

        out_loc = coord.out_loc()
        location = coord.get_url()
        return [
            VirtualRef(
                data_var=var,
                out_loc=out_loc,
                location=location,
                offset=start,
                length=end - start,
            )
            for var, start, end in zip(coord.data_vars, starts, ends, strict=True)
        ]

    def representative_var(
        self, coord: GefsForecast10DaySpatialSourceFileCoord
    ) -> GEFSDataVar:
        # A coord's file contains only its own vars (lead-0 and pre-b22 coords
        # carry a subset), so probe one of those rather than the base class's
        # default drawn from all of self.data_vars.
        return next(
            (v for v in coord.data_vars if v.attrs.step_type == "instant"),
            coord.data_vars[0],
        )

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,  # noqa: ARG003 - the icechunk manifest, not a coordinate, tracks ingested data
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[GEFSDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[RegionJob[GEFSDataVar, GefsForecast10DaySpatialSourceFileCoord]],
        xr.Dataset,
    ]:
        """Return a single polling job spanning the active window of recent init times.

        The cron fires just before an init's publication window opens; the job
        polls until all expected files are ingested, with the pod deadline
        bounding how long it waits on a file that never publishes.
        filter_already_present derives the remaining work from the icechunk
        manifest. See "Operational updates" in docs/virtual_datasets.md.
        """
        append_dim_end = pd.Timestamp.now()
        template_ds = get_template_fn(append_dim_end)
        init_times = template_ds.get_index(append_dim)
        window_start = int(
            init_times.searchsorted(append_dim_end - cls.operational_window)
        )
        job = cls(
            tmp_store=tmp_store,
            template_ds=template_ds,
            data_vars=all_data_vars,
            append_dim=append_dim,
            region=slice(window_start, len(init_times)),
            reformat_job_name=reformat_job_name,
            processing_mode="update",
        )
        return [job], template_ds


def _list_objects(prefix: str) -> dict[str, int]:
    """All object keys under `prefix` in the GEFS bucket, mapped to size in bytes."""
    store = s3_store(_S3_LOCATION_PREFIX, region=_S3_BUCKET_REGION)
    return {
        meta["path"]: meta["size"]
        for batch in obstore.list(store, prefix=prefix, chunk_size=10_000)
        for meta in batch
    }
