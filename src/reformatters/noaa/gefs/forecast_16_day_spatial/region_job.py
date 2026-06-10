from collections.abc import Callable, Iterator, Sequence
from itertools import batched
from pathlib import Path
from typing import ClassVar, Literal

import httpx
import pandas as pd
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.download import http_download_to_disk
from reformatters.common.iterating import item
from reformatters.common.logging import get_logger
from reformatters.common.region_job import RegionJob
from reformatters.common.types import AppendDim, DatetimeLike, Timedelta
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_CURRENT_ARCHIVE_START,
    GEFSDataVar,
    GefsEnsembleSourceFileCoord,
    GEFSFileType,
)
from reformatters.noaa.noaa_grib_index import (
    GRIB_INDEX_UNKNOWN_END_PAD,
    grib_message_byte_ranges_from_index,
)
from reformatters.noaa.noaa_utils import has_hour_0_values

log = get_logger(__name__)

# Virtual chunks decode the raw GRIB message, so every lead reads the native
# 0.5 degree a/b files; the 0.25 degree "s" files end at 240h and the
# materialized pipeline's upsampling can't apply to a byte-range reference.
_AB_FILE_TYPE: dict[GEFSFileType, Literal["a", "b"]] = {
    "a": "a",
    "s+a": "a",
    "b": "b",
    "s+b": "b",
    "s+b-b22": "b",
}

_S3_LOCATION_PREFIX = "s3://noaa-gefs-pds/"


class GefsForecast16DaySpatialSourceFileCoord(GefsEnsembleSourceFileCoord):
    """One a or b file: (init_time, ensemble_member, lead_time) and the vars it packs."""

    @property
    def gefs_file_type(self) -> Literal["a", "b"]:
        assert self.init_time >= GEFS_CURRENT_ARCHIVE_START, (
            f"Only the current GEFS archive is supported, got {self.init_time}"
        )
        return item(
            {_AB_FILE_TYPE[v.internal_attrs.gefs_file_type] for v in self.data_vars}
        )

    def get_s3_location(self) -> str:
        """The ref location, matching the dataset's virtual chunk container prefix."""
        url = self.get_url()
        https_prefix = f"https://{self.primary_base_url}/"
        assert url.startswith(https_prefix), url
        return _S3_LOCATION_PREFIX + url.removeprefix(https_prefix)


class GefsForecast16DaySpatialRegionJob(
    VirtualRegionJob[GEFSDataVar, GefsForecast16DaySpatialSourceFileCoord]
):
    """RegionJob for the GEFS 16-day spatial (virtual) forecast dataset."""

    # Reprocess this span of recent init times each operational update; an init's
    # files finish publishing ~7h after init time.
    operational_window: ClassVar[Timedelta] = pd.Timedelta("24h")
    # Whole files per yield (= per commit); amortizes commit overhead in backfills.
    files_per_yield: ClassVar[int] = 16

    def generate_source_file_coords(
        self, processing_region_ds: xr.Dataset, data_var_group: Sequence[GEFSDataVar]
    ) -> Sequence[GefsForecast16DaySpatialSourceFileCoord]:
        file_type_vars: dict[Literal["a", "b"], list[GEFSDataVar]] = {}
        for var in data_var_group:
            file_type_vars.setdefault(
                _AB_FILE_TYPE[var.internal_attrs.gefs_file_type], []
            ).append(var)

        coords = []
        for init_time in processing_region_ds["init_time"].values:
            for ensemble_member in processing_region_ds["ensemble_member"].values:
                for lead_time in processing_region_ds["lead_time"].values:
                    for file_vars in file_type_vars.values():
                        # Accumulated/avg values don't exist in the 0-hour forecast
                        vars_in_file = (
                            [v for v in file_vars if has_hour_0_values(v)]
                            if pd.Timedelta(lead_time) == pd.Timedelta(0)
                            else file_vars
                        )
                        if not vars_in_file:
                            continue
                        coords.append(
                            GefsForecast16DaySpatialSourceFileCoord(
                                init_time=pd.Timestamp(init_time),
                                ensemble_member=int(ensemble_member),
                                lead_time=pd.Timedelta(lead_time),
                                data_vars=vars_in_file,
                            )
                        )
        return coords

    def process_virtual_refs(
        self, remaining: Sequence[GefsForecast16DaySpatialSourceFileCoord]
    ) -> Iterator[Sequence[VirtualRef]]:
        for batch in batched(remaining, self.files_per_yield, strict=False):
            refs: list[VirtualRef] = []
            available = 0
            for coord in batch:
                try:
                    index_path = http_download_to_disk(
                        coord.get_index_url(), self.dataset_id
                    )
                except FileNotFoundError:
                    # Not yet published; a later operational fire will pick it up.
                    continue
                refs.extend(self._file_refs(coord, index_path))
                available += 1
            log.info(f"{available}/{len(batch)} files available in batch")
            if refs:
                yield refs

    def _file_refs(
        self, coord: GefsForecast16DaySpatialSourceFileCoord, index_path: Path
    ) -> list[VirtualRef]:
        location = coord.get_s3_location()
        out_loc = coord.out_loc()
        refs = []
        for var in coord.data_vars:
            starts, ends = grib_message_byte_ranges_from_index(
                index_path, [var], coord.init_time, coord.lead_time
            )
            start, end = item(zip(starts, ends, strict=True))
            if end - start >= GRIB_INDEX_UNKNOWN_END_PAD:
                # Last message in the file; the index doesn't know its end byte.
                end = _content_length(coord.get_url())
            refs.append(
                VirtualRef(
                    data_var=var,
                    out_loc=out_loc,
                    location=location,
                    offset=start,
                    length=end - start,
                )
            )
        return refs

    def representative_var(
        self, coord: GefsForecast16DaySpatialSourceFileCoord
    ) -> GEFSDataVar:
        # A coord's file contains only its own vars (a and b files are separate
        # coords), so probe one of those rather than the base class's default
        # drawn from all of self.data_vars.
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
        Sequence[RegionJob[GEFSDataVar, GefsForecast16DaySpatialSourceFileCoord]],
        xr.Dataset,
    ]:
        """Return a single job spanning the active window of recent init times.

        filter_already_present derives the remaining work from the icechunk
        manifest, so the window just needs to cover any still-publishing inits.
        See "Operational updates" in docs/virtual_datasets.md.
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
        )
        return [job], template_ds


def _content_length(url: str) -> int:
    response = httpx.head(url)
    response.raise_for_status()
    return int(response.headers["content-length"])
