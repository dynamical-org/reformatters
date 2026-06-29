from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.config_models import ROOT
from reformatters.common.download import s3_download_to_disk, s3_store
from reformatters.common.logging import get_logger
from reformatters.common.region_job import CoordinateValue, RegionJob
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import AppendDim, DatetimeLike, Dim
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,
)
from reformatters.noaa.hrrr.forecast_48_hour_spatial.template_config import (
    MODEL_LEVELS,
    PRESSURE_LEVELS,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from reformatters.noaa.hrrr.region_job import DownloadSource, NoaaHrrrSourceFileCoord
from reformatters.noaa.noaa_grib_index import _lead_time_str, parse_grib_index_lines
from reformatters.noaa.noaa_utils import has_hour_0_values

log = get_logger(__name__)

_S3_LOCATION_PREFIX = "s3://noaa-hrrr-bdp-pds/"
_S3_HTTPS_PREFIX = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
_S3_BUCKET_REGION = "us-east-1"

# A representative vertical level per group-sourced product file. A prs/nat file
# carries only vertical-group variables (no root chunk), so the per-file manifest
# probe needs one concrete level to resolve to a single chunk; commits are atomic
# per file, so this level being present means the whole file is present.
_REPRESENTATIVE_LEVEL: dict[str, tuple[Dim, int]] = {
    "prs": ("pressure_level", PRESSURE_LEVELS[0]),
    "nat": ("model_level", MODEL_LEVELS[0]),
}


class NoaaHrrrForecast48HourSpatialSourceFileCoord(NoaaHrrrSourceFileCoord):
    """One HRRR product file (init_time, lead_time, file_type) and the vars it packs."""

    def get_url(self, source: DownloadSource = "s3") -> str:
        """The s3:// source location refs point at, matching the virtual chunk container."""
        url = super().get_url(source=source)
        assert url.startswith(_S3_HTTPS_PREFIX), url
        return _S3_LOCATION_PREFIX + url.removeprefix(_S3_HTTPS_PREFIX)

    def get_index_url(self) -> str:
        return self.get_url() + ".idx"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        loc: dict[Dim, CoordinateValue] = {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
        }
        if (rep := _REPRESENTATIVE_LEVEL.get(self.file_type)) is not None:
            dim, level = rep
            loc[dim] = level
        return loc


class NoaaHrrrForecast48HourSpatialRegionJob(
    VirtualRegionJob[NoaaHrrrDataVar, NoaaHrrrForecast48HourSpatialSourceFileCoord]
):
    """RegionJob for the virtual HRRR 48-hour spatial forecast dataset."""

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaHrrrDataVar],
    ) -> Sequence[NoaaHrrrForecast48HourSpatialSourceFileCoord]:
        init_times = pd.to_datetime(processing_region_ds["init_time"].values)
        lead_times = pd.to_timedelta(processing_region_ds["lead_time"].values)
        file_types = sorted({v.internal_attrs.hrrr_file_type for v in data_var_group})

        coords = []
        for init_time in init_times:
            for lead_time in lead_times:
                for file_type in file_types:
                    # Accumulated/categorical vars have no valid hour-0 data, so drop
                    # them at lead 0 (keeps the completeness validator consistent).
                    vars_in_file = [
                        var
                        for var in data_var_group
                        if var.internal_attrs.hrrr_file_type == file_type
                        and (lead_time > pd.Timedelta(0) or has_hour_0_values(var))
                    ]
                    if not vars_in_file:
                        continue
                    coords.append(
                        NoaaHrrrForecast48HourSpatialSourceFileCoord(
                            init_time=init_time,
                            lead_time=lead_time,
                            domain="conus",
                            file_type=file_type,
                            data_vars=vars_in_file,
                        )
                    )
        return coords

    def discover_available(
        self, pending: list[NoaaHrrrForecast48HourSpatialSourceFileCoord]
    ) -> list[tuple[NoaaHrrrForecast48HourSpatialSourceFileCoord, int]]:
        return discover_available_by_obstore_listing(
            pending,
            store=s3_store(_S3_LOCATION_PREFIX, region=_S3_BUCKET_REGION),
            location_prefix=_S3_LOCATION_PREFIX,
            require_index=True,
        )

    def file_refs(
        self, coord: NoaaHrrrForecast48HourSpatialSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        index_path = s3_download_to_disk(
            coord.get_index_url(), self.dataset_id, region=_S3_BUCKET_REGION
        )
        try:
            index_lines = parse_grib_index_lines(index_path)
        finally:
            index_path.unlink()

        lookup = self._message_lookup(coord.data_vars, whole_hours(coord.lead_time))
        # Each message's end byte is the next message's start; the last is the file end.
        starts = [start for start, *_ in index_lines]
        ends = [*starts[1:], file_size]

        out_loc_base = dict(coord.out_loc())
        location = coord.get_url()
        refs = []
        for (start, element, level, window), end in zip(index_lines, ends, strict=True):
            # One message can fill several variables: at lead 1h the run-total and
            # per-hour accumulation windows render the same idx string (0->1), so a
            # single APCP/WEASD/FROZR message feeds both variants.
            matches = lookup.get((element, level, window))
            if not matches:
                continue
            # Byte ranges past the data file mean a stale/mismatched index; skip it.
            if end > file_size or end <= start:
                log.warning(
                    f"Skipping {location}: index byte ranges fall outside the "
                    f"{file_size}-byte data file; stale or mismatched index"
                )
                return []
            for var, level_label in matches:
                refs.append(
                    VirtualRef(
                        data_var=var,
                        out_loc={**out_loc_base, **level_label},
                        location=location,
                        offset=start,
                        length=end - start,
                    )
                )
        return refs

    def _message_lookup(
        self, data_vars: Sequence[NoaaHrrrDataVar], lead_hours: int
    ) -> dict[
        tuple[str, str, str], list[tuple[NoaaHrrrDataVar, dict[Dim, CoordinateValue]]]
    ]:
        """Map each (element, idx level string, idx window string) to the variables it
        fills and the vertical label each ref carries. A root var contributes one entry;
        a vertical-group var one per level (its grib_index_level is a "{level} ..."
        format string). The mapping is one-to-many: at lead 1h the run-total and
        per-hour accumulation variants of APCP/WEASD/FROZR share a key (window 0->1),
        and both must be filled from the single matching message."""
        lookup: dict[
            tuple[str, str, str],
            list[tuple[NoaaHrrrDataVar, dict[Dim, CoordinateValue]]],
        ] = {}
        for var in data_vars:
            window = _lead_time_str(var, lead_hours)
            element = var.internal_attrs.grib_element
            if var.group is ROOT:
                key = (element, var.internal_attrs.grib_index_level, window)
                lookup.setdefault(key, []).append((var, {}))
            else:
                dim = var.group  # group name equals its dimension name
                level_format = var.internal_attrs.grib_index_level
                for level in self.template_ds[var.path].get_index(dim):
                    key = (element, level_format.format(level=level), window)
                    lookup.setdefault(key, []).append((var, {dim: int(level)}))
        return lookup

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,  # noqa: ARG003 - the icechunk manifest, not a coordinate, tracks ingested data
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[NoaaHrrrDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[
            RegionJob[NoaaHrrrDataVar, NoaaHrrrForecast48HourSpatialSourceFileCoord]
        ],
        xr.DataTree,
    ]:
        """A single polling job over the recent inits (14h window = current + 2 prior).

        14h = two 6h cycles back + ~2h publication slack, so a couple of missed runs
        still self-heal. Polls until every expected file is ingested;
        filter_already_present derives the remaining work from the manifest. See
        "Operational updates" in docs/virtual_datasets.md.
        """
        append_dim_end = pd.Timestamp.now()
        template_ds = get_template_fn(append_dim_end)
        init_times = template_ds.to_dataset().get_index(append_dim)
        window_start = int(
            init_times.searchsorted(append_dim_end - pd.Timedelta("14h"))
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
