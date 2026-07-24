from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, assert_never
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.download import get_local_path
from reformatters.common.logging import get_logger
from reformatters.common.materialized_region_job import MaterializedRegionJob
from reformatters.common.region_job import (
    CoordinateValue,
    RegionJob,
    SourceFileCoord,
)
from reformatters.common.retry import retry
from reformatters.common.types import (
    AppendDim,
    ArrayFloat32,
    DatetimeLike,
    Dim,
    Timestamp,
)
from reformatters.nasa.imerg.imerg_config_models import ImergRun, NasaImergDataVar
from reformatters.nasa.imerg.template_config import GRID_LAT_SIZE, GRID_LON_SIZE
from reformatters.nasa.nasa_auth import get_earthdata_session, get_pps_session

log = get_logger(__name__)

# Filenames carry V07C from this granule time onward, V07B before.
_V07C_START: dict[ImergRun, pd.Timestamp] = {
    "early": pd.Timestamp("2026-03-04T00:00"),
    "late": pd.Timestamp("2026-03-03T14:00"),
}

# The PPS NRT server (jsimpson) carries a rolling window of recent granules; older
# granules roll off to the GES DISC archive. Pick jsimpson for granules younger than
# this (operational updates), GES DISC for older ones (backfill).
_JSIMPSON_MAX_AGE = pd.Timedelta(days=3)

type DownloadSource = Literal["gesdisc", "jsimpson"]

_RUN_TAGS: dict[ImergRun, tuple[str, str, str]] = {
    # run -> (GES DISC product dir, filename run tag, jsimpson subdir)
    "early": ("GPM_3IMERGHHE.07", "3B-HHR-E", "early"),
    "late": ("GPM_3IMERGHHL.07", "3B-HHR-L", "late"),
}


class NasaImergAnalysisSourceFileCoord(SourceFileCoord):
    """One IMERG half-hourly granule for a given run and start time."""

    run: ImergRun
    time: Timestamp

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"time": self.time}

    @property
    def version(self) -> str:
        return "V07C" if self.time >= _V07C_START[self.run] else "V07B"

    def _source_by_age(self) -> DownloadSource:
        age = pd.Timestamp.now() - self.time
        return "jsimpson" if age < _JSIMPSON_MAX_AGE else "gesdisc"

    def _filename(self, version: str) -> str:
        _, run_tag, _ = _RUN_TAGS[self.run]
        start = self.time
        end = start + pd.Timedelta("29min59s")
        minutes_into_day = start.hour * 60 + start.minute
        return (
            f"{run_tag}.MS.MRG.3IMERG.{start.strftime('%Y%m%d')}"
            f"-S{start.strftime('%H%M%S')}-E{end.strftime('%H%M%S')}"
            f".{minutes_into_day:04d}.{version}"
        )

    def get_url(
        self, source: DownloadSource = "gesdisc", version: str | None = None
    ) -> str:
        version = version if version is not None else self.version
        product_dir, _, jsimpson_subdir = _RUN_TAGS[self.run]
        filename = self._filename(version)
        match source:
            case "gesdisc":
                base = "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3"
                return (
                    f"{base}/{product_dir}/{self.time.strftime('%Y')}/"
                    f"{self.time.dayofyear:03d}/{filename}.HDF5"
                )
            case "jsimpson":
                base = "https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg"
                return f"{base}/{jsimpson_subdir}/{self.time.strftime('%Y%m')}/{filename}.RT-H5"
            case _ as unreachable:
                assert_never(unreachable)

    def candidate_urls(self) -> list[tuple[DownloadSource, str]]:
        """Ordered (source, url) attempts. The primary source is chosen by granule
        age; GES DISC (the permanent archive) is the fallback when it isn't already
        primary. jsimpson is never a fallback — its rolling window can only ever hold
        recent granules, for which it is already the primary. Each source is tried with
        the computed version first, then the other to cover the V07B/V07C boundary."""
        primary = self._source_by_age()
        sources: list[DownloadSource] = [primary]
        if primary != "gesdisc":
            sources.append("gesdisc")
        versions = [self.version, "V07B" if self.version == "V07C" else "V07C"]
        return [
            (source, self.get_url(source, version))
            for source in sources
            for version in versions
        ]


class NasaImergAnalysisMaterializedRegionJob(
    MaterializedRegionJob[NasaImergDataVar, NasaImergAnalysisSourceFileCoord]
):
    run: ImergRun

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NasaImergDataVar],  # noqa: ARG002
    ) -> Sequence[NasaImergAnalysisSourceFileCoord]:
        return [
            NasaImergAnalysisSourceFileCoord(run=self.run, time=pd.Timestamp(time))
            for time in processing_region_ds["time"].values
        ]

    def download_file(self, coord: NasaImergAnalysisSourceFileCoord) -> Path:
        def _download() -> Path:
            for source, url in coord.candidate_urls():
                session = (
                    get_pps_session()
                    if source == "jsimpson"
                    else get_earthdata_session()
                )
                response = session.get(
                    url, timeout=30, stream=True, allow_redirects=True
                )
                if response.status_code == 404:
                    log.warning(f"File not found at {url}, trying next candidate")
                    continue
                response.raise_for_status()
                local_path = get_local_path(
                    self.template_ds.attrs["dataset_id"], path=urlparse(url).path
                )
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    f.writelines(response.iter_content(chunk_size=8192))
                return local_path
            raise FileNotFoundError(
                f"No IMERG granule found for {coord.run} {coord.time}"
            )

        return retry(_download, max_attempts=6)

    def read_data(
        self,
        coord: NasaImergAnalysisSourceFileCoord,
        data_var: NasaImergDataVar,
    ) -> ArrayFloat32:
        assert coord.downloaded_path is not None, "File must be downloaded first"
        subdataset = f"HDF5:{coord.downloaded_path}:{data_var.internal_attrs.h5_path}"
        with rasterio.open(subdataset) as reader:
            raw: ArrayFloat32 = reader.read(1, out_dtype=np.float32)

        # Source fields are stored (time=1, lon, lat); GDAL reads the 2D band as
        # (lon, lat). Transpose to (lat, lon) and flip latitude so it runs
        # north -> south, matching the descending latitude coordinate.
        assert raw.shape == (GRID_LON_SIZE, GRID_LAT_SIZE), (
            f"unexpected source band shape {raw.shape} for {data_var.name}"
        )
        data = raw.T[::-1, :]

        # Mask the exact source sentinel to NaN, then apply the units scale as a plain
        # multiply.
        data[data == np.float32(data_var.internal_attrs.source_fill_value)] = np.nan
        source_scale = data_var.internal_attrs.source_scale
        if source_scale != 1.0:
            data *= np.float32(source_scale)
        return data

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[NasaImergDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[RegionJob[NasaImergDataVar, NasaImergAnalysisSourceFileCoord]],
        xr.DataTree,
    ]:
        existing_ds = xr.open_zarr(primary_store)
        append_dim_start = existing_ds[append_dim].max()
        append_dim_end = pd.Timestamp.now()
        template_ds = get_template_fn(append_dim_end)

        jobs = cls.get_jobs(
            tmp_store=tmp_store,
            template_ds=template_ds,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            filter_start=append_dim_start,
        )
        return jobs, template_ds
