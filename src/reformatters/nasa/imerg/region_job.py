from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.download import get_local_path
from reformatters.common.logging import get_logger
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

from .earthdata_auth import get_authenticated_session
from .template_config import LATITUDE_SIZE, LONGITUDE_SIZE, NasaImergDataVar

log = get_logger(__name__)

_GESDISC_BASE = "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3"

# Early/Late filenames switched from the V07B to the V07C label on 2026-03-04.
# The change is packaging-only (algorithm identical), so we try the date-appropriate
# version first and fall back to the other.
_V07C_CUTOVER = pd.Timestamp("2026-03-04")


def _candidate_versions(time: pd.Timestamp) -> tuple[str, str]:
    return ("V07C", "V07B") if time >= _V07C_CUTOVER else ("V07B", "V07C")


def _reorient_imerg_array(raw: ArrayFloat32) -> ArrayFloat32:
    """Reorient a raw IMERG variable array to (latitude, longitude).

    Source arrays arrive as (longitude ascending, latitude ascending south->north).
    Transpose to (latitude, longitude) then flip latitude to descending north->south
    to match the template grid.
    """
    return np.ascontiguousarray(raw.T[::-1, :])


class NasaImergSourceFileCoord(SourceFileCoord):
    """Base coordinates of a single IMERG half-hourly granule (one HDF5 file).

    The filename/path layout is identical across runs; the per-run subclasses
    (in each variant's region_job.py) set the GES DISC product id and the run
    code that appears in the filename.
    """

    time: Timestamp

    # Set by per-run subclasses.
    gesdisc_product_id: ClassVar[str]
    run_code: ClassVar[str]

    def get_url(self, version: str = "V07C") -> str:
        end = self.time + pd.Timedelta(minutes=29, seconds=59)
        minutes_of_day = self.time.hour * 60 + self.time.minute
        filename = (
            f"3B-HHR-{self.run_code}.MS.MRG.3IMERG.{self.time:%Y%m%d}-S{self.time:%H%M%S}-"
            f"E{end:%H%M%S}.{minutes_of_day:04d}.{version}.HDF5"
        )
        return f"{_GESDISC_BASE}/{self.gesdisc_product_id}/{self.time:%Y}/{self.time.dayofyear:03d}/{filename}"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"time": self.time}


class NasaImergRegionJob(RegionJob[NasaImergDataVar, NasaImergSourceFileCoord]):
    # Set by the per-run subclasses to their NasaImergSourceFileCoord subclass.
    source_file_coord_class: ClassVar[type[NasaImergSourceFileCoord]]

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NasaImergDataVar],  # noqa: ARG002
    ) -> Sequence[NasaImergSourceFileCoord]:
        """Return a coord for each half-hourly granule covered by processing_region_ds.

        All requested variables live in a single IMERG file per granule, so the default
        single source group (one file per timestamp) is used.
        """
        return [
            self.source_file_coord_class(time=pd.Timestamp(time))
            for time in processing_region_ds["time"].values
        ]

    def download_file(self, coord: NasaImergSourceFileCoord) -> Path:
        """Download the IMERG granule from GES DISC (Earthdata Login) to local disk."""
        urls = [coord.get_url(version) for version in _candidate_versions(coord.time)]
        local_path = get_local_path(self.dataset_id, path=urlparse(urls[0]).path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        def _download() -> Path:
            session = get_authenticated_session()
            response = None
            for url in urls:
                # GES DISC issues a 303 redirect to a signed CloudFront URL.
                response = session.get(
                    url, timeout=30, stream=True, allow_redirects=True
                )
                # Stop on success; on any error (404 missing, or a server/auth error
                # on this version) fall through to try the other version label.
                if response.status_code < 400:
                    break
            assert response is not None
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.writelines(response.iter_content(chunk_size=8192))
            return local_path

        return retry(_download, max_attempts=10)

    def read_data(
        self,
        coord: NasaImergSourceFileCoord,
        data_var: NasaImergDataVar,
    ) -> ArrayFloat32:
        """Read one variable from an IMERG HDF5 granule as a (latitude, longitude) array."""
        assert coord.downloaded_path is not None, "File must be downloaded first"

        subdataset_path = (
            f"HDF5:{coord.downloaded_path}:{data_var.internal_attrs.h5_path}"
        )
        with rasterio.open(subdataset_path) as reader:
            # IMERG stores data variables in (time=1, lon=3600, lat=1800) order.
            # GDAL exposes the trailing two dims as a single band of shape (lon, lat).
            raw = reader.read(1, out_dtype=np.float32)

        # Reorient (lon, lat) -> (lat descending, lon ascending) to match the template grid.
        data: ArrayFloat32 = _reorient_imerg_array(raw)
        assert data.shape == (LATITUDE_SIZE, LONGITUDE_SIZE), (
            f"Unexpected IMERG array shape {data.shape}, expected {(LATITUDE_SIZE, LONGITUDE_SIZE)}"
        )

        data[data == np.float32(data_var.internal_attrs.source_fill_value)] = np.nan

        scale = data_var.internal_attrs.units_scale_factor
        if scale != 1.0:
            data *= np.float32(scale)

        return data

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[NasaImergDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[RegionJob[NasaImergDataVar, NasaImergSourceFileCoord]],
        xr.Dataset,
    ]:
        """Return RegionJobs to append the latest available IMERG granules."""
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
