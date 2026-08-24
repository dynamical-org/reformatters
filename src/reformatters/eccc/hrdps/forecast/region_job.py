from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.common.download import FALLBACK_EXCEPTIONS, http_download_to_disk
from reformatters.common.iterating import item
from reformatters.common.logging import get_logger
from reformatters.common.materialized_region_job import MaterializedRegionJob
from reformatters.common.region_job import (
    CoordinateValue,
    RegionJob,
    SourceFileCoord,
)
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import (
    AppendDim,
    ArrayFloat32,
    DatetimeLike,
    Dim,
    Timedelta,
    Timestamp,
)

from .template_config import EcccHrdpsDataVar

log = get_logger(__name__)

# ECCC's MSC Datamart keeps a rolling ~30 day window and publishes each run about 4 hours
# after its init time, hours before our archiver copies the run to Source Co-Op.
DATAMART_PREFERRED_AGE = pd.Timedelta("2D")


class EcccHrdpsForecastSourceFileCoord(SourceFileCoord):
    """Coordinates of a single source file to process.

    HRDPS is published as one single-message GRIB2 file per init time, lead time,
    field and level.
    """

    init_time: Timestamp
    lead_time: Timedelta
    data_var: EcccHrdpsDataVar

    def get_url(self) -> str:
        """URL of this file in dynamical.org's HRDPS archive on Source Co-Op."""
        return (
            "https://s3-us-west-2.amazonaws.com/us-west-2.opendata.source.coop/"
            f"dynamical/eccc-hrdps-grib/{self.init_time.strftime('%Y%m%d')}/{self._path_within_date()}"
        )

    def get_datamart_url(self) -> str:
        """URL of this file on ECCC's MSC Datamart."""
        return (
            f"https://dd.weather.gc.ca/{self.init_time.strftime('%Y%m%d')}/WXO-DD"
            f"/model_hrdps/continental/2.5km/{self._path_within_date()}"
        )

    def _path_within_date(self) -> str:
        """`{init hour}/{lead hour}/{file name}`, the layout both sources share."""
        internal_attrs = self.data_var.internal_attrs
        lead_time_hours = whole_hours(self.lead_time)
        return (
            f"{self.init_time.strftime('%H')}/{lead_time_hours:03d}/"
            f"{self.init_time.strftime('%Y%m%dT%HZ')}_MSC_HRDPS_"
            f"{internal_attrs.grib_field}_{internal_attrs.grib_level}"
            f"_RLatLon0.0225_PT{lead_time_hours:03d}H.grib2"
        )

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"init_time": self.init_time, "lead_time": self.lead_time}


class EcccHrdpsForecastRegionJob(
    MaterializedRegionJob[EcccHrdpsDataVar, EcccHrdpsForecastSourceFileCoord]
):
    # The Datamart's throughput plateaus at 8 concurrent downloads and it asks callers
    # to keep their request rate modest, see src/reformatters/eccc/README.md.
    download_parallelism: int = 8

    @classmethod
    def source_file_var_groups(
        cls,
        data_vars: Sequence[EcccHrdpsDataVar],
    ) -> Sequence[Sequence[EcccHrdpsDataVar]]:
        # Each HRDPS grib file contains a single variable.
        return [[data_var] for data_var in data_vars]

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[EcccHrdpsDataVar],
    ) -> Sequence[EcccHrdpsForecastSourceFileCoord]:
        data_var = item(data_var_group)
        lead_times = pd.to_timedelta(processing_region_ds["lead_time"].values)
        if not data_var.has_hour_0_values():
            lead_times = lead_times[lead_times > pd.Timedelta(0)]

        return [
            EcccHrdpsForecastSourceFileCoord(
                init_time=init_time,
                lead_time=lead_time,
                data_var=data_var,
            )
            for init_time in pd.to_datetime(processing_region_ds["init_time"].values)
            for lead_time in lead_times
        ]

    def download_file(self, coord: EcccHrdpsForecastSourceFileCoord) -> Path:
        recent = pd.Timestamp.now() - coord.init_time < DATAMART_PREFERRED_AGE
        primary_url, fallback_url = (
            (coord.get_datamart_url(), coord.get_url())
            if recent
            else (coord.get_url(), coord.get_datamart_url())
        )
        try:
            return http_download_to_disk(primary_url, self.dataset_id)
        except FALLBACK_EXCEPTIONS as e:
            # An update that falls back to the archive gets the previous run at best,
            # because the archive lags the Datamart by hours.
            log.info(f"Failed to download '{primary_url}', falling back: {e}")
            return http_download_to_disk(fallback_url, self.dataset_id)

    def read_data(
        self,
        coord: EcccHrdpsForecastSourceFileCoord,
        data_var: EcccHrdpsDataVar,
    ) -> ArrayFloat32:
        assert coord.downloaded_path is not None  # for type check, system guarantees it
        with rasterio.open(coord.downloaded_path) as reader:
            assert reader.count == 1, (
                f"Expected exactly 1 message in each HRDPS grib file, found {reader.count}. "
                f"{data_var.name=}, {coord.downloaded_path=}"
            )
            result: ArrayFloat32 = reader.read(1, out_dtype=np.float32)
            return result

    def apply_data_transformations(
        self, data_array: xr.DataArray, data_var: EcccHrdpsDataVar
    ) -> None:
        internal_attrs = data_var.internal_attrs

        if internal_attrs.deaccumulate_to_rate:
            assert internal_attrs.window_reset_frequency is not None
            log.info(
                f"Converting {data_var.name} from accumulations to rates along lead_time"
            )
            try:
                deaccumulate_to_rates_inplace(
                    data_array,
                    dim="lead_time",
                    reset_frequency=internal_attrs.window_reset_frequency,
                    invalid_below_threshold_rate=internal_attrs.deaccumulation_invalid_below_threshold_rate,
                    expected_clamp_fraction=internal_attrs.deaccumulation_expected_clamp_fraction,
                )
            except ValueError:
                # The array is deaccumulated either way; the raise reports only that more
                # steps than expected were clamped to 0 or invalidated, so log it rather
                # than discard an otherwise good forecast.
                log.exception(f"Error deaccumulating {data_var.name}")

        if (scale_factor := internal_attrs.scale_factor) is not None:
            data_array.values *= np.float32(scale_factor)

        if data_var.attrs.flag_values is not None:
            # The source's lossy packing leaves a small fraction of cells fractionally
            # off their integer code.
            np.round(data_array.values, out=data_array.values)

        super().apply_data_transformations(data_array, data_var)

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[EcccHrdpsDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[RegionJob[EcccHrdpsDataVar, EcccHrdpsForecastSourceFileCoord]],
        xr.DataTree,
    ]:
        existing_ds = xr.open_zarr(primary_store, chunks=None, decode_timedelta=True)
        append_dim_start = pd.Timestamp(existing_ds[append_dim].max().item())
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
