from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.common.download import FALLBACK_EXCEPTIONS, http_download_to_disk
from reformatters.common.grib_eccodes import read_unstructured_grib
from reformatters.common.iterating import item
from reformatters.common.logging import get_logger
from reformatters.common.materialized_region_job import MaterializedRegionJob
from reformatters.common.region_job import (
    CoordinateValue,
    InitLeadSourceFileCoord,
    RegionJob,
)
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import (
    AppendDim,
    ArrayFloat32,
    DatetimeLike,
    Dim,
)
from reformatters.dwd.icon_eu.forecast_5_day.template_config import DwdIconEuDataVar

from . import icosahedral_grid

log = get_logger(__name__)


class DwdIconEuForecast5DayIcosahedralSourceFileCoord(InitLeadSourceFileCoord):
    data_var: DwdIconEuDataVar

    def get_url(self) -> str:
        return (
            "https://s3-us-west-2.amazonaws.com/us-west-2.opendata.source.coop/"
            "dynamical/dwd-icon-grib/icon-eu/icosahedral/"
            f"{self.init_time:%Y-%m-%dT%H}/{self.variable_name_in_filename.upper()}/"
            f"PT{whole_hours(self.lead_time):03d}H00M.grib2"
        )

    def get_fallback_url(self) -> str:
        return (
            "https://opendata.dwd.de/weather/nwp/v1/m/icon-eu/p/"
            f"{self.variable_name_in_filename.upper()}/r/{self.init_time:%Y-%m-%dT%H}:00/"
            f"s/PT{whole_hours(self.lead_time):03d}H00M.grib2"
        )

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"init_time": self.init_time, "lead_time": self.lead_time}

    @property
    def variable_name_in_filename(self) -> str:
        return self.data_var.internal_attrs.variable_name_in_filename


class DwdIconEuForecast5DayIcosahedralRegionJob(
    MaterializedRegionJob[
        DwdIconEuDataVar, DwdIconEuForecast5DayIcosahedralSourceFileCoord
    ]
):
    @classmethod
    def source_file_var_groups(
        cls,
        data_vars: Sequence[DwdIconEuDataVar],
    ) -> Sequence[Sequence[DwdIconEuDataVar]]:
        return [[var] for var in data_vars]

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[DwdIconEuDataVar],
    ) -> Sequence[DwdIconEuForecast5DayIcosahedralSourceFileCoord]:
        init_times = pd.to_datetime(processing_region_ds["init_time"].values)
        lead_times = pd.to_timedelta(processing_region_ds["lead_time"].values)
        data_var = item(data_var_group)

        assert len(init_times) > 0
        assert len(lead_times) > 0

        return [
            DwdIconEuForecast5DayIcosahedralSourceFileCoord(
                init_time=init_time,
                lead_time=lead_time,
                data_var=data_var,
            )
            for init_time in init_times
            for lead_time in lead_times
        ]

    def download_file(
        self, coord: DwdIconEuForecast5DayIcosahedralSourceFileCoord
    ) -> Path:
        url = coord.get_url()
        try:
            file_path = http_download_to_disk(url, self.dataset_id)
        except FALLBACK_EXCEPTIONS as e:
            log.debug(f"Failed to download '{url}': {e}")
            fallback_url = coord.get_fallback_url()
            log.debug(f"Attempting to download from {fallback_url=}")
            file_path = http_download_to_disk(fallback_url, self.dataset_id)
        return file_path

    def read_data(
        self,
        coord: DwdIconEuForecast5DayIcosahedralSourceFileCoord,
        data_var: DwdIconEuDataVar,
    ) -> ArrayFloat32:
        assert coord.downloaded_path is not None
        result = read_unstructured_grib(
            coord.downloaded_path,
            expected_grid_uuid=icosahedral_grid.GRID_UUID,
            expected_number_of_points=icosahedral_grid.CELL_COUNT,
        )
        if data_var.attrs.units == "degree_Celsius":
            result -= np.float32(273.15)
        return result

    def apply_data_transformations(
        self, data_array: xr.DataArray, data_var: DwdIconEuDataVar
    ) -> None:
        attrs = data_var.internal_attrs

        if attrs.scale_factor is not None:
            data_array *= attrs.scale_factor

        if attrs.deaccumulate_to_rate:
            assert attrs.window_reset_frequency is not None
            log.info(
                f"Converting {data_var.name} from accumulations to rates along lead_time"
            )
            optional_kwargs: dict[str, Any] = {
                "invalid_below_threshold_rate": attrs.deaccumulation_invalid_below_threshold_rate,
                "expected_clamp_fraction": attrs.deaccumulation_expected_clamp_fraction,
            }
            optional_kwargs = {
                k: v for k, v in optional_kwargs.items() if v is not None
            }
            try:
                deaccumulate_to_rates_inplace(
                    data_array,
                    dim="lead_time",
                    reset_frequency=attrs.window_reset_frequency,
                    accumulation_type=attrs.deaccumulation_type,
                    **optional_kwargs,
                )
            except ValueError:
                log.exception(f"Error deaccumulating {data_var.name}")

        super().apply_data_transformations(data_array, data_var)

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[DwdIconEuDataVar],
        reformat_job_name: str,
        *,
        job_reference_time: pd.Timestamp,
    ) -> tuple[
        Sequence[
            RegionJob[DwdIconEuDataVar, DwdIconEuForecast5DayIcosahedralSourceFileCoord]
        ],
        xr.DataTree,
    ]:
        existing_ds = xr.open_zarr(primary_store, chunks=None)
        append_dim_start = existing_ds[append_dim].max()
        append_dim_end = job_reference_time
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
