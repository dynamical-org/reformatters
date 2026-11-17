from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.common.download import FALLBACK_EXCEPTIONS, http_download_to_disk
from reformatters.common.logging import get_logger
from reformatters.common.materialized_region_job import MaterializedRegionJob
from reformatters.common.region_job import (
    CoordinateValue,
    InitLeadSourceFileCoord,
    RegionJob,
)
from reformatters.common.types import (
    AppendDim,
    DatetimeLike,
    Dim,
)
from reformatters.dwd.icon_eu.icon_eu_template_config import DwdIconEuDataVar

log = get_logger(__name__)


class DwdIconEuSourceFileCoord(InitLeadSourceFileCoord):
    data_var: DwdIconEuDataVar

    @abstractmethod
    def get_url(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_fallback_url(self) -> str:
        raise NotImplementedError

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        """Return the output location for this file's data in the dataset."""
        return {"init_time": self.init_time, "lead_time": self.lead_time}

    @property
    def variable_name_in_filename(self) -> str:
        return self.data_var.internal_attrs.variable_name_in_filename


class DwdIconEuRegionJob[SourceFileCoordT: DwdIconEuSourceFileCoord](
    MaterializedRegionJob[DwdIconEuDataVar, SourceFileCoordT]
):
    @classmethod
    def source_file_var_groups(
        cls,
        data_vars: Sequence[DwdIconEuDataVar],
    ) -> Sequence[Sequence[DwdIconEuDataVar]]:
        # Each ICON-EU grib file contains a single variable.
        return [[var] for var in data_vars]

    def download_file(self, coord: SourceFileCoordT) -> Path:
        """Download the file for the given coordinate and return the local path.

        Downloads the `.grib.bz2` file and returns its local `Path`.
        """
        url = coord.get_url()
        try:
            bz2_file_path = http_download_to_disk(url, self.dataset_id)
        except FALLBACK_EXCEPTIONS as e:
            log.debug(f"Failed to download '{url}': {e}")
            fallback_url = coord.get_fallback_url()
            log.debug(f"Attempting to download from {fallback_url=}")
            bz2_file_path = http_download_to_disk(fallback_url, self.dataset_id)
        return bz2_file_path

    def apply_data_transformations(
        self, data_array: xr.DataArray, data_var: DwdIconEuDataVar
    ) -> None:
        """Apply in-place data transformations to the output data array for a given data variable."""
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
        Sequence[RegionJob[DwdIconEuDataVar, SourceFileCoordT]],
        xr.DataTree,
    ]:
        """Return RegionJob instances to update the dataset from its current state to the latest available data."""
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
