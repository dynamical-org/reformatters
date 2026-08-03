from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.deaccumulation import (
    deaccumulate_to_rates_inplace_logging_errors,
)
from reformatters.common.download import (
    DOWNLOAD_FALLBACK_EXCEPTIONS,
    http_download_to_disk,
)
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
from reformatters.eccc.hrdps.archive_gribs.copy_files_from_eccc import (
    MSC_DATAMART_HOST,
)
from reformatters.eccc.hrdps.hrdps_config_models import EcccHrdpsDataVar
from reformatters.eccc.hrdps.template_config import HRDPS_INIT_FREQUENCY

log = get_logger(__name__)

DYNAMICAL_GRIB_ARCHIVE_URL: Final[str] = (
    "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/dynamical/eccc-hrdps-grib"
)


class EcccHrdpsSourceFileCoord(SourceFileCoord):
    """Coordinates of a single source file to process.

    HRDPS is published as one single-message GRIB2 file per (init time, lead time, variable, level).
    """

    init_time: Timestamp
    lead_time: Timedelta
    data_var: EcccHrdpsDataVar

    def get_url(self) -> str:
        """URL on dynamical.org's public archive of HRDPS gribs hosted on Source Co-Op."""
        date_str = self.init_time.strftime("%Y%m%d")
        hour_str = self.init_time.strftime("%H")
        lead_str = f"{whole_hours(self.lead_time):03d}"
        return f"{DYNAMICAL_GRIB_ARCHIVE_URL}/{date_str}/{hour_str}/{lead_str}/{self._basename()}"

    def get_fallback_url(self) -> str:
        """URL on ECCC's MSC Datamart, which retains a rolling ~30 days."""
        date_str = self.init_time.strftime("%Y%m%d")
        hour_str = self.init_time.strftime("%H")
        lead_str = f"{whole_hours(self.lead_time):03d}"
        return (
            f"{MSC_DATAMART_HOST}/{date_str}/WXO-DD/model_hrdps/continental/2.5km/"
            f"{hour_str}/{lead_str}/{self._basename()}"
        )

    def _basename(self) -> str:
        init_str = self.init_time.strftime("%Y%m%dT%H")
        lead_str = f"{whole_hours(self.lead_time):03d}"
        variable = self.data_var.internal_attrs.variable_name_in_filename
        return f"{init_str}Z_MSC_HRDPS_{variable}_RLatLon0.0225_PT{lead_str}H.grib2"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        raise NotImplementedError  # depends on if the dataset is a forecast or analysis


class EcccHrdpsRegionJob(
    MaterializedRegionJob[EcccHrdpsDataVar, EcccHrdpsSourceFileCoord]
):
    """Base RegionJob for HRDPS based datasets. Subclassed by specific HRDPS datasets."""

    @classmethod
    def source_file_var_groups(
        cls,
        data_vars: Sequence[EcccHrdpsDataVar],
    ) -> Sequence[Sequence[EcccHrdpsDataVar]]:
        # Each HRDPS grib file contains a single variable.
        return [[var] for var in data_vars]

    def download_file(self, coord: EcccHrdpsSourceFileCoord) -> Path:
        url = coord.get_url()
        try:
            return http_download_to_disk(url, self.dataset_id)
        except DOWNLOAD_FALLBACK_EXCEPTIONS as e:
            log.debug(f"Failed to download '{url}': {e}")
            fallback_url = coord.get_fallback_url()
            log.debug(f"Attempting to download from {fallback_url=}")
            return http_download_to_disk(fallback_url, self.dataset_id)

    def read_data(
        self,
        coord: EcccHrdpsSourceFileCoord,
        data_var: EcccHrdpsDataVar,
    ) -> ArrayFloat32:
        assert coord.downloaded_path is not None  # for type check, system guarantees it

        expected_element = data_var.internal_attrs.grib_element
        if data_var.internal_attrs.include_lead_time_suffix:
            expected_element += f"{whole_hours(coord.lead_time):02d}"

        with rasterio.open(coord.downloaded_path) as reader:
            assert reader.count == 1, (
                f"Expected exactly 1 message in each HRDPS grib file, found {reader.count=}. "
                f"{coord.downloaded_path=}"
            )
            element = reader.tags(1)["GRIB_ELEMENT"]
            assert element == expected_element, (
                f"Expected GRIB_ELEMENT {expected_element!r}, found {element!r}. "
                f"{coord.downloaded_path=}"
            )
            result: ArrayFloat32 = reader.read(1, out_dtype=np.float32)
            return result

    def apply_data_transformations(
        self, data_array: xr.DataArray, data_var: EcccHrdpsDataVar
    ) -> None:
        attrs = data_var.internal_attrs

        if attrs.deaccumulate_to_rate:
            deaccum_dim = "lead_time" if "lead_time" in data_array.dims else "time"
            # HRDPS accumulations run from forecast start without resetting. Along
            # lead_time that's a never-resetting window; flattened into an hourly
            # analysis (leads 1-6h per init) the accumulation restarts at each init.
            reset_frequency = (
                pd.Timedelta.max if deaccum_dim == "lead_time" else HRDPS_INIT_FREQUENCY
            )
            deaccumulate_to_rates_inplace_logging_errors(
                data_array,
                dim=deaccum_dim,
                reset_frequency=reset_frequency,
                invalid_below_threshold_rate=attrs.deaccumulation_invalid_below_threshold_rate,
                expected_clamp_fraction=attrs.deaccumulation_expected_clamp_fraction,
            )

        if (scale_factor := attrs.scale_factor) is not None:
            data_array.values *= np.float32(scale_factor)

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
        Sequence[RegionJob[EcccHrdpsDataVar, EcccHrdpsSourceFileCoord]],
        xr.DataTree,
    ]:
        """Return RegionJob instances to update the dataset from its current state to the latest available data."""
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
