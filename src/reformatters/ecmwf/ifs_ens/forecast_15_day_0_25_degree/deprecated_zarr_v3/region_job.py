from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.region_job import RegionJob
from reformatters.common.types import AppendDim, DatetimeLike
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar
from reformatters.ecmwf.ecmwf_utils import EcmwfOpenDataSource
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.region_job import (
    EcmwfIfsEnsForecast15Day025DegreeRegionJob,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.source_file_coord import (
    IfsEnsSourceFileCoord,
)


class EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3RegionJob(
    EcmwfIfsEnsForecast15Day025DegreeRegionJob
):
    # Stays off the throttled s3://ecmwf-forecasts bucket.
    open_data_sources: ClassVar[tuple[EcmwfOpenDataSource, ...]] = ("gcs",)

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[EcmwfDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[RegionJob[EcmwfDataVar, IfsEnsSourceFileCoord]],
        xr.DataTree,
    ]:
        jobs, template_ds = super().operational_update_jobs(
            primary_store=primary_store,
            tmp_store=tmp_store,
            get_template_fn=get_template_fn,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
        )
        stored = xr.open_zarr(primary_store, chunks=None)[append_dim].values
        expected = template_ds.dataset[append_dim].values[: stored.size]
        assert np.array_equal(stored, expected), (
            f"Stored {append_dim} labels do not match the template; refusing to write"
        )
        return jobs, template_ds
