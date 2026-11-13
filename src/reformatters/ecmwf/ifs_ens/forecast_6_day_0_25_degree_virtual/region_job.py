from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.region_job import RegionJob
from reformatters.common.types import AppendDim, DatetimeLike, Timedelta, Timestamp
from reformatters.ecmwf.ifs_ens.virtual_region_job import (
    EcmwfIfsEnsForecastVirtualRegionJob,
    EcmwfIfsEnsForecastVirtualSourceFileCoord,
)
from reformatters.ecmwf.ifs_ens.virtual_template_config import (
    EcmwfIfsEnsVirtualDataVar,
)


class EcmwfIfsEnsForecast6Day025DegreeVirtualRegionJob(
    EcmwfIfsEnsForecastVirtualRegionJob
):
    resolution: ClassVar[Literal["0p25", "0p1"]] = "0p25"
    s3_fallback_before: ClassVar[Timestamp | None] = pd.Timestamp("2025-02-13T00")
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("24h")

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[EcmwfIfsEnsVirtualDataVar],
        reformat_job_name: str,
        job_fire_time: Timestamp | None = None,
    ) -> tuple[
        Sequence[
            RegionJob[
                EcmwfIfsEnsVirtualDataVar, EcmwfIfsEnsForecastVirtualSourceFileCoord
            ]
        ],
        xr.DataTree,
    ]:
        # The fire comes more than one init step after its target init. Ending the window
        # a step earlier drops the next init, which cannot publish before the poll
        # deadline, so the poll exits once the target init is complete.
        return super().operational_update_jobs(
            primary_store=primary_store,
            tmp_store=tmp_store,
            get_template_fn=get_template_fn,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            job_fire_time=(job_fire_time or pd.Timestamp.now()) - pd.Timedelta("6h"),
        )
