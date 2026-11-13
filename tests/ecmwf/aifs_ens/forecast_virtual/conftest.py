from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from reformatters.ecmwf.aifs_ens.forecast_virtual.region_job import (
    EcmwfAifsEnsForecastVirtualRegionJob,
)
from reformatters.ecmwf.aifs_ens.forecast_virtual.template_config import (
    EcmwfAifsEnsForecastVirtualTemplateConfig,
    EcmwfAifsEnsVirtualDataVar,
)

CONFIG = EcmwfAifsEnsForecastVirtualTemplateConfig()
FIXTURES = Path(__file__).parent / "fixtures"


def get_var(path: str) -> EcmwfAifsEnsVirtualDataVar:
    return next(var for var in CONFIG.data_vars if var.path == path)


@pytest.fixture(scope="module")
def template() -> xr.DataTree:
    return CONFIG.get_template(CONFIG.append_dim_start + pd.Timedelta("6h"))


def make_job(
    template: xr.DataTree, data_vars: Sequence[EcmwfAifsEnsVirtualDataVar] | None = None
) -> EcmwfAifsEnsForecastVirtualRegionJob:
    return EcmwfAifsEnsForecastVirtualRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template,
        data_vars=data_vars if data_vars is not None else CONFIG.data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
    )
