from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from reformatters.ecmwf.ifs_ens.forecast_6_day_0_25_degree_virtual.region_job import (
    EcmwfIfsEnsForecast6Day025DegreeVirtualRegionJob,
)
from reformatters.ecmwf.ifs_ens.forecast_6_day_0_25_degree_virtual.template_config import (
    EcmwfIfsEnsForecast6Day025DegreeVirtualTemplateConfig,
)
from reformatters.ecmwf.ifs_ens.virtual_template_config import EcmwfIfsEnsVirtualDataVar

CONFIG = EcmwfIfsEnsForecast6Day025DegreeVirtualTemplateConfig()
FIXTURES = Path(__file__).parents[1] / "fixtures"


def get_var(path: str) -> EcmwfIfsEnsVirtualDataVar:
    return next(var for var in CONFIG.data_vars if var.path == path)


@pytest.fixture(scope="module")
def template() -> xr.DataTree:
    return CONFIG.get_template(CONFIG.append_dim_start + pd.Timedelta("6h"))


def make_job(
    template: xr.DataTree, data_vars: Sequence[EcmwfIfsEnsVirtualDataVar] | None = None
) -> EcmwfIfsEnsForecast6Day025DegreeVirtualRegionJob:
    return EcmwfIfsEnsForecast6Day025DegreeVirtualRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template,
        data_vars=data_vars if data_vars is not None else CONFIG.data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
    )
