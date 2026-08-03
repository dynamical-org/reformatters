from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from reformatters.common.pydantic import replace
from reformatters.eccc.hrdps.forecast.region_job import EcccHrdpsForecastRegionJob
from reformatters.eccc.hrdps.forecast.template_config import (
    EcccHrdpsForecastTemplateConfig,
)
from reformatters.eccc.hrdps.region_job import EcccHrdpsSourceFileCoord


@pytest.fixture
def template_config() -> EcccHrdpsForecastTemplateConfig:
    return EcccHrdpsForecastTemplateConfig()


def _coord(
    template_config: EcccHrdpsForecastTemplateConfig, var_name: str
) -> EcccHrdpsSourceFileCoord:
    data_var = next(v for v in template_config.data_vars if v.name == var_name)
    return EcccHrdpsSourceFileCoord(
        init_time=pd.Timestamp("2026-07-09T06:00"),
        lead_time=pd.Timedelta("7h"),
        data_var=data_var,
    )


def test_source_file_coord_get_url(
    template_config: EcccHrdpsForecastTemplateConfig,
) -> None:
    coord = _coord(template_config, "temperature_2m")
    assert coord.get_url() == (
        "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/"
        "dynamical/eccc-hrdps-grib/20260709/06/007/"
        "20260709T06Z_MSC_HRDPS_TMP_AGL-2m_RLatLon0.0225_PT007H.grib2"
    )


def test_source_file_coord_get_fallback_url(
    template_config: EcccHrdpsForecastTemplateConfig,
) -> None:
    coord = _coord(template_config, "precipitation_surface")
    assert coord.get_fallback_url() == (
        "https://dd.weather.gc.ca/20260709/WXO-DD/model_hrdps/continental/2.5km/"
        "06/007/20260709T06Z_MSC_HRDPS_APCP_Sfc_RLatLon0.0225_PT007H.grib2"
    )


def test_source_file_var_groups(
    template_config: EcccHrdpsForecastTemplateConfig,
) -> None:
    groups = EcccHrdpsForecastRegionJob.source_file_var_groups(
        template_config.data_vars
    )
    # Each HRDPS grib file contains a single variable
    assert all(len(group) == 1 for group in groups)
    assert len(groups) == len(template_config.data_vars)


@pytest.mark.slow
@pytest.mark.parametrize(
    "var_name",
    [
        "temperature_2m",
        # read_data must match APCP's GRIB_ELEMENT with its accumulation window suffix (APCP02)
        "precipitation_surface",
    ],
)
def test_download_and_read_data(
    template_config: EcccHrdpsForecastTemplateConfig,
    var_name: str,
) -> None:
    data_var = next(v for v in template_config.data_vars if v.name == var_name)
    mock_ds = Mock()
    mock_ds.attrs = {"dataset_id": template_config.dataset_id}
    region_job = EcccHrdpsForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=mock_ds,
        data_vars=[data_var],
        append_dim=template_config.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )

    coord = EcccHrdpsSourceFileCoord(
        init_time=pd.Timestamp("2026-07-09T00:00"),
        lead_time=pd.Timedelta("2h"),
        data_var=data_var,
    )
    downloaded_path = region_job.download_file(coord)
    updated_coord = replace(coord, downloaded_path=downloaded_path)

    data = region_job.read_data(updated_coord, data_var)
    assert data.shape == (1290, 2540)
    assert data.dtype == np.float32
    assert not np.all(np.isnan(data))
