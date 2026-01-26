from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from reformatters.common.config_models import DataVarAttrs, Encoding
from reformatters.dwd.icon_eu.forecast.region_job import (
    DwdIconEuForecastRegionJob,
    DwdIconEuForecastSourceFileCoord,
)
from reformatters.dwd.icon_eu.forecast.template_config import (
    DwdIconEuDataVar,
    DwdIconEuForecastTemplateConfig,
    DwdIconEuInternalAttrs,
)


@pytest.fixture
def t_2m_data_var() -> DwdIconEuDataVar:
    return DwdIconEuDataVar(
        name="temperature_2m",
        encoding=Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=(1, 100, 100),
            shards=(1, 100, 100),
        ),
        attrs=DataVarAttrs(
            short_name="2t",
            long_name="2 metre temperature",
            units="degree_Celsius",
            step_type="instant",
            comment="Temperature at 2m above ground, averaged over all tiles of a grid point.",
            standard_name="air_temperature",
        ),
        internal_attrs=DwdIconEuInternalAttrs(
            variable_name_in_filename="t_2m",
            keep_mantissa_bits=7,
        ),
    )


@pytest.fixture
def source_file_coord(
    t_2m_data_var: DwdIconEuDataVar,
) -> DwdIconEuForecastSourceFileCoord:
    return DwdIconEuForecastSourceFileCoord(
        init_time=pd.Timestamp("2000-01-01T00:00"),
        lead_time=pd.Timedelta(0),
        data_var=t_2m_data_var,
    )


@pytest.fixture
def basename() -> str:
    return "icon-eu_europe_regular-lat-lon_single-level_2000010100_000_T_2M.grib2.bz2"


def test_source_file_coord_get_url(
    source_file_coord: DwdIconEuForecastSourceFileCoord,
    basename: str,
) -> None:
    dir_name = "https://source.coop/dynamical/dwd-icon-grib/icon-eu/regular-lat-lon/2000-01-01T00Z/t_2m/"
    expected = dir_name + basename
    assert source_file_coord.get_url() == expected


def test_source_file_coord_get_fallback_url(
    source_file_coord: DwdIconEuForecastSourceFileCoord,
    basename: str,
) -> None:
    dir_name = "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/t_2m/"
    expected = dir_name + basename
    assert source_file_coord.get_fallback_url() == expected


def test_region_job_generete_source_file_coords() -> None:
    template_config = DwdIconEuForecastTemplateConfig()
    template_ds = template_config.get_template(
        template_config.append_dim_start + template_config.append_dim_frequency
    )

    # use `model_construct` to skip pydantic validation so we can pass mock stores
    region_job = DwdIconEuForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 10),
        reformat_job_name="test",
    )

    processing_region_ds, _ = region_job._get_region_datasets()

    # Test with a single data variable
    source_file_coords = region_job.generate_source_file_coords(
        processing_region_ds, template_config.data_vars[:1]
    )

    # 1 init_time x 1 variable x 93 time steps = 93
    assert len(source_file_coords) == 93
