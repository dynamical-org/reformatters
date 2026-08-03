from unittest.mock import Mock

import pandas as pd
import pytest

from reformatters.eccc.hrdps.forecast.region_job import (
    EcccHrdpsForecastRegionJob,
    EcccHrdpsForecastSourceFileCoord,
)
from reformatters.eccc.hrdps.forecast.template_config import (
    EcccHrdpsForecastTemplateConfig,
)


@pytest.fixture
def template_config() -> EcccHrdpsForecastTemplateConfig:
    return EcccHrdpsForecastTemplateConfig()


def test_source_file_coord_out_loc(
    template_config: EcccHrdpsForecastTemplateConfig,
) -> None:
    coord = EcccHrdpsForecastSourceFileCoord(
        init_time=pd.Timestamp("2026-07-09T06:00"),
        lead_time=pd.Timedelta("3h"),
        data_var=template_config.data_vars[0],
    )
    assert coord.out_loc() == {
        "init_time": pd.Timestamp("2026-07-09T06:00"),
        "lead_time": pd.Timedelta("3h"),
    }


@pytest.mark.parametrize(
    ("var_name", "expected_min_lead_hours"),
    [
        ("temperature_2m", 0),  # instant, has hour 0 values
        ("precipitation_surface", 1),  # no hour 0 file
        ("downward_short_wave_radiation_flux_surface", 1),  # hour 0 is all zeros
    ],
)
def test_generate_source_file_coords(
    template_config: EcccHrdpsForecastTemplateConfig,
    var_name: str,
    expected_min_lead_hours: int,
) -> None:
    data_var = next(v for v in template_config.data_vars if v.name == var_name)
    template_ds = template_config.get_template(pd.Timestamp("2026-07-09T12:00"))

    region_job = EcccHrdpsForecastRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=[data_var],
        append_dim=template_config.append_dim,
        region=slice(0, 2),
        reformat_job_name="test",
    )
    processing_region_ds, _ = region_job._get_region_datasets()

    coords = region_job.generate_source_file_coords(processing_region_ds, [data_var])

    num_lead_times = 49 - expected_min_lead_hours
    assert len(coords) == 2 * num_lead_times
    assert min(c.lead_time for c in coords) == pd.Timedelta(
        hours=expected_min_lead_hours
    )
    assert max(c.lead_time for c in coords) == pd.Timedelta("48h")
    assert {c.init_time for c in coords} == {
        pd.Timestamp("2026-07-09T00:00"),
        pd.Timestamp("2026-07-09T06:00"),
    }
