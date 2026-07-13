from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from reformatters.noaa.hrrr.forecast_18_hour_virtual.region_job import (
    NoaaHrrrForecast18HourVirtualRegionJob,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual.template_config import (
    NoaaHrrrForecast18HourVirtualTemplateConfig,
)
from reformatters.noaa.hrrr.forecast_virtual_region_job import (
    NoaaHrrrForecastVirtualSourceFileCoord,
)

TEMPLATE_CONFIG = NoaaHrrrForecast18HourVirtualTemplateConfig()


def test_source_file_coord_url_non_synoptic_init() -> None:
    # Hourly cycles between the synoptic 00/06/12/18 are what this dataset adds.
    coord = NoaaHrrrForecastVirtualSourceFileCoord(
        init_time=pd.Timestamp("2024-06-01T01:00"),
        lead_time=pd.Timedelta("18h"),
        domain="conus",
        file_type="sfc",
        data_vars=[TEMPLATE_CONFIG.data_vars[0]],
    )
    assert coord.get_url() == (
        "s3://noaa-hrrr-bdp-pds/hrrr.20240601/conus/hrrr.t01z.wrfsfcf18.grib2"
    )


def test_operational_update_jobs_single_polling_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2024-06-02T01:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *a, **kw: now))

    jobs, template_ds = NoaaHrrrForecast18HourVirtualRegionJob.operational_update_jobs(
        primary_store=Mock(),
        tmp_store=Path("unused-tmp.zarr"),
        get_template_fn=TEMPLATE_CONFIG.get_template,
        append_dim="init_time",
        all_data_vars=TEMPLATE_CONFIG.data_vars,
        reformat_job_name="test",
    )
    (job,) = jobs
    assert isinstance(job, NoaaHrrrForecast18HourVirtualRegionJob)
    assert job.processing_mode == "update"
    init_times = template_ds.to_dataset().get_index("init_time")
    # 6h window at the 1h cadence = the current + 5 prior cycles.
    assert job.region == slice(len(init_times) - 6, len(init_times))
