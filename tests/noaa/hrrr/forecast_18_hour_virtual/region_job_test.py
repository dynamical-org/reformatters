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
from reformatters.noaa.hrrr.virtual_region_job import (
    NoaaHrrrForecastVirtualSourceFileCoord,
)

TEMPLATE_CONFIG = NoaaHrrrForecast18HourVirtualTemplateConfig()


def test_source_file_coord_url_non_synoptic_init() -> None:
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


def test_operational_update_jobs_cover_six_hourly_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2024-06-02T01:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *args, **kwargs: now))

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
    assert job.region == slice(len(init_times) - 6, len(init_times))


def test_generate_source_file_coords_uses_the_declared_coord_class() -> None:
    class RoutedCoord(NoaaHrrrForecastVirtualSourceFileCoord):
        pass

    class RoutedJob(NoaaHrrrForecast18HourVirtualRegionJob):
        source_file_coord_class = RoutedCoord

    template_ds = TEMPLATE_CONFIG.get_template(pd.Timestamp("2018-07-13T13:00"))
    job = RoutedJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=TEMPLATE_CONFIG.data_vars[:2],
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
    )
    coords = job.generate_source_file_coords(job._processing_region_ds(), job.data_vars)
    assert coords
    assert all(type(coord) is RoutedCoord for coord in coords)
