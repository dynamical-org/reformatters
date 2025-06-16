from unittest.mock import Mock

import pandas as pd

from reformatters.noaa.gfs.forecast.region_job import (
    NoaaGfsForecastRegionJob,
    NoaaGfsSourceFileCoord,
)
from reformatters.noaa.gfs.forecast.template_config import NoaaGfsForecastTemplateConfig


def test_source_file_coord_get_url() -> None:
    coord = NoaaGfsSourceFileCoord(
        init_time=pd.Timestamp("2000-01-01T00:00"), lead_time=pd.Timedelta(hours=0)
    )
    expected = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20000101/00/atmos/gfs.t00z.pgrb2.0p25.f000"
    assert coord.get_url() == expected


def test_region_job_generete_source_file_coords() -> None:
    template_config = NoaaGfsForecastTemplateConfig()
    template_ds = template_config.get_template(pd.Timestamp("2000-01-23"))

    region_job = NoaaGfsForecastRegionJob(
        final_store=Mock(),
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=[Mock(), Mock()],
        append_dim=template_config.append_dim,
        region=slice(0, 10),
        reformat_job_name="test",
    )

    processing_region_ds, output_region_ds = region_job._get_region_datasets()

    source_file_coords = region_job.generate_source_file_coords(
        processing_region_ds, [Mock()]
    )

    assert isinstance(source_file_coords, list)
