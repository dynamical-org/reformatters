from unittest.mock import Mock

import pandas as pd

from reformatters.contrib.nasa.smap.level3_36km_v9.region_job import (
    NasaSmapLevel336KmV9RegionJob,
    NasaSmapLevel336KmV9SourceFileCoord,
)
from reformatters.contrib.nasa.smap.level3_36km_v9.template_config import (
    NasaSmapLevel336KmV9TemplateConfig,
)


def test_source_file_coord_get_url() -> None:
    coord = NasaSmapLevel336KmV9SourceFileCoord(time=pd.Timestamp("2025-09-30"))
    expected_url = "https://n5eil01u.ecs.nsidc.org/SMAP/SPL3SMP.009/2025.09.30/SMAP_L3_SM_P_20250930_R19240_001.h5"
    assert coord.get_url() == expected_url


def test_source_file_coord_out_loc() -> None:
    coord = NasaSmapLevel336KmV9SourceFileCoord(time=pd.Timestamp("2025-09-30"))
    assert coord.out_loc() == {"time": pd.Timestamp("2025-09-30")}


def test_region_job_generate_source_file_coords() -> None:
    template_config = NasaSmapLevel336KmV9TemplateConfig()
    template_ds = template_config.get_template(pd.Timestamp("2000-01-23"))

    region_job = NasaSmapLevel336KmV9RegionJob(
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

    assert len(source_file_coords) == len(processing_region_ds["time"])
    for i, coord in enumerate(source_file_coords):
        assert coord.time == processing_region_ds["time"].values[i]
