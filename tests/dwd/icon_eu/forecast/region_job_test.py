# from unittest.mock import Mock


import pandas as pd

from reformatters.dwd.icon_eu.forecast.region_job import (
    # DwdIconEuForecastRegionJob,
    DwdIconEuForecastSourceFileCoord,
)


def test_source_file_coord_get_url() -> None:
    coord = DwdIconEuForecastSourceFileCoord(
        init_time=pd.Timestamp("2000-01-01T00:00"),
        lead_time=pd.Timedelta(0),
        grib_element="t_2m",
    )
    expected = "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/t_2m/icon-eu_europe_regular-lat-lon_single-level_2000010100_000_T_2M.grib2.bz2"

    assert coord.get_url() == expected


# def test_region_job_generete_source_file_coords() -> None:
#     template_config = DwdIconEuForecastTemplateConfig()
#     template_ds = template_config.get_template(pd.Timestamp("2000-01-23"))

#     region_job = DwdIconEuForecastRegionJob(
#         tmp_store=Mock(),
#         template_ds=template_ds,
#         data_vars=[Mock(), Mock()],
#         append_dim=template_config.append_dim,
#         region=slice(0, 10),
#         reformat_job_name="test",
#     )

#     processing_region_ds, output_region_ds = region_job._get_region_datasets()

#     source_file_coords = region_job.generate_source_file_coords(
#         processing_region_ds, [Mock()]
#     )

#     assert len(source_file_coords) == ...
#     assert ...
