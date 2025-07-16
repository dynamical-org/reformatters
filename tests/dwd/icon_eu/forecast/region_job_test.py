# from unittest.mock import Mock

# import pandas as pd

# from reformatters.dwd.icon_eu.forecast.region_job import (
#     DwdIconEuForecastRegionJob,
#     DwdIconEuForecastSourceFileCoord,
# )
# from reformatters.dwd.icon_eu.forecast.template_config import DwdIconEuForecastTemplateConfig

# def test_source_file_coord_get_url() -> None:
#     coord = DwdIconEuForecastSourceFileCoord(time=pd.Timestamp("2000-01-01"))
#     assert coord.get_url() == "https://example.com/data/2000-01-01.grib2"


# def test_region_job_generete_source_file_coords() -> None:
#     template_config = DwdIconEuForecastTemplateConfig()
#     template_ds = template_config.get_template(pd.Timestamp("2000-01-23"))

#     region_job = DwdIconEuForecastRegionJob(
#         final_store=Mock(),
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
