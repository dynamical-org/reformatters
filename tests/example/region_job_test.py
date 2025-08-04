# from unittest.mock import Mock

# import pandas as pd

# from reformatters.example.region_job import (
#     ExampleRegionJob,
#     ExampleSourceFileCoord,
# )
# from reformatters.example.template_config import ExampleTemplateConfig

# def test_source_file_coord_get_url() -> None:
#     coord = ExampleSourceFileCoord(time=pd.Timestamp("2000-01-01"))
#     assert coord.get_url() == "https://example.com/data/2000-01-01.grib2"


# def test_region_job_generete_source_file_coords() -> None:
#     template_config = ExampleTemplateConfig()
#     template_ds = template_config.get_template(pd.Timestamp("2000-01-23"))

#     region_job = ExampleRegionJob(
#         primary_store_factory=Mock(),
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
