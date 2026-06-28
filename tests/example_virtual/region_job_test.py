# from unittest.mock import Mock

# import pandas as pd

# from reformatters.example_virtual.region_job import (
#     ExampleRegionJob,
#     ExampleSourceFileCoord,
# )
# from reformatters.example_virtual.template_config import ExampleTemplateConfig


# def test_source_file_coord_url_uses_container_prefix() -> None:
#     coord = ExampleSourceFileCoord(
#         init_time=pd.Timestamp("2020-01-01"),
#         lead_time=pd.Timedelta("3h"),
#         data_vars=[Mock()],
#     )
#     # Refs point at get_url(), so it must start with the registered virtual chunk
#     # container prefix (s3://...), not an https:// mirror of the same bucket.
#     assert coord.get_url().startswith("s3://")
#     assert coord.get_index_url() == coord.get_url() + ".idx"


# def test_out_loc_excludes_non_dim_fields() -> None:
#     coord = ExampleSourceFileCoord(
#         init_time=pd.Timestamp("2020-01-01"),
#         lead_time=pd.Timedelta("3h"),
#         data_vars=[Mock()],
#     )
#     # data_vars is not an output dimension and must not leak into out_loc.
#     assert set(coord.out_loc()) == {"init_time", "lead_time"}


# def test_generate_source_file_coords() -> None:
#     template_config = ExampleTemplateConfig()
#     template_ds = template_config.get_template(pd.Timestamp("2020-01-02"))

#     region_job = ExampleRegionJob(
#         tmp_store=Mock(),
#         template_ds=template_ds,
#         data_vars=template_config.data_vars,
#         append_dim=template_config.append_dim,
#         region=slice(0, 1),
#         reformat_job_name="test",
#     )
#     processing_region_ds = region_job._processing_region_ds()
#     coords = region_job.generate_source_file_coords(
#         processing_region_ds, template_config.data_vars
#     )
#     assert len(coords) == ...


# def test_file_refs_resolves_message_byte_ranges() -> None:
#     # The core virtual seam: one VirtualRef per (output cell, variable), each carrying
#     # the source location and the message's byte range. A real test downloads a known
#     # .idx, calls file_refs, and asserts the offsets/lengths match the index.
#     # See tests/common/virtual_region_job_test.py for a runnable end-to-end example that
#     # round-trips values through a local-filesystem virtual chunk container.
#     ...
