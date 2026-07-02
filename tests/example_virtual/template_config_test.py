# import numpy as np

# from reformatters.example_virtual.template_config import ExampleTemplateConfig


# def test_data_var_encoding_is_one_chunk_per_message() -> None:
#     # The defining property of a virtual dataset: each data var chunk is exactly one
#     # source message, so chunk size is 1 along the per-message dims and full-width along
#     # the spatial dims, with no shards and no compressors of our own - just a serializer.
#     config = ExampleTemplateConfig()
#     dim_coords = config.dimension_coordinates()
#     var = next(v for v in config.data_vars if v.name == "temperature_2m")

#     assert var.encoding.chunks == (
#         1,  # init_time
#         1,  # lead_time
#         len(dim_coords["latitude"]),
#         len(dim_coords["longitude"]),
#     )
#     assert var.encoding.shards is None
#     assert var.encoding.compressors == ()
#     assert var.encoding.serializer is not None  # e.g. GribberishCodec(...).to_dict()


# def test_coords_match_native_source_grid() -> None:
#     # Virtual chunks decode the raw message, so the grid must be the source's native
#     # grid (here GEFS-style: latitude descending, longitude 0-360), not a regridded one.
#     config = ExampleTemplateConfig()
#     dim_coords = config.dimension_coordinates()
#     assert dim_coords["latitude"][0] > dim_coords["latitude"][-1]
#     assert dim_coords["longitude"].min() >= 0
