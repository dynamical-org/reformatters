# import numpy as np

# from reformatters.example_virtual.template_config import ExampleSpatialTemplateConfig


# def test_data_var_encoding_is_one_chunk_per_message() -> None:
#     # The defining property of a virtual dataset: each data var chunk is exactly one
#     # source message, so chunk size is 1 along the per-message dims and full-width along
#     # the spatial dims, with no shards and no compressors of our own - just a serializer.
#     config = ExampleSpatialTemplateConfig()
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


# def test_coords_match_codec_decoded_grid() -> None:
#     # Virtual chunks decode the raw message, so the grid must match the codec's decoded
#     # grid (not a regridded one): north-first latitude and, for a global grid,
#     # adjust_longitude_range's monotonic -180..+180 longitude.
#     config = ExampleSpatialTemplateConfig()
#     dim_coords = config.dimension_coordinates()
#     assert dim_coords["latitude"][0] > dim_coords["latitude"][-1]
#     assert dim_coords["longitude"].min() >= -180
#     assert dim_coords["longitude"].max() < 180
