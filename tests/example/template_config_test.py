# Example of spatial validation pattern using a downloaded GRIB slice
# (kept commented to guide new integrations)
# import numpy as np
# import pandas as pd
# import pytest
# import rasterio  # type: ignore[import-untyped]
# import xarray as xr
# from pathlib import Path
# from unittest.mock import Mock
#
# from reformatters.example.region_job import ExampleRegionJob, ExampleSourceFileCoord
# from reformatters.example.template_config import ExampleTemplateConfig
#
#
# @pytest.fixture(scope="session")
# def example_first_message_path() -> Path:
#     cfg = ExampleTemplateConfig()
#     coord = ExampleSourceFileCoord(...)  # fill in per dataset
#     region_job = ExampleRegionJob.model_construct(
#         tmp_store=Mock(),
#         template_ds=cfg.get_template(cfg.append_dim_start),
#         data_vars=cfg.data_vars,
#         append_dim=cfg.append_dim,
#         region=slice(0, 1),
#         reformat_job_name="test",
#     )
#     return region_job.download_file(coord)
#
#
# @pytest.mark.slow
# def test_spatial_ref_matches_grib(example_first_message_path: Path) -> None:
#     cfg = ExampleTemplateConfig()
#     ds = cfg.get_template(cfg.append_dim_start)
#     ds_raster = xr.open_dataset(example_first_message_path, engine="rasterio")
#
#     assert ds.rio.shape == ds_raster.rio.shape
#     assert np.allclose(ds.rio.bounds(), ds_raster.rio.bounds())
#     assert ds.rio.resolution() == ds_raster.rio.resolution()
#     assert ds.rio.crs.to_proj4() == ds_raster.rio.crs.to_proj4()
#
#
# @pytest.mark.slow
# def test_lat_lon_pixel_centers_from_source_grib(
#     example_first_message_path: Path,
# ) -> None:
#     cfg = ExampleTemplateConfig()
#     coords = cfg.dimension_coordinates()
#
#     with rasterio.open(example_first_message_path) as reader:
#         bounds = reader.bounds
#         pixel_size_x = reader.transform.a
#         pixel_size_y = abs(reader.transform.e)
#
#     lon = coords["longitude"]
#     lat = coords["latitude"]
#
#     atol = 1e-6
#     rtol = 0.0
#     assert np.isclose(bounds.left + pixel_size_x / 2, lon.min(), atol=atol, rtol=rtol)
#     assert np.isclose(bounds.right - pixel_size_x / 2, lon.max(), atol=atol, rtol=rtol)
#     assert np.isclose(bounds.top - pixel_size_y / 2, lat.max(), atol=atol, rtol=rtol)
#     assert np.isclose(bounds.bottom + pixel_size_y / 2, lat.min(), atol=atol, rtol=rtol)
