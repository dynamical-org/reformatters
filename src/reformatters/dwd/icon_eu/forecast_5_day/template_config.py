from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import computed_field

from reformatters.common.config_models import (
    ROOT,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.template_config import (
    SPATIAL_REF_COORDS,
    TemplateConfig,
)
from reformatters.common.types import AppendDim, Dim, Dims, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.dwd.icon_eu.icon_eu_template_config import (
    DwdIconEuDataVar,
    derive_forecast_coordinates,
    forecast_coordinate_configs,
    forecast_dimension_coordinates,
    icon_eu_data_vars,
)


class DwdIconEuForecast5DayTemplateConfig(TemplateConfig[DwdIconEuDataVar]):
    dims: Dims = {ROOT: ("init_time", "lead_time", "latitude", "longitude")}
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = pd.Timestamp("2026-02-10T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="dwd-icon-eu-forecast-5-day",
            dataset_version="0.2.0",
            name="DWD ICON-EU forecast, 5 day",
            description=(
                "High-resolution weather forecasts for Europe from the ICON-EU model operated by"
                " Deutscher Wetterdienst (DWD)."
            ),
            attribution="DWD ICON-EU data processed by dynamical.org.",
            license="CC-BY-4.0",
            spatial_domain="Europe",
            spatial_resolution="0.0625 degrees (~7km)",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:.0f} hours",
            forecast_domain="Forecast lead time 0-120 hours (0-5 days) ahead",
            forecast_resolution="Forecast step 0-78 hours: hourly, 81-120 hours: 3 hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        """Returns a dictionary of dimension names to coordinates for the dataset."""
        return {
            **forecast_dimension_coordinates(
                self.append_dim_coordinates(
                    self.append_dim_start + self.append_dim_frequency
                )
            ),
            "latitude": np.linspace(70.5, 29.5, 657),  # descending north->south
            "longitude": np.linspace(-23.5, 62.5, 1377),
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        """
        Return a dictionary of non-dimension coordinates for the dataset.

        Called whenever len(ds.append_dim) changes.
        """
        return {
            **derive_forecast_coordinates(ds),
            "spatial_ref": SPATIAL_REF_COORDS,
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        """Define metadata and encoding for each coordinate."""
        dim_coords = self.dimension_coordinates()
        append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()

        return [
            *forecast_coordinate_configs(
                dim_coords["init_time"], append_dim_coordinate_chunk_size
            ),
            Coordinate(
                name="latitude",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["latitude"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Latitude",
                    standard_name="latitude",
                    units="degree_north",
                    axis="Y",
                    statistics_approximate=StatisticsApproximate(
                        min=float(dim_coords["latitude"].min()),
                        max=float(dim_coords["latitude"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="longitude",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["longitude"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Longitude",
                    standard_name="longitude",
                    units="degree_east",
                    axis="X",
                    statistics_approximate=StatisticsApproximate(
                        min=float(dim_coords["longitude"].min()),
                        max=float(dim_coords["longitude"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="spatial_ref",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    chunks=(),  # Scalar coordinate
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units=None,
                    statistics_approximate=None,
                    # Derived by installing xarray, cfgrib, and rioxarray, and then running:
                    #     from pyproj import CRS
                    #     spherical_crs = CRS.from_wkt(WKT_STRING_EXTRACTED_FROM_ICON_EU_GRIB_BY_GDALINFO)
                    #     ds = xr.load_dataset(ICON_EU_GRIB_FILENAME_FROM_DWD, engine='cfgrib')
                    #     ds.rio.write_crs(spherical_crs)["spatial_ref"].attrs
                    crs_wkt='GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",SPHEROID["Sphere",6371229,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]',
                    semi_major_axis=6371229.0,
                    semi_minor_axis=6371229.0,
                    inverse_flattening=0.0,
                    reference_ellipsoid_name="Sphere",
                    longitude_of_prime_meridian=0.0,
                    prime_meridian_name="Greenwich",
                    geographic_crs_name="Coordinate System imported from GRIB file",
                    horizontal_datum_name="unnamed",
                    grid_mapping_name="latitude_longitude",
                    spatial_ref='GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",SPHEROID["Sphere",6371229,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]',
                    comment="A perfect sphere geographic CRS with a radius of 6,371,229m, extracted from grib.",
                ),
            ),
        ]

    @computed_field
    @property
    def data_vars(self) -> Sequence[DwdIconEuDataVar]:
        """Define metadata and encoding for each data variable."""
        # ~12MB uncompressed, ~2.4MB compressed
        var_chunks: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 93,  # All lead times
            "latitude": 219,  # All latitudes / 3
            "longitude": 153,  # All longitudes / 9
        }
        # ~321MB uncompressed, ~64MB compressed
        var_shards: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 93,  # All lead times
            "latitude": 657,  # All latitudes
            "longitude": 1377,  # All longitudes
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(var_chunks[d] for d in self.dims[ROOT]),
            shards=tuple(var_shards[d] for d in self.dims[ROOT]),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        return icon_eu_data_vars(encoding_float32_default)
