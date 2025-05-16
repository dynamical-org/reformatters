from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import (
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVar,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.template_config import (
    TemplateConfig,
)
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp


class SWANNDataVar(DataVar[Any]):
    pass


class SWANNTemplateConfig(TemplateConfig[SWANNDataVar]):
    dims: tuple[Dim, ...] = ("time", "latitude", "longitude")
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = pd.Timestamp("1981-10-01")
    append_dim_frequency: Timedelta = pd.Timedelta("1D")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="nsidc-0719-snow",
            dataset_version="1.0.0",
            name="NSIDC Daily 4 km Gridded SWE and Snow Depth over CONUS",
            description="Daily 4 km gridded snow water equivalent (SWE) and snow depth from assimilated in-situ and modeled data over the conterminous US.",
            # TODO: Attribution is close but not quite right. See NSIDC docs.
            attribution="Broxton, P., X. Zeng, and N. Dawson. 2019. Daily 4 km Gridded SWE and Snow Depth from Assimilated In-Situ and Modeled Data over the Conterminous US, Version 1. NASA NSIDC.",
            spatial_domain="Conterminous US",
            spatial_resolution="4 km",
            time_domain="1981-10-01 to 2023-09-30",
            time_resolution="Daily",
            forecast_domain=None,
            forecast_resolution=None,
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        # latitude: 50.0 to 24.0 N, longitude: -125.0 to -66.5 W
        # 4 km grid, so 0.036 degrees per pixel (approx), but user guide says 4 km, so let's use np.arange
        latitude = np.arange(50.0, 24.0 - 0.001, -0.036)  # decreasing order
        longitude = np.arange(-125.0, -66.5 + 0.001, 0.036)  # increasing order
        return {
            self.append_dim: self.append_dim_coordinates(
                self.append_dim_start + pd.Timedelta(days=1)
            ),
            "latitude": latitude,
            "longitude": longitude,
        }

    @computed_field  # type: ignore[prop-decorator]
    @property
    def coords(self) -> Sequence[Coordinate]:
        dim_coords = self.dimension_coordinates()
        append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()
        return [
            Coordinate(
                name=self.append_dim,
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    compressors=[],
                    calendar="proleptic_gregorian",
                    units="seconds since 1970-01-01 00:00:00",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=pd.Timestamp(self.append_dim_start).isoformat(),
                        max=pd.Timestamp(
                            self.append_dim_start
                            + pd.Timedelta(days=len(dim_coords[self.append_dim]) - 1)
                        ).isoformat(),
                    ),
                ),
            ),
            Coordinate(
                name="latitude",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[],
                    chunks=len(dim_coords["latitude"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units="degrees_north",
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
                    compressors=[],
                    chunks=len(dim_coords["longitude"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units="degrees_east",
                    statistics_approximate=StatisticsApproximate(
                        min=float(dim_coords["longitude"].min()),
                        max=float(dim_coords["longitude"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="spatial_ref",  # TODO: This spatial reference is coming from the docs
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    chunks=(),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units=None,
                    statistics_approximate=None,
                    crs_wkt='GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4269"]]',
                    semi_major_axis=6378137.0,
                    semi_minor_axis=6356752.314140356,
                    inverse_flattening=298.257222101,
                    reference_ellipsoid_name="GRS 1980",
                    longitude_of_prime_meridian=0.0,
                    prime_meridian_name="Greenwich",
                    geographic_crs_name="NAD83",
                    horizontal_datum_name="North American Datum 1983",
                    grid_mapping_name="latitude_longitude",
                    spatial_ref='GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4269"]]',
                    # TODO: Add comment field?
                ),
            ),
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[SWANNDataVar]:
        # TODO:
        #  - Same style as GFSForecast?
        #  - Not sure this chunking is correct?
        #  - We need shards for the data variables
        #  - Need compressor(s)
        #  - Lowercase variable names?
        lat_len = len(self.dimension_coordinates()["latitude"])
        lon_len = len(self.dimension_coordinates()["longitude"])
        return [
            SWANNDataVar(
                name="SWE",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=(1, lat_len, lon_len),
                    shards=None,
                    compressors=[],
                ),
                attrs=DataVarAttrs(
                    short_name="SWE",
                    long_name="Snow water equivalent",
                    standard_name="lwe_thickness_of_snow_amount",
                    units="mm",  # TODO: Do we need to say H2O?
                    step_type="instant",
                ),
                internal_attrs=None,
            ),
            SWANNDataVar(
                name="DEPTH",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=(1, lat_len, lon_len),
                    shards=None,
                    compressors=[],
                ),
                attrs=DataVarAttrs(
                    short_name="DEPTH",
                    long_name="Snow depth",
                    standard_name="surface_snow_thickness",
                    units="mm",
                    step_type="instant",
                ),
                internal_attrs=None,
            ),
        ]
