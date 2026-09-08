from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import (
    ROOT,
    BaseInternalAttrs,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVar,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import AppendDim, Dim, Dims, Timedelta
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.ucsb_chc.chirps.chirps_config_models import ChirpsProduct

# CHIRPS v3.0 quasi-global 0.05 degree grid; pixel centers, rows north -> south.
GRID_LAT_SIZE = 2400
GRID_LON_SIZE = 7200
_LAT_NORTH = 59.975
_LAT_SOUTH = -59.975
_LON_WEST = -179.975
_LON_EAST = 179.975

# mm/day -> kg m-2 s-1 (= mm/s): 1 mm/day of water is 1 kg m-2 per 86400 s.
MM_PER_DAY_TO_KG_M2_S = 1.0 / 86400.0

# Marks the ocean and marginal seas, where CHIRPS makes no estimate. The source files
# set no GDAL nodata tag, so it arrives as a plain value.
SOURCE_FILL_VALUE = -9999.0

_DESCRIPTIONS: dict[ChirpsProduct, str] = {
    "final": (
        "Daily precipitation from the Climate Hazards Center Infrared Precipitation "
        "with Stations (CHIRPS) version 3.0 final product, which incorporates station "
        "observations and splits each pentad total into days using ERA5."
    ),
    "preliminary": (
        "Daily precipitation from the Climate Hazards Center Infrared Precipitation "
        "with Stations (CHIRPS) version 3.0 preliminary product, a lower latency "
        "satellite estimate which splits each pentad total into days using IMERG "
        "rather than ERA5, so it differs from the final product in its daily "
        "distribution as well as its inputs."
    ),
}

_SPATIAL_REF_WKT = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'


class UcsbChcChirpsAnalysisTemplateConfig(TemplateConfig[DataVar[BaseInternalAttrs]]):
    dims: Dims = {ROOT: ("time", "latitude", "longitude")}
    append_dim: AppendDim = "time"
    append_dim_frequency: Timedelta = pd.Timedelta("1D")

    product: ChirpsProduct

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            "time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "latitude": np.linspace(_LAT_NORTH, _LAT_SOUTH, GRID_LAT_SIZE),
            "longitude": np.linspace(_LON_WEST, _LON_EAST, GRID_LON_SIZE),
        }

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id=f"ucsb-chc-chirps-analysis-{self.product}",
            dataset_version="0.1.0",
            name=f"UCSB CHC CHIRPS analysis {self.product}",
            description=_DESCRIPTIONS[self.product],
            attribution=(
                "UCSB Climate Hazards Center CHIRPS version 3.0 data processed by "
                "dynamical.org from the Climate Hazards Center data archive. Cite "
                "Funk, C., Peterson, P., Harrison, L. et al. The Climate Hazards "
                "Center Infrared Precipitation with Stations, Version 3. Sci Data 13, "
                "718 (2026), data https://doi.org/10.15780/G2JQ0P."
            ),
            license="CC-BY-4.0",
            spatial_domain="Global land, 60 degrees north to 60 degrees south",
            spatial_resolution="0.05 degrees (~5km)",
            time_domain=f"{self.append_dim_start} UTC to Present",
            time_resolution="1 day",
        )

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        dim_coords = self.dimension_coordinates()
        return [
            Coordinate(
                name="time",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    calendar="proleptic_gregorian",
                    units="seconds since 1970-01-01 00:00:00",
                    chunks=self.append_dim_coordinate_chunk_size(),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Time",
                    standard_name="time",
                    axis="T",
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=dim_coords["time"].min().isoformat(), max="Present"
                    ),
                ),
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
                    chunks=(),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units=None,
                    statistics_approximate=None,
                    crs_wkt=_SPATIAL_REF_WKT,
                    semi_major_axis=6378137.0,
                    semi_minor_axis=6356752.314245179,
                    inverse_flattening=298.257223563,
                    reference_ellipsoid_name="WGS 84",
                    longitude_of_prime_meridian=0.0,
                    prime_meridian_name="Greenwich",
                    geographic_crs_name="WGS 84",
                    horizontal_datum_name="World Geodetic System 1984",
                    grid_mapping_name="latitude_longitude",
                    spatial_ref=_SPATIAL_REF_WKT,
                    comment="This coordinate reference system is WGS84 (EPSG:4326).",
                ),
            ),
        ]

    @computed_field
    @property
    def data_vars(self) -> Sequence[DataVar[BaseInternalAttrs]]:
        # Time-optimized: a year of days per chunk with a small 2.5 x 2.5 degree
        # spatial chunk, so a time series read pulls little wasted spatial data.
        # ~3.5 MB uncompressed, ~1 MB compressed over land at the measured 3.3:1 and
        # far less over the structurally empty ocean. Deliberately under the chunk
        # layout tool's 2.5 MB compressed target, which assumes data that compresses
        # less well.
        var_chunks: dict[Dim, int] = {
            "time": 365,
            "latitude": 50,  # 48 chunks over 2400
            "longitude": 50,  # 144 chunks over 7200
        }
        var_shards: dict[Dim, int] = {
            "time": var_chunks["time"],
            "latitude": var_chunks["latitude"] * 24,  # 2 shards over 2400
            "longitude": var_chunks["longitude"] * 36,  # 4 shards over 7200
        }
        return [
            DataVar(
                name="precipitation_surface",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=tuple(var_chunks[d] for d in self.dims[ROOT]),
                    shards=tuple(var_shards[d] for d in self.dims[ROOT]),
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                ),
                attrs=DataVarAttrs(
                    short_name="prate",
                    long_name="Precipitation rate",
                    standard_name="precipitation_flux",
                    units="kg m-2 s-1",
                    step_type="avg",
                    comment=(
                        "Average precipitation rate over the 24 hours starting at "
                        "the time coordinate. Units equivalent to mm/s. NaN where "
                        "CHIRPS makes no estimate, which is the ocean and marginal "
                        "seas; large lakes carry values."
                    ),
                ),
                internal_attrs=BaseInternalAttrs(
                    source_fill_value=SOURCE_FILL_VALUE,
                    keep_mantissa_bits=8,
                ),
            ),
        ]
