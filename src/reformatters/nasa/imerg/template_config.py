from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import (
    ROOT,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVarAttrs,
    Encoding,
    Group,
    StatisticsApproximate,
)
from reformatters.common.template_config import (
    TemplateConfig,
)
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.nasa.imerg.imerg_config_models import (
    ImergRun,
    NasaImergDataVar,
    NasaImergInternalAttrs,
)

# IMERG global 0.1 degree WGS84 grid; pixel centers. Latitude is emitted descending
# (north-up) to match the repo convention, though the source file stores it ascending.
GRID_LAT_SIZE = 1800
GRID_LON_SIZE = 3600
_LAT_NORTH = 89.95
_LAT_SOUTH = -89.95
_LON_WEST = -179.95
_LON_EAST = 179.95

# mm/hr -> kg m-2 s-1 (= mm/s): 1 mm/hr of water is 1 kg m-2 per 3600 s.
MM_PER_HR_TO_KG_M2_S = 1.0 / 3600.0

# Sentinel the source stores for missing pixels, shared by all IMERG fields.
SOURCE_FILL_VALUE = -9999.9

# Earliest available Early/Late V07 granule (CMR-verified; V07 reprocessed the full
# TRMM era back to 1998, earlier than the pre-V07 2000-06 start).
APPEND_DIM_START = pd.Timestamp("1998-01-01T00:00")

# Standard WGS84 (EPSG:4326) spatial reference, derived via rioxarray.
_SPATIAL_REF_WKT = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'


class NasaImergAnalysisTemplateConfig(TemplateConfig[NasaImergDataVar]):
    """Shared structure for the NASA IMERG half-hourly analysis datasets.

    Concrete subclasses set `run` (early/late); everything else is shared.
    """

    dims: dict[Group, tuple[Dim, ...]] = {ROOT: ("time", "latitude", "longitude")}
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = APPEND_DIM_START
    append_dim_frequency: Timedelta = pd.Timedelta("30min")

    run: ImergRun

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
        run_name = self.run.capitalize()
        return DatasetAttributes(
            dataset_id=f"nasa-imerg-analysis-{self.run}",
            dataset_version="0.1.0",
            name=f"NASA IMERG analysis, {self.run}",
            description=(
                f"Global half-hourly precipitation from NASA GPM IMERG {run_name} Run, "
                "version 07."
            ),
            attribution="NASA GPM IMERG data processed by dynamical.org from NASA GES DISC and PPS archives.",
            license="CC-BY-4.0",
            spatial_domain="Global",
            spatial_resolution="0.1 degrees (~10km)",
            time_domain=f"{self.append_dim_start} UTC to Present",
            time_resolution="30 minutes",
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
    def data_vars(self) -> Sequence[NasaImergDataVar]:
        # Time-optimized: a long time chunk with a small spatial chunk, so a
        # time-series point read pulls little wasted spatial data. Sized to the
        # measured ~23:1 compression of precipitation (the less-compressible of the
        # two variables) -> chunk ~2.4 MB, shard ~200 MB compressed.
        var_chunks: dict[Dim, int] = {
            "time": 1440,  # 30 days of 30-minute data
            "latitude": 100,  # 18 chunks over 1800
            "longitude": 100,  # 36 chunks over 3600
        }
        var_shards: dict[Dim, int] = {
            "time": var_chunks["time"],
            "latitude": var_chunks["latitude"] * 9,  # 2 shards over 1800
            "longitude": var_chunks["longitude"] * 9,  # 4 shards over 3600
        }
        chunks = tuple(var_chunks[d] for d in self.dims[ROOT])
        shards = tuple(var_shards[d] for d in self.dims[ROOT])
        encoding_float32 = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=chunks,
            shards=shards,
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )
        return [
            NasaImergDataVar(
                name="precipitation_surface",
                encoding=encoding_float32,
                attrs=DataVarAttrs(
                    short_name="prate",
                    long_name="Precipitation rate",
                    standard_name="precipitation_flux",
                    units="kg m-2 s-1",
                    step_type="avg",
                    comment=(
                        "Mean precipitation rate over the half hour. Complete "
                        "merged microwave-infrared, gauge-adjusted where available "
                        "(formerly precipitationCal). Units equivalent to mm/s; "
                        "source is mm/hr."
                    ),
                ),
                internal_attrs=NasaImergInternalAttrs(
                    h5_path="//Grid/precipitation",
                    source_fill_value=SOURCE_FILL_VALUE,
                    source_scale=MM_PER_HR_TO_KG_M2_S,
                    keep_mantissa_bits=7,
                ),
            ),
            NasaImergDataVar(
                name="precipitation_quality_index_surface",
                encoding=encoding_float32,
                attrs=DataVarAttrs(
                    short_name="pqi",
                    long_name="Quality index for precipitation",
                    units="1",
                    step_type="avg",
                    comment="Dimensionless quality index for the precipitation estimate; higher is better.",
                ),
                internal_attrs=NasaImergInternalAttrs(
                    h5_path="//Grid/precipitationQualityIndex",
                    source_fill_value=SOURCE_FILL_VALUE,
                    keep_mantissa_bits=7,
                ),
            ),
        ]
