from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import computed_field

from reformatters.common.config_models import (
    BaseInternalAttrs,
    Coordinate,
    CoordinateAttrs,
    DataVar,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.template_config import SPATIAL_REF_COORDS, TemplateConfig
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)

# IMERG is a global 0.1 degree grid: 1800 latitudes x 3600 longitudes (pixel centers).
LATITUDE_SIZE = 1800
LONGITUDE_SIZE = 3600


class NasaImergInternalAttrs(BaseInternalAttrs):
    """Variable specific attributes used internally to drive processing. Not written to the dataset."""

    # HDF5 subdataset path within the IMERG file, e.g. "//Grid/precipitation".
    h5_path: str
    # Value the source uses for missing data; converted to NaN on read.
    source_fill_value: float
    # Multiplier applied on read to convert source units to the dataset units
    # (e.g. mm/hr -> kg m-2 s-1 is 1/3600).
    units_scale_factor: float = 1.0


class NasaImergDataVar(DataVar[NasaImergInternalAttrs]):
    pass


class NasaImergAnalysisTemplateConfig(TemplateConfig[NasaImergDataVar]):
    """Shared template configuration for the IMERG Early and Late analysis datasets.

    Early and Late differ only in their source product (URL) and update latency, so
    structure, coordinates, variables and encoding are all defined here. Subclasses
    only implement ``dataset_attributes``.
    """

    dims: tuple[Dim, ...] = ("time", "latitude", "longitude")
    append_dim: AppendDim = "time"
    # IMERG V07 half-hourly Early/Late record begins 1998-01-01 (verified on GES DISC;
    # V07 reprocessing extends back through the full TRMM era).
    append_dim_start: Timestamp = pd.Timestamp("1998-01-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("30min")

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            "time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            # Pixel centers. Latitude descending (north -> south) to match the
            # convention used by our other gridded datasets.
            "latitude": 89.95 - 0.1 * np.arange(LATITUDE_SIZE),
            "longitude": -179.95 + 0.1 * np.arange(LONGITUDE_SIZE),
        }

    def derive_coordinates(
        self,
        ds: xr.Dataset,  # noqa: ARG002
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        return {"spatial_ref": SPATIAL_REF_COORDS}

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        dim_coords = self.dimension_coordinates()
        append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()

        return [
            Coordinate(
                name="time",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    calendar="proleptic_gregorian",
                    units="seconds since 1970-01-01 00:00:00",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Time",
                    standard_name="time",
                    axis="T",
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=self.append_dim_start.isoformat(), max="Present"
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
                    # IMERG is on a regular WGS84 geographic grid (EPSG:4326).
                    # Derived via ds.rio.write_crs("EPSG:4326")["spatial_ref"].attrs
                    crs_wkt='GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]',
                    semi_major_axis=6378137.0,
                    semi_minor_axis=6356752.314245179,
                    inverse_flattening=298.257223563,
                    reference_ellipsoid_name="WGS 84",
                    longitude_of_prime_meridian=0.0,
                    prime_meridian_name="Greenwich",
                    geographic_crs_name="WGS 84",
                    horizontal_datum_name="World Geodetic System 1984",
                    grid_mapping_name="latitude_longitude",
                    spatial_ref='GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]',
                ),
            ),
        ]

    @computed_field
    @property
    def data_vars(self) -> Sequence[NasaImergDataVar]:
        # 30-minute global data makes a long time series per shard expensive: a worker
        # holds one full-spatial time-shard of a single variable in memory
        # (time_shard x 1800 x 3600 x 4 bytes). This buffer drives the K8s
        # shared_memory/memory requests in the dynamical_dataset cron jobs, and an
        # operational update reprocesses the whole current time shard, so a larger
        # time_shard also means more re-downloads per update. We trade all that against
        # read locality (fewer shards per long time series).
        #
        # Chosen: 30 days/shard. ~11MB raw / ~2.2MB compressed chunk,
        # ~1.1GB raw / ~220MB compressed shard, ~37GB worker buffer.
        # Alternatives to iterate on: time=720 (15 days, ~19GB buffer, smaller
        # chunks/cheaper updates) or time=2880 (60 days, ~74GB buffer, larger
        # chunks/better time-series locality).
        var_chunks: dict[Dim, int] = {
            "time": 1440,  # 30 days of 30-minute data (317 chunks over the record)
            "latitude": 45,  # 40 chunks over 1800 pixels
            "longitude": 45,  # 80 chunks over 3600 pixels
        }
        var_shards: dict[Dim, int] = {
            "time": var_chunks["time"],  # one time chunk per shard
            "latitude": var_chunks["latitude"] * 10,  # 4 shards over 1800 pixels
            "longitude": var_chunks["longitude"] * 10,  # 8 shards over 3600 pixels
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(var_chunks[d] for d in self.dims),
            shards=tuple(var_shards[d] for d in self.dims),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        default_keep_mantissa_bits = 7

        return [
            NasaImergDataVar(
                name="precipitation_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="prate",
                    standard_name="precipitation_flux",
                    long_name="Precipitation rate",
                    units="kg m-2 s-1",
                    step_type="avg",
                    comment="Multi-satellite precipitation estimate (gauge-adjusted via climatological calibration) representing the mean rate over the half-hour window. Units equivalent to mm/s; multiply by 3600 for mm/hr.",
                ),
                internal_attrs=NasaImergInternalAttrs(
                    h5_path="//Grid/precipitation",
                    source_fill_value=-9999.9,
                    units_scale_factor=1 / 3600,  # mm/hr -> kg m-2 s-1
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            NasaImergDataVar(
                name="probability_of_liquid_precipitation_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="plp",
                    long_name="Probability of liquid precipitation phase",
                    units="percent",
                    step_type="instant",
                    comment="Probability that the precipitation is in liquid phase (versus frozen). 0 = certainly frozen, 100 = certainly liquid.",
                ),
                internal_attrs=NasaImergInternalAttrs(
                    h5_path="//Grid/probabilityLiquidPrecipitation",
                    source_fill_value=-9999,  # int16 fill
                    keep_mantissa_bits="no-rounding",  # integer percent, keep exact
                ),
            ),
            NasaImergDataVar(
                name="precipitation_quality_index_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="pqi",
                    long_name="Precipitation quality index",
                    units="1",
                    step_type="instant",
                    comment="Quality index for the precipitation estimate, ranging 0 (lowest) to 1 (highest).",
                ),
                internal_attrs=NasaImergInternalAttrs(
                    h5_path="//Grid/precipitationQualityIndex",
                    source_fill_value=-9999.9,
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
        ]
