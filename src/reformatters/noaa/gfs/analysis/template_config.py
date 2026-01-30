from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import computed_field

from reformatters.common.config_models import (
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.template_config import SPATIAL_REF_COORDS
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.noaa.gfs.template_config import NoaaGfsCommonTemplateConfig
from reformatters.noaa.models import NoaaDataVar


class NoaaGfsAnalysisTemplateConfig(NoaaGfsCommonTemplateConfig):
    dims: tuple[Dim, ...] = ("time", "latitude", "longitude")
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = pd.Timestamp("2021-05-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("1h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="noaa-gfs-analysis",
            dataset_version="0.1.0",
            name="NOAA GFS analysis",
            description="Weather analysis from the Global Forecast System (GFS) operated by NOAA NWS NCEP.",
            attribution="NOAA NWS NCEP GFS data processed by dynamical.org from NOAA Open Data Dissemination archives.",
            spatial_domain="Global",
            spatial_resolution="0.25 degrees (~20km)",
            time_domain=f"{self.append_dim_start} UTC to Present",
            time_resolution=f"{whole_hours(self.append_dim_frequency)} hour",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            "time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            **self._latitude_longitude_coordinates(),
        }

    def derive_coordinates(
        self, _ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        return {
            "spatial_ref": SPATIAL_REF_COORDS,
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()
        common_coords = super().coords

        return [
            *common_coords,
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
        ]

    @computed_field
    @property
    def data_vars(self) -> Sequence[NoaaDataVar]:
        # ~16MB uncompressed, ~3.1MB compressed
        var_chunks: dict[Dim, int] = {
            "time": 1008,  # 42 days of 1-hourly data
            "latitude": 64,  # 12 chunks over 721 pixels
            "longitude": 64,  # 23 chunks over 1440 pixels
        }

        # ~1688MB uncompressed, ~338MB compressed
        var_shards: dict[Dim, int] = {
            "time": var_chunks["time"] * 3,  # 126 days per shard
            "latitude": var_chunks["latitude"] * 6,  # 2 shards over 721 pixels
            "longitude": var_chunks["longitude"] * 6,  # 4 shards over 1440 pixels
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(var_chunks[d] for d in self.dims),
            shards=tuple(var_shards[d] for d in self.dims),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        return self.get_data_vars(encoding_float32_default)
