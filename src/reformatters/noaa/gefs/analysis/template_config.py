from collections.abc import Sequence
from typing import Any

import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import (
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE
from reformatters.noaa.gefs.common_gefs_template_config import (
    get_shared_coordinate_configs,
    get_shared_data_var_configs,
    get_shared_template_dimension_coordinates,
)
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar


class GefsAnalysisTemplateConfig(TemplateConfig[GEFSDataVar]):
    """Template configuration for GEFS analysis dataset."""

    dims: tuple[Dim, ...] = ("time", "latitude", "longitude")
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = pd.Timestamp("2000-01-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("3h")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        """Dataset metadata attributes."""
        return DatasetAttributes(
            dataset_id="noaa-gefs-analysis",
            dataset_version="0.1.2",
            name="NOAA GEFS analysis",
            description="Weather analysis from the Global Ensemble Forecast System (GEFS) operated by NOAA NWS NCEP.",
            attribution="NOAA NWS NCEP GEFS data processed by dynamical.org from NOAA Open Data Dissemination archives.",
            spatial_domain="Global",
            spatial_resolution="0.25 degrees (~20km)",
            time_domain=f"{self.append_dim_start} UTC to Present",
            time_resolution=f"{self.append_dim_frequency.total_seconds() / (60 * 60)} hours",
        )

    def append_dim_coordinate_chunk_size(self) -> int:
        """
        Returns a stable, fixed chunk size for the append dimension to allow
        expansion while making an effort to keep all coordinates in a single chunk.
        """
        # Use 50 years (instead of the default impl) to match the existing dataset
        # that existed before refactoring things to use TemplateConfig/etc.
        return int(pd.Timedelta(days=365 * 50) / self.append_dim_frequency)

    def dimension_coordinates(self) -> dict[str, Any]:
        """Returns dimension coordinates for the dataset."""
        return {
            "time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            **get_shared_template_dimension_coordinates(),
        }

    @computed_field  # type: ignore[prop-decorator]
    @property
    def coords(self) -> Sequence[Coordinate]:
        """Define metadata and encoding for each coordinate."""
        append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()

        return (
            *get_shared_coordinate_configs(),
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
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[GEFSDataVar]:
        """Define metadata and encoding for each data variable."""
        # CHUNKS
        var_chunks: dict[Dim, int] = {
            "time": 180 * (24 // 3),  # 180 days of 3 hourly data
            "latitude": 32,  # 23 chunks over 721 pixels
            "longitude": 32,  # 45 chunks over 1440 pixels
        }

        # SHARDS
        var_shards: dict[Dim, int] = {
            "time": var_chunks["time"] * 2,
            "latitude": var_chunks["latitude"] * 12,  # 2 shards over 721 pixels
            "longitude": var_chunks["longitude"] * 12,  # 4 shards over 1440 pixels
        }

        assert self.dims == tuple(var_chunks.keys())

        var_chunks_ordered = tuple(var_chunks[dim] for dim in self.dims)
        var_shards_ordered = tuple(var_shards[dim] for dim in self.dims)

        return get_shared_data_var_configs(var_chunks_ordered, var_shards_ordered)
