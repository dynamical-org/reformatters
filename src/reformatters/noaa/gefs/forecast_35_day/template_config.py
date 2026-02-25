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
from reformatters.common.template_config import SPATIAL_REF_COORDS, TemplateConfig
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE
from reformatters.noaa.gefs.common_gefs_template_config import (
    get_shared_coordinate_configs,
    get_shared_data_var_configs,
    get_shared_template_dimension_coordinates,
)
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar


class GefsForecast35DayTemplateConfig(TemplateConfig[GEFSDataVar]):
    """Template configuration for GEFS 35-day forecast dataset."""

    dims: tuple[Dim, ...] = (
        "init_time",
        "ensemble_member",
        "lead_time",
        "latitude",
        "longitude",
    )
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = pd.Timestamp("2020-10-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("24h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        """Dataset metadata attributes."""
        return DatasetAttributes(
            dataset_id="noaa-gefs-forecast-35-day",
            dataset_version="dev-quick",
            name="NOAA GEFS forecast, 35 day",
            description="Weather forecasts from the Global Ensemble Forecast System (GEFS) operated by NOAA NWS NCEP.",
            attribution="NOAA NWS NCEP GEFS data processed by dynamical.org from NOAA Open Data Dissemination archives.",
            spatial_domain="Global",
            spatial_resolution="0-240 hours: 0.25 degrees (~20km), 243-840 hours: 0.5 degrees (~40km)",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution="Forecasts initialized every 24 hours",
            forecast_domain="Forecast lead time 0-840 hours (0-35 days) ahead",
            forecast_resolution="Forecast step 0-240 hours: 3 hourly, 243-840 hours: 6 hourly",
        )

    def append_dim_coordinate_chunk_size(self) -> int:
        """
        Returns a stable, fixed chunk size for the append dimension to allow
        expansion while making an effort to keep all coordinates in a single chunk.
        """
        # The init time dimension is our append dimension during updates.
        # We also want coordinates to be in a single chunk for dataset open speed.
        # By fixing the chunk size for coordinates along the append dimension to
        # something much larger than we will really use, the array is always
        # a fixed underlying chunk size and values in it can be safely updated
        # prior to metadata document updates that increase the reported array size.
        # This is a zarr format hack to allow expanding an array safely and requires
        # that new array values are written strictly before new metadata is written
        # (doing this correctly is a key benefit of icechunk).
        result: float = pd.Timedelta(days=365 * 15) / self.append_dim_frequency  # type: ignore[assignment]
        return int(result)

    def dimension_coordinates(self) -> dict[str, Any]:
        """Returns dimension coordinates for the dataset."""
        return {
            "init_time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "ensemble_member": np.arange(31),
            "lead_time": pd.timedelta_range("0h", "240h", freq="3h").union(
                pd.timedelta_range("246h", "840h", freq="6h")
            ),
            **get_shared_template_dimension_coordinates(),
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        """Return non-dimension coordinates for the dataset."""
        # Expected forecast length by init hour (00 UTC: 35 days, others: 16 days)
        expected_forecast_length_by_init_hour = pd.Series(
            {
                0: pd.Timedelta(hours=840),
                6: pd.Timedelta(hours=384),
                12: pd.Timedelta(hours=384),
                18: pd.Timedelta(hours=384),
            }
        )

        return {
            "valid_time": ds["init_time"] + ds["lead_time"],
            "ingested_forecast_length": (
                ("init_time", "ensemble_member"),
                np.full(
                    (ds["init_time"].size, ds["ensemble_member"].size),
                    np.timedelta64("NaT", "ns"),
                ),
            ),
            "expected_forecast_length": (
                ("init_time",),
                np.array(
                    [
                        expected_forecast_length_by_init_hour.get(
                            t.hour, pd.Timedelta(hours=384)
                        )
                        for t in pd.to_datetime(ds["init_time"].values)
                    ]
                ),
            ),
            "spatial_ref": SPATIAL_REF_COORDS,
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        """Define metadata and encoding for each coordinate."""
        dim_coords = self.dimension_coordinates()
        append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()

        return (
            *get_shared_coordinate_configs(),
            Coordinate(
                name="init_time",
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
                    long_name="Forecast initialization time",
                    standard_name="forecast_reference_time",
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=self.append_dim_start.isoformat(), max="Present"
                    ),
                ),
            ),
            Coordinate(
                name="ensemble_member",
                encoding=Encoding(
                    dtype="int16",
                    fill_value=-1,
                    chunks=len(dim_coords["ensemble_member"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Ensemble member",
                    standard_name="realization",
                    units="1",
                    statistics_approximate=StatisticsApproximate(
                        min=int(dim_coords["ensemble_member"].min()),
                        max=int(dim_coords["ensemble_member"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="lead_time",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=-1,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=len(dim_coords["lead_time"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Forecast lead time",
                    standard_name="forecast_period",
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].min()),
                        max=str(dim_coords["lead_time"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="valid_time",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    calendar="proleptic_gregorian",
                    units="seconds since 1970-01-01 00:00:00",
                    chunks=(
                        append_dim_coordinate_chunk_size,
                        len(dim_coords["lead_time"]),
                    ),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Valid time",
                    standard_name="time",
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=self.append_dim_start.isoformat(), max="Present + 35 days"
                    ),
                ),
            ),
            Coordinate(
                name="ingested_forecast_length",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=-1,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=(
                        append_dim_coordinate_chunk_size,
                        len(dim_coords["ensemble_member"]),
                    ),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Ingested forecast length",
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].min()),
                        max=str(dim_coords["lead_time"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="expected_forecast_length",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=-1,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Expected forecast length",
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].min()),
                        max=str(dim_coords["lead_time"].max()),
                    ),
                ),
            ),
        )

    @computed_field
    @property
    def data_vars(self) -> Sequence[GEFSDataVar]:
        """Define metadata and encoding for each data variable."""
        # CHUNKS - These chunks are about 2mb of uncompressed float32s
        var_chunks: dict[Dim, int] = {
            "init_time": 1,  # one forecast per chunk
            "ensemble_member": 31,  # all ensemble members in one chunk
            "lead_time": 64,  # 3 chunks, first chunk includes days 0-7, second days 8-mid day 21, third days 21-35
            "latitude": 17,  # 43 chunks over 721 pixels
            "longitude": 16,  # 90 chunks over 1440 pixels
        }

        # SHARDS - About 300-550MB compressed, about 3GB uncompressed
        var_shards: dict[Dim, int] = {
            "init_time": 1,  # one forecast per shard
            "ensemble_member": 31,  # all ensemble members in one shard
            "lead_time": var_chunks["lead_time"] * 3,  # all lead times in one shard
            "latitude": var_chunks["latitude"] * 22,  # 2 shards over 721 pixels
            "longitude": var_chunks["longitude"] * 23,  # 4 shards over 1440 pixels
        }

        assert self.dims == tuple(var_chunks.keys())
        var_chunks_ordered = tuple(var_chunks[dim] for dim in self.dims)
        var_shards_ordered = tuple(var_shards[dim] for dim in self.dims)

        return get_shared_data_var_configs(var_chunks_ordered, var_shards_ordered)
