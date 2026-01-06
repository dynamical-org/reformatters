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
from reformatters.noaa.hrrr.hrrr_config_models import (
    NoaaHrrrDataVar,
)
from reformatters.noaa.hrrr.template_config import NoaaHrrrCommonTemplateConfig


class NoaaHrrrAnalysisTemplateConfig(NoaaHrrrCommonTemplateConfig):
    dims: tuple[Dim, ...] = ("time", "y", "x")
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = pd.Timestamp("2018-07-14T00:00")  # start of HRRR v3
    append_dim_frequency: Timedelta = pd.Timedelta("1h")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="noaa-hrrr-analysis",
            dataset_version="0.1.0",
            name="NOAA HRRR analysis",
            description="Analysis data from the High Resolution Rapid Refresh (HRRR) model operated by NOAA NWS NCEP.",
            attribution="NOAA NWS NCEP HRRR data processed by dynamical.org from NOAA Open Data Dissemination archives.",
            spatial_domain="Continental United States",
            spatial_resolution="3km",
            time_domain=f"{self.append_dim_start} UTC to Present",
            time_resolution=f"{whole_hours(self.append_dim_frequency)} hour",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        y_coords, x_coords = self._y_x_coordinates()
        return {
            "time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "y": y_coords,
            "x": x_coords,
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        latitudes, longitudes = self._latitude_longitude_coordinates(
            ds["x"].values, ds["y"].values
        )

        return {
            "latitude": (("y", "x"), latitudes),
            "longitude": (("y", "x"), longitudes),
            "spatial_ref": SPATIAL_REF_COORDS,
        }

    @computed_field  # type: ignore[prop-decorator]
    @property
    def coords(self) -> Sequence[Coordinate]:
        append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()

        hrrr_common_coords = super().coords

        return [
            *hrrr_common_coords,
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
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=self.append_dim_start.isoformat(), max="Present"
                    ),
                ),
            ),
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[NoaaHrrrDataVar]:
        # ~18MB uncompressed, ~3.5MB compressed
        var_chunks: dict[Dim, int] = {
            "time": 90 * 24,  # 90 days of hourly data
            # Chunks are 135 km by 135 km spatially
            "x": 45,  # 40 chunks over 1799 pixels
            "y": 45,  # 24 chunks over 1059 pixels
        }

        # ~2GB uncompressed, ~400MB compressed
        var_shards: dict[Dim, int] = {
            "time": var_chunks["time"],
            # Shards are 1350 km by 1620 km spatially
            "x": var_chunks["x"] * 10,  # 4 shards over 1799 pixels
            "y": var_chunks["y"] * 12,  # 2 shards over 1059 pixels
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(var_chunks[d] for d in self.dims),
            shards=tuple(var_shards[d] for d in self.dims),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        return self.get_data_vars(encoding_float32_default)
