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
    Group,
    StatisticsApproximate,
)
from reformatters.common.template_config import SPATIAL_REF_COORDS
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.eccc.hrdps.hrdps_config_models import EcccHrdpsDataVar
from reformatters.eccc.hrdps.template_config import EcccHrdpsCommonTemplateConfig


class EcccHrdpsAnalysisTemplateConfig(EcccHrdpsCommonTemplateConfig):
    dims: dict[Group, tuple[Dim, ...]] = {ROOT: ("time", "y", "x")}
    append_dim: AppendDim = "time"
    # Start of dynamical.org's continuous grib archive of HRDPS on Source Co-Op
    append_dim_start: Timestamp = pd.Timestamp("2026-07-09T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("1h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="eccc-hrdps-analysis",
            dataset_version="0.1.0",
            name="ECCC HRDPS analysis",
            description="Analysis data from the High Resolution Deterministic "
            "Prediction System (HRDPS) continental domain, operated by Environment "
            "and Climate Change Canada (ECCC).",
            attribution="Data Source: Environment and Climate Change Canada. "
            "Processed by dynamical.org.",
            license="ECCC Data Servers End-use Licence v2.1",
            spatial_domain="Canada and the northern continental United States",
            spatial_resolution="2.5 km",
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

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()

        return [
            *super().coords,
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
    def data_vars(self) -> Sequence[EcccHrdpsDataVar]:
        # ~11MB uncompressed, ~2.1MB compressed
        var_chunks: dict[Dim, int] = {
            "time": 30 * 24,  # 30 days of hourly data
            "y": 60,  # 22 chunks over 1290 pixels
            "x": 64,  # 40 chunks over 2540 pixels
        }

        # ~1160MB uncompressed, ~232MB compressed
        var_shards: dict[Dim, int] = {
            "time": var_chunks["time"],
            "y": var_chunks["y"] * 11,  # 2 shards over 1290 pixels
            "x": var_chunks["x"] * 10,  # 4 shards over 2540 pixels
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(var_chunks[d] for d in self.dims[ROOT]),
            shards=tuple(var_shards[d] for d in self.dims[ROOT]),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        return self.get_data_vars(encoding_float32_default)
