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
    prepend_comment,
)
from reformatters.common.template_config import SPATIAL_REF_COORDS
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import AppendDim, Dims, Timedelta, Timestamp
from reformatters.common.zarr import BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE
from reformatters.noaa.gfs.virtual_template_config import NoaaGfsVirtualTemplateConfig
from reformatters.noaa.models import NoaaDataVar

# Windowed variables are read at leads 1-6 of the 6-hourly cycles, so a value's window
# opens at the most recent 00, 06, 12 or 18 hour strictly before its time. This store
# has no lead_time dimension, so the window is described in the UTC times a reader sees.
_WINDOW_EXTENT = (
    "the preceding 1-6 hours, since the 00, 06, 12 or 18 UTC hour before this time."
)
_WINDOW_COMMENTS = {
    "accum": f"Accumulated over {_WINDOW_EXTENT}",
    "avg": f"Averaged over {_WINDOW_EXTENT}",
    "max": f"Maximum value over {_WINDOW_EXTENT}",
    "min": f"Minimum value over {_WINDOW_EXTENT}",
}


class NoaaGfsAnalysisVirtualTemplateConfig(NoaaGfsVirtualTemplateConfig):
    """Virtual GFS analysis with hourly valid times."""

    dims: Dims = {
        ROOT: ("time", "latitude", "longitude"),
        "pressure_level": ("time", "latitude", "longitude", "pressure_level"),
        "height_above_mean_sea_level": (
            "time",
            "latitude",
            "longitude",
            "height_above_mean_sea_level",
        ),
    }
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = pd.Timestamp("2021-05-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("1h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="noaa-gfs-analysis-virtual",
            dataset_version="0.1.0",
            name="NOAA GFS analysis, virtual",
            description="Weather analysis from the Global Forecast System (GFS) operated by NOAA NWS NCEP.",
            attribution="NOAA NWS NCEP GFS data processed by dynamical.org from NOAA Open Data Dissemination archives.",
            license="CC-BY-4.0",
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
            **self._vertical_dimension_coordinates(),
        }

    def derive_coordinates(
        self,
        ds: xr.Dataset,  # noqa: ARG002
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        return {"spatial_ref": SPATIAL_REF_COORDS}

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
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
                    chunks=self.append_dim_coordinate_chunk_size(),
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
            *self._vertical_coords(),
        ]

    def _catalog_data_vars(self) -> list[NoaaDataVar]:
        """The shared catalog without its running totals, which duplicate the 6 hour
        buckets at the leads an analysis reads, and with each windowed variable's
        window described in UTC times."""
        return [
            _with_window_comment(var)
            for var in super()._catalog_data_vars()
            if var.internal_attrs.window_reset_frequency != pd.Timedelta.max
        ]


def _with_window_comment(var: NoaaDataVar) -> NoaaDataVar:
    window_comment = _WINDOW_COMMENTS.get(var.attrs.step_type)
    if window_comment is None:
        return var
    return prepend_comment(var, window_comment)
