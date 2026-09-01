from collections.abc import Sequence
from typing import Any

import pandas as pd
from pydantic import computed_field

from reformatters.common.config_models import (
    ROOT,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.types import AppendDim, Dims, Timedelta, Timestamp
from reformatters.common.zarr import BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_CURRENT_ARCHIVE_START,
    GEFSSourceFileType,
)
from reformatters.noaa.gefs.virtual_template_config import NoaaGefsVirtualTemplateConfig


class NoaaGefsAnalysis025DegreeVirtualTemplateConfig(NoaaGefsVirtualTemplateConfig):
    """Virtual GEFS analysis: the control member's shortest available lead at each
    3-hourly valid time."""

    source_file_types: frozenset[GEFSSourceFileType] = frozenset({"s"})
    # Each valid time takes the shortest lead carrying it, so a windowed value covers
    # the 6 hours since the previous synoptic cycle at 00/06/12/18 UTC and the 3 hours
    # since it at 03/09/15/21 UTC.
    window_comments: dict[str, str] = {
        "avg": "Average value in the last 6 hour period (00, 06, 12, 18 UTC) or 3 hour period (03, 09, 15, 21 UTC).",
        "accum": (
            "Total accumulated in the last 6 hour period (00, 06, 12, 18 UTC) or 3 hour "
            "period (03, 09, 15, 21 UTC). Subtracting the value at an earlier time with "
            "the same window start gives the exact total between those two times."
        ),
        "max": "Maximum value in the last 6 hour period (00, 06, 12, 18 UTC) or 3 hour period (03, 09, 15, 21 UTC).",
        "min": "Minimum value in the last 6 hour period (00, 06, 12, 18 UTC) or 3 hour period (03, 09, 15, 21 UTC).",
    }
    dims: Dims = {ROOT: ("time", "latitude", "longitude")}
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = GEFS_CURRENT_ARCHIVE_START
    append_dim_frequency: Timedelta = pd.Timedelta("3h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return self._dataset_attributes(
            dataset_id="noaa-gefs-analysis-0-25-degree-virtual",
            dataset_version="0.1.0",
            name="NOAA GEFS analysis, 0.25 degree, virtual",
            description=(
                "Weather analysis from the Global Ensemble Forecast System (GEFS) "
                "operated by NOAA NWS NCEP, served as references to the source GRIB "
                "messages. Each time step is the control member's shortest available "
                "forecast lead. Coverage begins with the GEFS v12 0.25 degree archive; "
                "the materialized noaa-gefs-analysis reaches further back through "
                "coarser and reforecast sources, which cannot share one grid here."
            ),
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            "time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            **self._spatial_dimension_coordinates(),
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        return (
            *self._spatial_coords(),
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
        )
