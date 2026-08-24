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
)
from reformatters.common.pydantic import replace
from reformatters.common.template_config import SPATIAL_REF_COORDS
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import AppendDim, Dims, Timedelta, Timestamp
from reformatters.common.zarr import BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from reformatters.noaa.hrrr.virtual_template_config import (
    NoaaHrrrVirtualTemplateConfig,
)


class NoaaHrrrAnalysisVirtualTemplateConfig(NoaaHrrrVirtualTemplateConfig):
    """Virtual HRRR analysis with hourly valid times."""

    dims: Dims = {
        ROOT: ("time", "y", "x"),
        "pressure_level": ("time", "y", "x", "pressure_level"),
        "model_level": ("time", "y", "x", "model_level"),
    }
    append_dim: AppendDim = "time"
    # HRRR operational start is 2014-09-30; skip the incomplete first day.
    append_dim_start: Timestamp = pd.Timestamp("2014-10-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("1h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="noaa-hrrr-analysis-virtual",
            dataset_version="0.1.0",
            name="NOAA HRRR analysis, virtual",
            description="Analysis data from the High-Resolution Rapid Refresh (HRRR) model operated by NOAA NWS NCEP.",
            attribution="NOAA NWS NCEP HRRR data processed by dynamical.org from NOAA Open Data Dissemination archives.",
            license="CC-BY-4.0",
            spatial_domain="Continental United States",
            spatial_resolution="3 km",
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
            **self._vertical_dimension_coordinates(),
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

    def _catalog_data_vars(self) -> list[NoaaHrrrDataVar]:
        catalog = super()._catalog_data_vars()
        names = {var.name for var in catalog}
        return [
            _with_run_total_comment(var, names)
            if var.internal_attrs.window_reset_frequency == pd.Timedelta.max
            else var
            for var in catalog
        ]


def _with_run_total_comment(var: NoaaHrrrDataVar, names: set[str]) -> NoaaHrrrDataVar:
    assert var.attrs.comment is None, f"{var.path} already has a comment"
    hourly_name = var.name.replace("_run_total", "")
    comment = (
        f"Identical to the one hour accumulated {hourly_name} in this analysis dataset."
        if hourly_name in names
        else "Accumulated over the one hour ending at this time."
    )
    return replace(var, attrs=replace(var.attrs, comment=comment))
