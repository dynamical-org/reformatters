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
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import AppendDim, Dims, Timedelta, Timestamp
from reformatters.common.zarr import BLOSC_4BYTE_ZSTD_LEVEL3_BITSHUFFLE
from reformatters.dwd.icon_eu.forecast_5_day.template_config import (
    DwdIconEuDataVar,
    DwdIconEuForecast5DayTemplateConfig,
)

from . import icosahedral_grid


class DwdIconEuForecast5DayIcosahedralTemplateConfig(TemplateConfig[DwdIconEuDataVar]):
    dims: Dims = {ROOT: ("init_time", "lead_time", "cell")}
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = pd.Timestamp("2026-10-07T00")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")

    def _regular_grid_config(self) -> DwdIconEuForecast5DayTemplateConfig:
        return DwdIconEuForecast5DayTemplateConfig(
            append_dim_start=self.append_dim_start
        )

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return replace(
            self._regular_grid_config().dataset_attributes,
            dataset_id="dwd-icon-eu-forecast-5-day-icosahedral",
            dataset_version="0.1.0",
            name="DWD ICON-EU forecast, 5 day, icosahedral",
            description="High-resolution weather forecasts for Europe from the ICON-EU model operated by Deutscher Wetterdienst (DWD), on its native icosahedral grid.",
            spatial_resolution="6.5 km",
            source_grid_name=icosahedral_grid.GRID_NAME,
            source_grid_id=icosahedral_grid.GRID_UUID,
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        regular = self._regular_grid_config().dimension_coordinates()
        return {
            "init_time": regular["init_time"],
            "lead_time": regular["lead_time"],
            "cell": icosahedral_grid.cell_indices(),
        }

    def static_groups(self) -> dict[str, xr.Dataset]:
        return {"mesh": icosahedral_grid.mesh_dataset()}

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        latitude, longitude = icosahedral_grid.cell_centres()
        return {
            **self._regular_grid_config().derive_coordinates(ds),
            "latitude": (("cell",), latitude),
            "longitude": (("cell",), longitude),
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        regular = self._regular_grid_config().coords
        latitude, longitude = icosahedral_grid.cell_centres()
        result = [
            coord for coord in regular if coord.name not in ("latitude", "longitude")
        ]
        result.append(
            Coordinate(
                name="cell",
                encoding=Encoding(
                    dtype="int32",
                    fill_value=-1,
                    chunks=icosahedral_grid.CELL_COUNT,
                    shards=None,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_BITSHUFFLE],
                ),
                attrs=CoordinateAttrs(
                    long_name="Native grid cell index",
                    units="1",
                    statistics_approximate=StatisticsApproximate(
                        min=0, max=icosahedral_grid.CELL_COUNT - 1
                    ),
                ),
            )
        )
        for name, values in (("latitude", latitude), ("longitude", longitude)):
            original = next(coord for coord in regular if coord.name == name)
            result.append(
                replace(
                    original,
                    encoding=Encoding(
                        dtype="float32",
                        fill_value=np.nan,
                        chunks=icosahedral_grid.CELL_COUNT,
                        shards=None,
                        compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_BITSHUFFLE],
                    ),
                    attrs=replace(
                        original.attrs,
                        axis=None,
                        statistics_approximate=StatisticsApproximate(
                            min=float(values.min()), max=float(values.max())
                        ),
                    ),
                )
            )
        return result

    @computed_field
    @property
    def data_vars(self) -> Sequence[DwdIconEuDataVar]:
        result = []
        for var in self._regular_grid_config().data_vars:
            attrs = replace(
                var.attrs,
                mesh="/mesh/mesh",
                location="face",
                cell_measures="area: /mesh/cell_area",
            )
            if var.name == "wind_gust_10m":
                attrs = replace(
                    attrs,
                    comment="Maximum over the hour ending at the forecast step. At 3-hourly steps (81-120 hours), this covers only the last hour of each step.",
                )
            result.append(
                replace(
                    var,
                    attrs=attrs,
                    encoding=replace(
                        var.encoding,
                        chunks=(1, 93, 24019),
                        shards=(1, 93, icosahedral_grid.CELL_COUNT),
                    ),
                )
            )
        return result
