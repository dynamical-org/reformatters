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
from reformatters.common.template_config import SPATIAL_REF_COORDS, TemplateConfig
from reformatters.common.types import AppendDim, Dims, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_BITSHUFFLE,
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.dwd.icon_eu.icon_eu_template_config import (
    DwdIconEuDataVar,
    derive_forecast_coordinates,
    forecast_coordinate_configs,
    forecast_dimension_coordinates,
    icon_eu_data_vars,
)

from . import icosahedral_grid


class DwdIconEuForecast5DayIcosahedralTemplateConfig(TemplateConfig[DwdIconEuDataVar]):
    dims: Dims = {ROOT: ("init_time", "lead_time", "cell")}
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = pd.Timestamp("2026-10-07T00")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="dwd-icon-eu-forecast-5-day-icosahedral",
            dataset_version="0.1.0",
            name="DWD ICON-EU forecast, 5 day, icosahedral",
            description="High-resolution weather forecasts for Europe from the ICON-EU model operated by Deutscher Wetterdienst (DWD), on its native icosahedral grid.",
            attribution="DWD ICON-EU data processed by dynamical.org.",
            license="CC-BY-4.0",
            spatial_domain="Europe",
            spatial_resolution="6.5 km",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:.0f} hours",
            forecast_domain="Forecast lead time 0-120 hours (0-5 days) ahead",
            forecast_resolution="Forecast step 0-78 hours: hourly, 81-120 hours: 3 hourly",
            source_grid_name=icosahedral_grid.GRID_NAME,
            source_grid_id=icosahedral_grid.GRID_UUID,
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            **forecast_dimension_coordinates(
                self.append_dim_coordinates(
                    self.append_dim_start + self.append_dim_frequency
                )
            ),
            "cell": icosahedral_grid.cell_indices(),
        }

    def static_groups(self) -> dict[str, xr.Dataset]:
        return {"mesh": icosahedral_grid.mesh_dataset()}

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        latitude, longitude = icosahedral_grid.cell_centers()
        return {
            **derive_forecast_coordinates(ds),
            "spatial_ref": SPATIAL_REF_COORDS,
            "latitude": (("cell",), latitude),
            "longitude": (("cell",), longitude),
            "cell_area": (("cell",), icosahedral_grid.grid_arrays()["cell_area"]),
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        latitude, longitude = icosahedral_grid.cell_centers()
        result = [
            *forecast_coordinate_configs(
                self.dimension_coordinates()["init_time"],
                self.append_dim_coordinate_chunk_size(),
            ),
            icosahedral_grid.spatial_reference_coordinate(),
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
            result.append(
                Coordinate(
                    name=name,
                    encoding=Encoding(
                        dtype="float32",
                        fill_value=np.nan,
                        chunks=icosahedral_grid.CELL_COUNT,
                        shards=None,
                        compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_BITSHUFFLE],
                    ),
                    attrs=CoordinateAttrs(
                        long_name=name.capitalize(),
                        standard_name=name,
                        units="degree_north" if name == "latitude" else "degree_east",
                        statistics_approximate=StatisticsApproximate(
                            min=float(values.min()), max=float(values.max())
                        ),
                    ),
                )
            )
        area = icosahedral_grid.grid_arrays()["cell_area"]
        result.append(
            Coordinate(
                name="cell_area",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    chunks=icosahedral_grid.CELL_COUNT,
                    shards=None,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_BITSHUFFLE],
                ),
                attrs=CoordinateAttrs(
                    long_name="Cell area",
                    standard_name="cell_area",
                    units="m2",
                    statistics_approximate=StatisticsApproximate(
                        min=float(area.min()), max=float(area.max())
                    ),
                ),
            )
        )
        return result

    @computed_field
    @property
    def data_vars(self) -> Sequence[DwdIconEuDataVar]:
        result = []
        encoding = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=(1, 93, 24019),
            shards=(1, 93, icosahedral_grid.CELL_COUNT),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )
        for var in icon_eu_data_vars(encoding):
            attrs = replace(
                var.attrs,
                mesh="/mesh/mesh",
                location="face",
                cell_measures="area: cell_area",
            )
            if var.name == "wind_gust_10m":
                attrs = replace(
                    attrs,
                    comment="Maximum over the hour ending at the forecast step. At 3-hourly steps (81-120 hours), this covers only the last hour of each step.",
                )
            result.append(replace(var, attrs=attrs))
        return result
