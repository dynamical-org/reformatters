from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, computed_field

from reformatters.common import template_utils
from reformatters.common.config_models import Coordinate, DatasetAttributes, DataVar
from reformatters.common.types import DatetimeLike

type Dim = Literal["init_time", "ensemble_member", "lead_time", "latitude", "longitude"]
type AppendDim = Literal["init_time", "time"]


class TemplateConfig(BaseModel):
    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_id(self) -> str:
        return self.dataset_attributes.dataset_id

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_version(self) -> str:
        return self.dataset_attributes.dataset_version

    dataset_attributes: DatasetAttributes

    time_start: pd.Timestamp
    time_frequency: pd.Timedelta

    dims: tuple[Dim, ...]
    append_dim: AppendDim

    var_chunks: dict[Dim, int]
    var_shards: dict[Dim, int]

    coords: Sequence[Coordinate]

    def dimension_coordinates(self) -> dict[str, Any]:
        """
        Returns a dictionary of dimension names to coordinates for the dataset.
        """
        raise NotImplementedError(
            "Return a dictionary from dimension names to coordinate arrays"
        )

    def append_dim_coordinates(self, end: DatetimeLike) -> pd.DatetimeIndex:
        """
        Returns DatetimeIndex for the append dimension from the configured start time to the given end time.

        Args:
            end (DatetimeLike): End time (exclusive) for the coordinates
        """
        return pd.date_range(
            self.time_start, end, freq=self.time_frequency, inclusive="left"
        )

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        """
        Compute non-dimension coordinates which are dependent on coordinates of the append dimension.
        For example, if init_time is the append dimension, `{"valid_time": ds["init_time"] + ds["lead_time"]}`

        For analysis and climatology datasets you likely don't need to implement this.
        """
        if missing := {
            "valid_time",
            "ingested_forecast_length",
            "expected_forecast_length",
        }.intersection({coord.name for coord in self.coords}):
            raise NotImplementedError(
                f"Coordinates {missing} are defined in self.coords and should be derived in your template config's derive_coordinates method"
            )
        return {}

    def get_template(self, end_time: pd.Timestamp) -> xr.Dataset:
        """
        Returns a template dataset expanded to the given end time.

        Args:
            end_time (pd.Timestamp): End time for the append dimension

        Returns:
            xr.Dataset: Template dataset with dimension coordinates
        """
        ds: xr.Dataset = xr.open_zarr(self.template_path(), decode_timedelta=True)

        # Expand init_time dimension with complete coordinates
        ds = template_utils.empty_copy_with_reindex(
            ds,
            self.append_dim,
            self.append_dim_coordinates(end_time),
            derive_coordinates_fn=self.derive_coordinates,
        )

        # Coordinates which are dask arrays are not written with .to_zarr(store, compute=False)
        # We want to write all coords when writing metadata, so ensure they are loaded as numpy arrays.
        for coordinate in ds.coords.values():
            coordinate.load()

        return ds

    def update_template(self) -> None:
        """
        Updates the template file on disk with the latest configuration.
        """
        coords = self.dimension_coordinates()

        data_vars = {
            var_config.name: template_utils.make_empty_variable(
                self.dims, coords, var_config.encoding.dtype
            )
            for var_config in self.data_vars
        }

        ds = xr.Dataset(
            data_vars,
            coords,
            self.dataset_attributes.model_dump(exclude_none=True),
        )

        ds = ds.assign_coords(self.derive_coordinates(ds))

        assert {d.name for d in self.data_vars} == set(ds.data_vars)
        for var_config in self.data_vars:
            template_utils.assign_var_metadata(ds[var_config.name], var_config)

        assert {c.name for c in self.coords} == set(ds.coords)
        for coord_config in self.coords:
            template_utils.assign_var_metadata(
                ds.coords[coord_config.name], coord_config
            )

        template_utils.write_metadata(ds, self.template_path(), mode="w")

    def template_path(self, template_config_file_path: str) -> Path:
        return Path(template_config_file_path).parent / "templates" / "latest.zarr"
