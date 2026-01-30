import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import computed_field

from reformatters.common import template_utils
from reformatters.common.config_models import (
    Coordinate,
    DatasetAttributes,
    DataVar,
)
from reformatters.common.pydantic import FrozenBaseModel
from reformatters.common.types import AppendDim, DatetimeLike, Dim, Timedelta, Timestamp

DATA_VAR = TypeVar("DATA_VAR", bound=DataVar[Any])

# Value is ignored, coordinate reference system metadata is stored in attributes
SPATIAL_REF_COORDS = ((), np.array(0))


class TemplateConfig(FrozenBaseModel, Generic[DATA_VAR]):
    """Define a subclass of this class to configure the structure of a dataset."""

    dims: tuple[Dim, ...]
    append_dim: AppendDim
    append_dim_start: Timestamp
    append_dim_frequency: Timedelta

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        raise NotImplementedError("Implement `dataset_attributes` in your subclass")

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        raise NotImplementedError("Implement `coords` in your subclass")

    @computed_field
    @property
    def data_vars(self) -> Sequence[DATA_VAR]:
        raise NotImplementedError("Implement `data_vars` in your subclass")

    def dimension_coordinates(self) -> dict[str, Any]:
        """Returns a dictionary of dimension names to coordinates for the dataset."""
        raise NotImplementedError("Implement `dimension_coordinates` in your subclass")

    def derive_coordinates(
        self, _ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        """
        Compute non-dimension coordinates.
        For example, if init_time is the append dimension, `{"valid_time": ds["init_time"] + ds["lead_time"]}`
        """
        non_dimension_coords = {c.name for c in self.coords if c.name not in self.dims}
        missing = non_dimension_coords - {"spatial_ref"}
        if len(missing):
            raise NotImplementedError(
                f"Coordinates {missing} are defined in self.coords and should be returned from your template config's derive_coordinates method"
            )
        return {
            "spatial_ref": SPATIAL_REF_COORDS,
        }

    # ----- Most subclasses will not need to override the attributes and methods below -----

    @computed_field
    @property
    def dataset_id(self) -> str:
        return self.dataset_attributes.dataset_id

    @computed_field
    @property
    def version(self) -> str:
        return self.dataset_attributes.dataset_version

    def append_dim_coordinates(self, end: DatetimeLike) -> pd.DatetimeIndex:
        """
        Returns DatetimeIndex for the append dimension from the configured start time to the given end time.

        Args:
            end (DatetimeLike): End time (exclusive) for the coordinates
        """
        return pd.date_range(
            self.append_dim_start, end, freq=self.append_dim_frequency, inclusive="left"
        )

    def get_template(self, end_time: DatetimeLike) -> xr.Dataset:
        """
        Returns a template dataset expanded to the given end time.

        Args:
            end_time (pd.Timestamp): End time (exclusive) for the append dimension

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

        # Ensure attributes have not been changed without calling update_template()
        assert ds.attrs["dataset_id"] == self.dataset_id
        assert ds.attrs["dataset_version"] == self.version

        # Coordinates which are dask arrays are not written with .to_zarr(store, compute=False)
        # We want to write all coords when writing metadata, so ensure they are loaded as numpy arrays.
        for coordinate in ds.coords.values():
            coordinate.load()

        # Work around what appears to be a bug where fill_value is not set in encodings read from existing zarr template
        for coord in self.coords:
            assert "fill_value" not in ds[coord.name].encoding, (
                "Fill value round tripped. That's good but not the previous behavior and if you see this AND the fill_value is correct, you can remove the workaround."
            )
            ds[coord.name].encoding["fill_value"] = coord.encoding.fill_value
        for var in self.data_vars:
            assert "fill_value" not in ds[var.name].encoding, (
                "Fill value round tripped. That's good but not the previous behavior and if you see this AND the fill_value is correct, you can remove the workaround."
            )
            ds[var.name].encoding["fill_value"] = var.encoding.fill_value

        return ds

    def update_template(self) -> None:
        """
        Updates the template file on disk with the latest configuration.
        """
        coords = self.dimension_coordinates()
        assert set(coords) == set(self.dims), (
            f"`dimension_coordinates` must return coordinates for all dimensions {self.dims}"
        )

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

        derived_coords = self.derive_coordinates(ds)
        assert set(derived_coords).isdisjoint(ds.dims), (
            f"Return coordinates for dataset dimensions {self.dims} from `dimension_coordinates` rather than `derive_coordinates`."
        )
        ds = ds.assign_coords(derived_coords)

        assert {d.name for d in self.data_vars} == set(ds.data_vars)
        for var_config in self.data_vars:
            template_utils.assign_var_metadata(ds[var_config.name], var_config)

        assert {c.name for c in self.coords} == set(ds.coords)
        for coord_config in self.coords:
            template_utils.assign_var_metadata(
                ds.coords[coord_config.name], coord_config
            )

        template_utils.write_metadata(ds, self.template_path())

    def append_dim_coordinate_chunk_size(self) -> int:
        """
        Returns a stable, fixed chunk size for the append dimension to allow
        expansion while making an effort to keep all coordinates in a single chunk.
        """
        # Give ourselves about 15 years of future dataset appending.
        # 10 as a minimum to ensure we have some buffer.
        num_years = max(2025 - self.append_dim_start.year + 15, 10)
        return int(pd.Timedelta(days=365 * num_years) / self.append_dim_frequency)

    def template_path(self) -> Path:
        """Returns the templates/latest.zarr which is a sibling of the template config file."""
        cls = self.__class__
        assert cls is not TemplateConfig, "template_path() should only be called from a subclass"  # fmt: skip
        subclass_template_config_file = sys.modules[cls.__module__].__file__
        assert subclass_template_config_file is not None
        return Path(subclass_template_config_file).parent / "templates" / "latest.zarr"
