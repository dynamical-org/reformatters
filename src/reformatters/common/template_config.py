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
    ROOT,
    Coordinate,
    DatasetAttributes,
    DataVar,
    Group,
)
from reformatters.common.iterating import node_group_name
from reformatters.common.pydantic import FrozenBaseModel
from reformatters.common.types import AppendDim, DatetimeLike, Dim, Timedelta, Timestamp

DATA_VAR = TypeVar("DATA_VAR", bound=DataVar[Any])

# Value is ignored, coordinate reference system metadata is stored in attributes
SPATIAL_REF_COORDS = ((), np.array(0))


class TemplateConfig(FrozenBaseModel, Generic[DATA_VAR]):
    """Define a subclass of this class to configure the structure of a dataset."""

    # Dimensions per zarr group, keyed by group. dims[ROOT] is the root/single-level
    # dims; dims["pressure_level"] etc. are a vertical group's full dims (the shared
    # dims plus the vertical dim, whose name equals the group name). A single-level
    # dataset declares only {ROOT: (...)}. See docs/plans/vertical_dimension_structure.md.
    dims: dict[Group, tuple[Dim, ...]]
    append_dim: AppendDim
    append_dim_start: Timestamp
    append_dim_frequency: Timedelta

    @property
    def all_dims(self) -> tuple[Dim, ...]:
        """De-duplicated union of every group's dims, root dims first."""
        ordered: dict[Dim, None] = {}
        for group_dims in self.dims.values():
            for dim in group_dims:
                ordered[dim] = None
        return tuple(ordered)

    @property
    def groups(self) -> tuple[Group, ...]:
        """Groups to write: ROOT (always, holds shared coords) then each vertical group
        that has at least one data var (empty vertical groups are omitted)."""
        used = {var.group for var in self.data_vars}
        vertical = tuple(g for g in self.dims if g is not ROOT and g in used)
        return (ROOT, *vertical)

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
        self,
        ds: xr.Dataset,  # noqa: ARG002
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        """
        Compute non-dimension coordinates.
        For example, if init_time is the append dimension, `{"valid_time": ds["init_time"] + ds["lead_time"]}`
        """
        non_dimension_coords = {
            c.name for c in self.coords if c.name not in self.all_dims
        }
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

    def get_template(self, end_time: DatetimeLike) -> xr.DataTree:
        """
        Returns the template, as a DataTree, expanded to the given end time.

        A single-level dataset is a one-node tree (just the root); a dataset with
        vertical groups has one child node per group. Each node's append dimension
        is reindexed and its coordinates re-derived.

        Args:
            end_time (pd.Timestamp): End time (exclusive) for the append dimension

        Returns:
            xr.DataTree: Template tree with dimension coordinates
        """
        on_disk = xr.open_datatree(self.template_path(), decode_timedelta=True)
        new_append_coords = self.append_dim_coordinates(end_time)
        coord_fill_values = {c.name: c.encoding.fill_value for c in self.coords}
        var_by_path = {var.path: var for var in self.data_vars}

        nodes: dict[str, xr.Dataset] = {}
        for node in on_disk.subtree:
            group_prefix = f"{group}/" if (group := node_group_name(node)) else ""
            ds = template_utils.empty_copy_with_reindex(
                node.to_dataset(),
                self.append_dim,
                new_append_coords,
                derive_coordinates_fn=self.derive_coordinates,
            )
            # Coordinates which are dask arrays are not written with .to_zarr(store, compute=False)
            # We want to write all coords when writing metadata, so ensure they are loaded as numpy arrays.
            for coordinate in ds.coords.values():
                coordinate.load()

            # Work around what appears to be a bug where fill_value is not set in encodings read from existing zarr template
            for coord_name in ds.coords:
                assert "fill_value" not in ds[coord_name].encoding, (
                    "Fill value round tripped. That's good but not the previous behavior and if you see this AND the fill_value is correct, you can remove the workaround."
                )
                ds[coord_name].encoding["fill_value"] = coord_fill_values[
                    str(coord_name)
                ]
            for var_name in ds.data_vars:
                var = var_by_path[f"{group_prefix}{var_name}"]
                assert "fill_value" not in ds[var_name].encoding, (
                    "Fill value round tripped. That's good but not the previous behavior and if you see this AND the fill_value is correct, you can remove the workaround."
                )
                ds[var_name].encoding["fill_value"] = var.encoding.fill_value
            nodes[node.path] = ds

        template = xr.DataTree.from_dict(nodes)

        # Ensure attributes have not been changed without calling update_template()
        assert template.attrs["dataset_id"] == self.dataset_id
        assert template.attrs["dataset_version"] == self.version

        return template

    def update_template(self) -> None:
        """
        Updates the template file on disk with the latest configuration.
        """
        coords = self.dimension_coordinates()
        assert set(coords) == set(self.all_dims), (
            f"`dimension_coordinates` must return coordinates for all dims {self.all_dims}"
        )
        self._assert_valid_structure()

        nodes = {
            self._group_node_path(group): self._build_node_dataset(group, coords)
            for group in self.groups
        }
        written_coords = {name for ds in nodes.values() for name in ds.coords}
        assert written_coords == {c.name for c in self.coords}, (
            f"coords {written_coords} written across groups must match self.coords "
            f"{ {c.name for c in self.coords} }"
        )
        template = xr.DataTree.from_dict(nodes)
        template_utils.write_metadata(template, self.template_path())

    @staticmethod
    def _group_node_path(group: Group) -> str:
        return "/" if group is ROOT else f"/{group}"

    def _build_node_dataset(self, group: Group, coords: dict[str, Any]) -> xr.Dataset:
        """Build one zarr group's empty dataset: its data vars plus its dims' coords
        (shared coords + any vertical coord) and the derived coords."""
        group_dims = self.dims[group]
        data_vars = {
            var.name: template_utils.make_empty_variable(
                group_dims, coords, var.encoding.dtype
            )
            for var in self.data_vars
            if var.group == group
        }
        ds = xr.Dataset(
            data_vars,
            {dim: coords[dim] for dim in group_dims},
            self.dataset_attributes.model_dump(exclude_none=True),
        )
        derived_coords = self.derive_coordinates(ds)
        assert set(derived_coords).isdisjoint(ds.dims), (
            "Return coordinates for dataset dimensions from `dimension_coordinates` "
            "rather than `derive_coordinates`."
        )
        ds = ds.assign_coords(derived_coords)

        for var in self.data_vars:
            if var.group == group:
                template_utils.assign_var_metadata(ds[var.name], var)
        coord_config = {c.name: c for c in self.coords}
        for coord_name in ds.coords:
            template_utils.assign_var_metadata(
                ds.coords[coord_name], coord_config[str(coord_name)]
            )
        return ds

    def _assert_valid_structure(self) -> None:
        """Enforce the group invariants (see docs/plans/vertical_dimension_structure.md)."""
        root_dims = self.dims[ROOT]
        for group, group_dims in self.dims.items():
            if group is ROOT:
                continue
            added = tuple(d for d in group_dims if d not in root_dims)
            assert added == (group,), (
                f"vertical group {group!r} must add exactly its own dim to the root "
                f"dims; dims[{group!r}] adds {added}, expected ({group!r},)"
            )

        declared = set(self.dims)
        used = {var.group for var in self.data_vars}
        assert used <= declared, (
            f"data var group(s) {used - declared} are not declared in dims"
        )
        orphans = {g for g in self.dims if g is not ROOT} - used
        assert not orphans, f"vertical group(s) {orphans} declared in dims but unused"

        paths = [var.path for var in self.data_vars]
        assert len(paths) == len(set(paths)), (
            f"duplicate (group, name) across data_vars: {paths}"
        )

        group_names = {g for g in self.dims if g is not ROOT}
        root_var_names = {var.name for var in self.data_vars if var.group is ROOT}
        assert root_var_names.isdisjoint(group_names), (
            f"root data var name(s) {root_var_names & group_names} collide with a group name"
        )

        append_dim_chunks = {self._append_dim_chunk_size(var) for var in self.data_vars}
        assert len(append_dim_chunks) <= 1, (
            f"all data vars must share one append-dim chunk size, got {append_dim_chunks}"
        )

        for var in self.data_vars:
            n_dims = len(self.dims[var.group])
            for kind in ("chunks", "shards"):
                value = getattr(var.encoding, kind)
                if isinstance(value, tuple):
                    assert len(value) == n_dims, (
                        f"{var.path} encoding {kind} has {len(value)} entries, "
                        f"expected {n_dims} (the dims of group {var.group!r})"
                    )

    def _append_dim_chunk_size(self, var: DataVar[Any]) -> int:
        dims = self.dims[var.group]
        chunks = var.encoding.chunks
        if isinstance(chunks, int):
            return chunks
        return chunks[dims.index(self.append_dim)]

    def append_dim_coordinate_chunk_size(self) -> int:
        """
        Returns a stable, fixed chunk size for the append dimension to allow
        expansion while making an effort to keep all coordinates in a single chunk.
        """
        # Give ourselves about 15 years of future dataset appending.
        # 10 as a minimum to ensure we have some buffer.
        num_years = max(2025 - self.append_dim_start.year + 15, 10)
        total_timedelta = pd.Timedelta(days=365 * num_years)
        result: float = total_timedelta / self.append_dim_frequency  # ty: ignore[invalid-assignment]
        return int(result)

    def template_path(self) -> Path:
        """Returns the templates/latest.zarr which is a sibling of the template config file."""
        cls = self.__class__
        assert cls is not TemplateConfig, "template_path() should only be called from a subclass"  # fmt: skip
        subclass_template_config_file = sys.modules[cls.__module__].__file__
        assert subclass_template_config_file is not None
        return Path(subclass_template_config_file).parent / "templates" / "latest.zarr"
