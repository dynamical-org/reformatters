from collections.abc import Callable, Iterator, Mapping, Sequence
from enum import Enum, auto
from itertools import chain, pairwise
from pathlib import Path
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    Self,
    TypeVar,
    get_args,
)

import numpy as np
import pandas as pd
import pydantic
import xarray as xr
from pydantic import (
    AfterValidator,
    Field,
    computed_field,
    field_serializer,
    field_validator,
)
from zarr.abc.store import Store

from reformatters.common import storage
from reformatters.common.config_models import DataVar
from reformatters.common.iterating import (
    chunk_slices,
    node_group_name,
    split_groups,
    spread_evenly,
)
from reformatters.common.logging import get_logger
from reformatters.common.pydantic import FrozenBaseModel
from reformatters.common.types import (
    AppendDim,
    DatetimeLike,
    Dim,
    Timestamp,
)

log = get_logger(__name__)

type CoordinateValue = int | float | pd.Timestamp | pd.Timedelta | str


def walk_data_arrays(tree: xr.DataTree) -> Iterator[tuple[str, xr.DataArray]]:
    """Yield (var_path, DataArray) for every data var across all of a template's groups."""
    for node in tree.subtree:
        prefix = f"{group}/" if (group := node_group_name(node)) else ""
        for name, data_array in node.to_dataset().data_vars.items():
            yield f"{prefix}{name}", data_array


class SourceFileStatus(Enum):
    Processing = auto()
    DownloadFailed = auto()
    ReadFailed = auto()
    Succeeded = auto()


class SourceFileCoord(FrozenBaseModel):
    """
    Base class representing the coordinates and status of a single source file required for processing.

    Subclasses should define dataset-specific fields (e.g., data_vars, init_time, lead_time, file_type) required
    to uniquely identify a source file and implement the `get_url` and `out_loc` methods.

    Attributes
    ----------
    status : SourceFileStatus
        The current processing status of this file (Processing, DownloadFailed, ReadFailed, Succeeded).
    downloaded_path : Path | None
        Local filesystem path to the downloaded file, or None if not downloaded.
    """

    status: SourceFileStatus = Field(default=SourceFileStatus.Processing, frozen=False)
    downloaded_path: Path | None = Field(default=None, frozen=False)

    def get_url(self) -> str:
        """Return the URL for this source file."""
        raise NotImplementedError("Return the URL of the source file.")

    def get_index_url(self) -> str:
        """Return the URL of this source file's byte-range index, if it has one."""
        raise NotImplementedError("Return the URL of the source file's index.")

    def out_loc(
        self,
    ) -> Mapping[Dim, CoordinateValue]:
        """
        Return a data array indexer which identifies the region in the output dataset
        to write the data from the source file. The indexer is a dict from dimension
        names to coordinate values.

        If the names of the coordinate attributes of your SourceFileCoord subclass are also all
        dimension names in the output dataset, use the default implementation of this method.

        Examples where you would override this method:
        - For an analysis dataset created from forecast data: {"time": self.init_time + self.lead_time}
        """
        # .model_dump() returns a dict from attribute names to values
        return self.model_dump(exclude={"status", "downloaded_path"})

    @property
    def append_dim_coord(self) -> CoordinateValue:
        """Return the coordinate value for the append dimension."""
        out_loc = self.out_loc()
        for d in get_args(AppendDim.__value__):
            if coord := out_loc.get(d):
                return coord
        return pd.Timestamp.min


DATA_VAR = TypeVar("DATA_VAR", bound=DataVar[Any])
SOURCE_FILE_COORD = TypeVar("SOURCE_FILE_COORD", bound=SourceFileCoord)


def region_slice(s: slice) -> slice:
    if not (isinstance(s.start, int) and isinstance(s.stop, int) and s.step is None):
        raise ValueError("region must be integer slice")
    return s


class RegionJob(pydantic.BaseModel, Generic[DATA_VAR, SOURCE_FILE_COORD]):
    tmp_store: Path
    # Whole-dataset template tree: root + one node per vertical group (single-level = one node).
    template_ds: xr.DataTree
    data_vars: Sequence[DATA_VAR]
    append_dim: AppendDim
    # integer slice along append_dim
    region: Annotated[slice, AfterValidator(region_slice)]
    reformat_job_name: str

    # Limit the number of variables processed in each job if set.
    # Rule of thumb: leave unset unless a job takes > 15 minutes.
    max_vars_per_job: ClassVar[int | None] = None

    @classmethod
    def source_file_var_groups(
        cls,
        data_vars: Sequence[DATA_VAR],
    ) -> Sequence[Sequence[DATA_VAR]]:
        """
        Return groups of variables, where all variables in a group can be retrieved from the same source file.

        Distinct from a dataset's vertical groups (DataVar.group): this groups vars by
        which source file they share, and a single source file may span vertical groups.

        This is a class method so it can be called by RegionJob factory methods.
        """
        return [data_vars]

    @classmethod
    def num_variable_groups(cls, data_vars: Sequence[DATA_VAR]) -> int:
        """Number of variable groups produced by get_jobs for a given set of data_vars."""
        if cls.max_vars_per_job is None:
            return 1
        return len(
            split_groups(cls.source_file_var_groups(data_vars), cls.max_vars_per_job)
        )

    def get_processing_region(self) -> slice:
        """
        Return a slice of integer offsets into self.template_ds along self.append_dim that identifies
        the region to process. In most cases this is exactly self.region, but if additional data outside
        the region is required, for example for correct interpolation or deaccumulation, this method can
        return a modified slice (e.g. `slice(self.region.start - 1, self.region.stop + 1)`).
        """
        return self.region

    def _flat_job_dataset(self) -> xr.Dataset:
        """This job's data vars gathered from the template tree into one flat Dataset,
        keyed by bare variable name (unique within a job's source-file var group)."""
        paths = {var.path for var in self.data_vars}
        arrays = {
            data_array.name: data_array
            for path, data_array in walk_data_arrays(self.template_ds)
            if path in paths
        }
        assert len(arrays) == len(paths), (
            f"data var names collide across groups within one job: {sorted(paths)}"
        )
        return xr.Dataset(arrays)

    def generate_source_file_coords(
        self, processing_region_ds: xr.Dataset, data_var_group: Sequence[DATA_VAR]
    ) -> Sequence[SOURCE_FILE_COORD]:
        """Return a sequence of coords, one for each source file required to process the data covered by processing_region_ds."""
        raise NotImplementedError(
            "Return a sequence of SourceFileCoord objects, one for each source file required to process the data covered by processing_region_ds."
        )

    def update_template_with_results(
        self, process_results: Mapping[str, Sequence[SourceFileResult]]
    ) -> xr.DataTree:
        """
        Update template dataset based on processing results. This method is called
        during operational updates.

        Subclasses should implement this method to apply dataset-specific adjustments
        based on the processing results. Examples include:
        - Trimming dataset along append_dim to only include successfully processed data
        - Loading existing coordinate values from the primary store and updating them based on results
        - Updating metadata based on what was actually processed vs what was planned

        The default implementation here trims along append_dim to end at the most recent
        successfully processed time.

        Parameters
        ----------
        process_results : Mapping[str, Sequence[SourceFileResult]]
            Mapping from variable names to their SourceFileResult with final processing status.

        Returns
        -------
        xr.DataTree
            Updated template tree reflecting the actual processing results.
        """
        max_append_dim_processed = max(
            (
                c.out_loc[self.append_dim]
                for c in chain.from_iterable(process_results.values())
                if c.status == SourceFileStatus.Succeeded
            ),
            default=None,
        )
        if max_append_dim_processed is None:
            # No data was processed, trim the template to stop before this job's region
            # This is using isel's exclusive slice end behavior
            return self.template_ds.isel(
                {self.append_dim: slice(None, self.region.start)}
            )
        else:
            return self.template_ds.sel(
                {self.append_dim: slice(None, max_append_dim_processed)}
            )

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[DATA_VAR],
        reformat_job_name: str,
    ) -> tuple[Sequence[RegionJob[DATA_VAR, SOURCE_FILE_COORD]], xr.DataTree]:
        """
        Return the sequence of RegionJob instances necessary to update the dataset
        from its current state to include the latest available data.

        Also return the template_ds, expanded along append_dim through the end of
        the data to process. The dataset returned here may extend beyond the
        available data at the source, in which case `update_template_with_results`
        will trim the dataset to the actual data processed.

        The exact logic is dataset-specific, but it generally follows this pattern:
        1. Figure out the range of time to process: append_dim_start (inclusive) and append_dim_end (exclusive)
            a. Read existing data from the primary store to determine what's already processed
            b. Optionally identify recent incomplete/non-final data for reprocessing
        2. Call get_template_fn(append_dim_end) to get the template_ds
        3. Create RegionJob instances by calling cls.get_jobs(..., filter_start=append_dim_start)

        Parameters
        ----------
        primary_store : Store
            The primary store to read existing data from and write updates to.
        tmp_store : Path
            The temporary Zarr store to write into while processing.
        get_template_fn : Callable[[DatetimeLike], xr.DataTree]
            Function to get the template_ds for the operational update.
        append_dim : AppendDim
            The dimension along which data is appended (e.g., "time").
        all_data_vars : Sequence[DATA_VAR]
            Sequence of all data variable configs for this dataset.
        reformat_job_name : str
            The name of the reformatting job, used for progress tracking.
            This is often the name of the Kubernetes job, or "local".

        Returns
        -------
        Sequence[RegionJob[DATA_VAR, SOURCE_FILE_COORD]]
            RegionJob instances that need processing for operational updates.
        xr.DataTree
            The template_ds for the operational update.
        """
        raise NotImplementedError(
            "Subclasses implement operational_update_jobs() with dataset-specific logic"
        )

    # ----- Most subclasses will not need to override the attributes and methods below -----

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True, frozen=True, strict=True
    )

    @computed_field
    @property
    def dataset_id(self) -> str:
        return str(self.template_ds.attrs["dataset_id"])

    @classmethod
    def get_jobs(
        cls,
        tmp_store: Path,
        template_ds: xr.DataTree,
        append_dim: AppendDim,
        all_data_vars: Sequence[DATA_VAR],
        reformat_job_name: str,
        filter_start: Timestamp | None = None,
        filter_end: Timestamp | None = None,
        filter_contains: list[Timestamp] | None = None,
        filter_variable_names: list[str] | None = None,
    ) -> Sequence[Self]:
        """
        Return a sequence of RegionJob instances to process.

        If any of the `filter_*` arguments are provided, the returned jobs are filtered
        to only include jobs which intersect all provided filters. Complete jobs are always
        returned, so regions may extend outside `filter_start` and `filter_end` and include
        values in addition to those in `filter_contains`.

        Parameters
        ----------
        tmp_store : Path
            The temporary Zarr store to write into while processing.
        template_ds : xr.DataTree
            Template tree defining structure and metadata (one node per group).
        append_dim : AppendDim
            The dimension along which data is appended (e.g., "time").
        all_data_vars : Sequence[DATA_VAR]
            Sequence of all data variable configs for this dataset.
            Provided so that grouping and RegionJob made access DataVar.internal_attrs.
        reformat_job_name : str
            The name of the reformatting job, used for progress tracking.
            This is often the name of the Kubernetes job, or "local".
        filter_start : Timestamp | None, default None
            Keep shards where max(shard_coords) >= filter_start. Inclusive.
        filter_end : Timestamp | None, default None
            Keep shards where min(shard_coords) < filter_end. Exclusive.
        filter_contains : list[Timestamp] | None, default None
            Keep only shards containing at least one of the specified timestamps.
            Timestamps must exactly match coordinate values. Empty list returns no jobs.
        filter_variable_names : list[str] | None, default None
            Keep only the specified variables, matched by bare name or by var.path
            (a bare name selects that var in every group). If None, all are included.

        Returns
        -------
        Sequence[Self]
            All RegionJob instances matching the filters. Worker partitioning
            is handled by the caller via get_worker_jobs.
        """

        # Data variables -- filter and group
        template_arrays = list(walk_data_arrays(template_ds))
        assert {v.path for v in all_data_vars} == {path for path, _ in template_arrays}

        data_vars: Sequence[DATA_VAR]
        if filter_variable_names:
            data_vars = [
                v
                for v in all_data_vars
                if v.name in filter_variable_names or v.path in filter_variable_names
            ]
        else:
            data_vars = all_data_vars

        if cls.max_vars_per_job is not None:
            # Split by source file var groups first as those are efficient groupings,
            # then split further to create smaller jobs for parallelism.
            data_var_groups = cls.source_file_var_groups(data_vars)
            data_var_groups = split_groups(data_var_groups, cls.max_vars_per_job)
        else:
            data_var_groups = [data_vars]

        # Regions along append dimension.
        # Materialized arrays partition by shard; virtual arrays use chunks. All data
        # vars must agree: a mix would partition by whichever var is first, and a
        # chunk-sized region over a sharded var would let multiple workers write the
        # same shard. The tree-walking twin of iterating.dimension_slices (whose flat
        # Dataset can't hold same-named vars from different groups).
        sharded = {da.encoding.get("shards") is not None for _, da in template_arrays}
        assert len(sharded) == 1, "all data vars must agree on whether they are sharded"
        kind = "shards" if sharded.pop() else "chunks"
        append_chunk_sizes = {
            da.encoding[kind][da.dims.index(append_dim)] for _, da in template_arrays
        }
        assert len(append_chunk_sizes) == 1, (
            f"Inconsistent {kind} sizes along {append_dim}: {append_chunk_sizes}"
        )
        regions = chunk_slices(
            len(template_ds.coords[append_dim]), append_chunk_sizes.pop()
        )

        # Filter regions by time
        if (
            filter_start is not None
            or filter_end is not None
            or filter_contains is not None
        ):
            coord_values = template_ds.coords[append_dim].values
            shard_size = regions[0].stop - regions[0].start
            n_shards = len(regions)

            # Validate assumptions required for binary search
            assert len(coord_values) > 1, f"{append_dim} must have > 1 coordinate"
            assert np.all(coord_values[1:] > coord_values[:-1]), (
                f"{append_dim} coordinates must be sorted ascending"
            )
            assert all(r.stop - r.start == shard_size for r in regions[:-1]), (
                "all shards except last must have uniform size"
            )
            assert all(r1.stop == r2.start for r1, r2 in pairwise(regions)), (
                "shards must be contiguous"
            )

            start_shard = 0
            end_shard = n_shards

            if filter_start is not None:
                filter_start_np = np.array(filter_start, dtype=coord_values.dtype)
                idx = int(np.searchsorted(coord_values, filter_start_np, side="left"))
                start_shard = min(idx // shard_size, n_shards)

            if filter_end is not None:
                filter_end_np = np.array(filter_end, dtype=coord_values.dtype)
                idx = int(np.searchsorted(coord_values, filter_end_np, side="left"))
                end_shard = min((idx - 1) // shard_size + 1, n_shards) if idx > 0 else 0

            regions = regions[start_shard:end_shard]

            if filter_contains is not None:
                coord_index = pd.Index(coord_values)
                indices = coord_index.get_indexer(pd.Index(filter_contains))
                valid_indices = indices[indices >= 0]
                shard_indices = set(valid_indices // shard_size)
                regions = [
                    r
                    for i, r in enumerate(regions, start=start_shard)
                    if i in shard_indices
                ]

        # Spread regions so any contiguous worker-index window (the workers
        # running concurrently) covers the whole append dim, not a clustered band.
        regions = spread_evenly(regions)

        all_jobs = [
            cls(
                tmp_store=tmp_store,
                template_ds=template_ds,
                data_vars=data_var_group,
                append_dim=append_dim,
                region=region,
                reformat_job_name=reformat_job_name,
            )
            for region in regions
            for data_var_group in data_var_groups
        ]
        return all_jobs

    @classmethod
    def process_worker_jobs(
        cls,
        worker_jobs: Sequence[RegionJob[DATA_VAR, SOURCE_FILE_COORD]],
        store_factory: storage.StoreFactory,
        branch_name: str,
        worker_index: int,
    ) -> dict[str, list[SourceFileResult]]:
        """Process one worker's region jobs against ``branch_name`` and
        return the per-variable source file results.

        Materialized and virtual datasets each own their store/session lifecycle and
        commit cadence behind this one call (see "The worker-processing seam" in
        docs/parallel_processing.md). Callers must pass at least one job.
        """
        raise NotImplementedError(
            "Subclasses implement process_worker_jobs with variant-specific logic"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(region=({self.region.start}, {self.region.stop}), data_vars={[v.name for v in self.data_vars]})"


class SourceFileResult(FrozenBaseModel):
    """
    Per SourceFileCoord result passed to `update_template_with_results`.

    This class is distinct from SourceFileCoord to support JSON serialization.
    Each worker writes its own serialized results, which are then read back
    and merged by the last worker using `parallel_coordination.collect_results`
    before being passed into `update_template_with_results`.
    """

    status: SourceFileStatus
    out_loc: dict[str, Any]
    url: str

    @field_serializer("out_loc")
    def _ser_out_loc(self, v: dict[str, Any]) -> dict[str, Any]:
        return {k: self._ser_out_loc_value(val) for k, val in v.items()}

    @field_validator("out_loc", mode="before")
    @classmethod
    def _val_out_loc(cls, v: Any) -> Any:  # noqa: ANN401
        if not isinstance(v, dict):
            return v
        return {k: cls._val_out_loc_value(val) for k, val in v.items()}

    @staticmethod
    def _ser_out_loc_value(v: Any) -> Any:  # noqa: ANN401
        # pd.Timestamp / pd.Timedelta values are round-tripped via tagged dicts
        # so JSON deserialization reconstructs the original pandas type.
        if isinstance(v, pd.Timestamp):
            return {"__t": "ts", "v": v.isoformat()}
        if isinstance(v, pd.Timedelta):
            return {"__t": "td", "v": v.isoformat()}
        return v

    @staticmethod
    def _val_out_loc_value(v: Any) -> Any:  # noqa: ANN401
        if isinstance(v, dict) and "__t" in v:
            if v["__t"] == "ts":
                return pd.Timestamp(v["v"])
            if v["__t"] == "td":
                return pd.Timedelta(v["v"])
        return v
