from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import computed_field
from zarr.abc.store import Store

from reformatters.common import materialized_region_job, template_utils, validation
from reformatters.common import zarr as zarr_utils
from reformatters.common.config_models import (
    ROOT,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    Encoding,
)
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.iterating import flatten_groups, walk_data_arrays
from reformatters.common.kubernetes import CronJob, ReformatCronJob
from reformatters.common.materialized_region_job import MaterializedRegionJob
from reformatters.common.pydantic import replace
from reformatters.common.region_job import RegionJob, SourceFileCoord
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.common.types import (
    AppendDim,
    ArrayFloat32,
    DatetimeLike,
    Dims,
    Timestamp,
)
from reformatters.common.virtual_region_job import VirtualRegionJob
from tests.common.multi_group_template_test import (
    _DV,
    MultiGroupConfig,
    _coord,
    _data_var,
)


class StaticGroupConfig(MultiGroupConfig):
    dims: Dims = {ROOT: ("init_time", "cell")}
    include_grid_identity: bool = False

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        attrs = super().dataset_attributes
        return (
            replace(attrs, source_grid_name="Native grid", source_grid_id="grid-uuid")
            if self.include_grid_identity
            else attrs
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            "init_time": self.append_dim_coordinates("2020-01-01T06:00"),
            "cell": np.arange(2, dtype=np.int32),
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        time_coord = _coord("init_time", "int64")
        return [
            replace(
                time_coord,
                encoding=replace(
                    time_coord.encoding, units="seconds since 1970-01-01 00:00:00"
                ),
            ),
            _coord("latitude"),
            _coord("longitude"),
            Coordinate(
                name="cell",
                encoding=Encoding(dtype="int32", chunks=2, shards=None, fill_value=-1),
                attrs=CoordinateAttrs(units=None, statistics_approximate=None),
            ),
            Coordinate(
                name="spatial_ref",
                encoding=Encoding(dtype="int64", chunks=(), shards=None, fill_value=-1),
                attrs=CoordinateAttrs(units=None, statistics_approximate=None),
            ),
        ]

    @computed_field
    @property
    def data_vars(self) -> Sequence[_DV]:
        var = _data_var("temperature", ROOT, 2)
        return [
            replace(
                var,
                encoding=replace(var.encoding, chunks=(1, 2), shards=(2, 2)),
                attrs=replace(var.attrs, mesh="/mesh/mesh", location="face"),
            )
        ]

    def derive_coordinates(
        self,
        ds: xr.Dataset,  # noqa: ARG002
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        return {
            "latitude": (("cell",), np.array([0.25, 0.75])),
            "longitude": (("cell",), np.array([0.25, 0.75])),
            "spatial_ref": ((), np.array(0)),
        }

    def static_groups(self) -> dict[str, xr.Dataset]:
        ds = xr.Dataset(
            {
                "mesh": xr.DataArray(
                    np.int32(7),
                    attrs={"cf_role": "mesh_topology", "topology_dimension": 2},
                ),
                "face_node_connectivity": xr.DataArray(
                    np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
                    dims=("cell", "max_nodes_per_face"),
                    attrs={"cf_role": "face_node_connectivity", "start_index": 0},
                ),
            },
            coords={
                "node_latitude": ("node", np.array([0, 1, 0, 1], dtype=np.float32)),
                "node_longitude": ("node", np.array([0, 0, 1, 1], dtype=np.float32)),
            },
            attrs={"Conventions": "UGRID-1.0"},
        )
        for name in ("mesh", "face_node_connectivity"):
            ds[name].encoding = Encoding(
                dtype="int32",
                chunks=ds[name].shape,
                shards=None,
                fill_value=-1,
                compressors=[zarr_utils.BLOSC_4BYTE_ZSTD_LEVEL3_BITSHUFFLE],
            ).model_dump(exclude_none=True)
        for name in ("node_latitude", "node_longitude"):
            ds[name].encoding = Encoding(
                dtype="float32",
                chunks=(4,),
                shards=None,
                fill_value=np.nan,
                compressors=[zarr_utils.BLOSC_4BYTE_ZSTD_LEVEL3_BITSHUFFLE],
            ).model_dump(exclude_none=True)
        return {"mesh": ds}


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StaticGroupConfig:
    monkeypatch.setattr(
        StaticGroupConfig,
        "template_path",
        lambda self: tmp_path / "templates/latest.zarr",
    )
    return StaticGroupConfig()


def _assert_static_values(actual: xr.Dataset, config: StaticGroupConfig) -> None:
    expected = config.static_groups()["mesh"]
    for name in expected.variables:
        xr.testing.assert_identical(actual[name].variable, expected[name].variable)
        assert actual[name].encoding["dtype"] == expected[name].encoding["dtype"]
        assert [c.to_dict() for c in actual[name].encoding["compressors"]] == list(
            expected[name].encoding["compressors"]
        )
    assert actual.attrs["Conventions"] == "UGRID-1.0"


def test_template_expansion_preserves_static_values_and_excludes_them_from_jobs(
    config: StaticGroupConfig, tmp_path: Path
) -> None:
    config.update_template()
    tree = config.get_template("2020-01-02")
    _assert_static_values(tree.children["mesh"].to_dataset(inherit=False), config)
    mesh = xr.open_zarr(config.template_path(), group="mesh", chunks=None)
    assert set(mesh.variables) == set(config.static_groups()["mesh"].variables)
    _assert_static_values(mesh, config)
    assert set(flatten_groups(tree, include_static=False).data_vars) == {"temperature"}
    assert "mesh/mesh" in dict(walk_data_arrays(tree))
    jobs = StaticRegionJob.get_jobs(
        tmp_store=tmp_path,
        template_ds=tree,
        append_dim="init_time",
        all_data_vars=config.data_vars,
        reformat_job_name="test",
    )
    assert [(j.region.start, j.region.stop) for j in jobs] == [(0, 2), (2, 4)]
    assert all([v.path for v in j.data_vars] == ["temperature"] for j in jobs)


@pytest.mark.parametrize("change", ["values", "removed_coordinate", "codec"])
def test_structural_guard_rejects_static_array_drift(
    config: StaticGroupConfig, change: str
) -> None:
    config.update_template()
    existing = config.get_template("2020-01-02")
    updated = config.get_template("2020-01-03")
    mesh = updated.children["mesh"].to_dataset(inherit=False).copy(deep=True)
    match change:
        case "values":
            mesh["face_node_connectivity"].values[0, 0] = 3
        case "removed_coordinate":
            mesh = mesh.drop_vars("node_latitude")
        case "codec":
            mesh["face_node_connectivity"].encoding["compressors"] = ()
    updated["mesh"] = xr.DataTree(mesh)
    with pytest.raises(ValueError, match="mesh/"):
        template_utils.assert_no_structural_drift_from_existing_store(
            updated, existing, "init_time"
        )


def test_get_template_rejects_undeclared_static_group(
    config: StaticGroupConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.update_template()
    monkeypatch.setattr(StaticGroupConfig, "static_groups", lambda self: {})
    with pytest.raises(AssertionError, match="run update-template"):
        config.get_template("2020-01-02")


@pytest.mark.parametrize("append_dim", ["time", "init_time"])
def test_static_group_with_append_array_is_rejected(
    config: StaticGroupConfig, monkeypatch: pytest.MonkeyPatch, append_dim: str
) -> None:
    mesh = config.static_groups()["mesh"].assign(extra=(append_dim, [1]))
    monkeypatch.setattr(StaticGroupConfig, "static_groups", lambda self: {"mesh": mesh})
    with pytest.raises(AssertionError, match="must not include an append dimension"):
        config.update_template()


def test_virtual_dataset_rejects_static_group_before_writing(
    config: StaticGroupConfig,
) -> None:
    with pytest.raises(ValueError, match="not yet implemented for virtual datasets"):
        DynamicalDataset[_DV, StaticSourceFileCoord](
            template_config=config,
            region_job_class=VirtualRegionJob[_DV, StaticSourceFileCoord],
            primary_storage_config=StorageConfig(
                base_path="unused", format=DatasetFormat.ICECHUNK
            ),
        )


class StaticSourceFileCoord(SourceFileCoord):
    init_time: Timestamp

    def get_url(self) -> str:
        return "https://test.org/unused"


class StaticRegionJob(MaterializedRegionJob[_DV, StaticSourceFileCoord]):
    download_parallelism: int = 1
    read_parallelism: int = 1

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,  # noqa: ARG003
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[_DV],
        reformat_job_name: str,
        *,
        job_reference_time: Timestamp,  # noqa: ARG003
    ) -> tuple[Sequence[RegionJob[_DV, StaticSourceFileCoord]], xr.DataTree]:
        template = get_template_fn("2020-01-02")
        return cls.get_jobs(
            tmp_store=tmp_store,
            template_ds=template,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            filter_start=pd.Timestamp("2020-01-01T12:00"),
        ), template

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[_DV],  # noqa: ARG002
    ) -> Sequence[StaticSourceFileCoord]:
        return [
            StaticSourceFileCoord(init_time=pd.Timestamp(t))
            for t in processing_region_ds.init_time.values
        ]

    def download_file(self, coord: StaticSourceFileCoord) -> Path:  # noqa: ARG002
        return Path("unused-source-file")

    def read_data(
        self,
        coord: StaticSourceFileCoord,
        data_var: _DV,  # noqa: ARG002
    ) -> ArrayFloat32:
        return np.array(
            [coord.init_time.hour, coord.init_time.hour + 1], dtype=np.float32
        )


class StaticDataset(DynamicalDataset[_DV, StaticSourceFileCoord]):
    template_config: StaticGroupConfig
    region_job_class: type[StaticRegionJob] = StaticRegionJob

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        return [
            ReformatCronJob(
                name=f"{self.dataset_id}-update",
                schedule="0 0 * * *",
                pod_active_deadline=timedelta(minutes=1),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1",
                memory="1G",
                shared_memory="1G",
                ephemeral_storage="1G",
            )
        ]

    def validators(self) -> Sequence[validation.Validator]:
        return ()


@pytest.mark.parametrize(
    "dataset_format", [DatasetFormat.ICECHUNK, DatasetFormat.ZARR3]
)
def test_materialized_backfill_update_and_refresh_preserve_static_group(
    config: StaticGroupConfig,
    monkeypatch: pytest.MonkeyPatch,
    dataset_format: DatasetFormat,
) -> None:
    monkeypatch.setattr(
        materialized_region_job, "ProcessPoolExecutor", ThreadPoolExecutor
    )
    config.update_template()
    dataset = StaticDataset(
        template_config=config,
        primary_storage_config=StorageConfig(base_path="unused", format=dataset_format),
    )
    dataset.backfill_local(pd.Timestamp("2020-01-01T12:00"))
    before = dataset.store_factory.open_primary_datatree()
    _assert_static_values(before.children["mesh"].to_dataset(inherit=False), config)
    dataset.update("test-update")
    after = dataset.store_factory.open_primary_datatree()
    _assert_static_values(after.children["mesh"].to_dataset(inherit=False), config)
    np.testing.assert_array_equal(
        after["temperature"].values, [[0, 1], [6, 7], [12, 13], [18, 19]]
    )
    xr.testing.assert_identical(
        before.children["mesh"].to_dataset(inherit=False),
        after.children["mesh"].to_dataset(inherit=False),
    )
    template_utils.refresh_store_metadata(
        dataset.store_factory,
        config.get_template("2020-01-03"),
        "init_time",
        dataset._tmp_store(),
        consolidated=True,
    )
    refreshed = dataset.store_factory.open_primary_datatree()
    _assert_static_values(refreshed.children["mesh"].to_dataset(inherit=False), config)
    xr.testing.assert_identical(after, refreshed)


def test_optional_mesh_and_grid_attributes_round_trip_and_omit_none(
    config: StaticGroupConfig,
) -> None:
    config.update_template()
    tree = config.get_template("2020-01-02")
    assert "source_grid_name" not in tree.attrs
    assert "source_grid_id" not in tree.attrs
    assert tree["temperature"].attrs["mesh"] == "/mesh/mesh"
    assert tree["temperature"].attrs["location"] == "face"
    grid_config = replace(config, include_grid_identity=True)
    grid_config.update_template()
    grid_tree = grid_config.get_template("2020-01-02")
    assert grid_tree.attrs["source_grid_name"] == "Native grid"
    assert grid_tree.attrs["source_grid_id"] == "grid-uuid"
