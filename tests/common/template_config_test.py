import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
import zarr.core.sync
from zarr.core.buffer import default_buffer_prototype

from reformatters.common import template_utils

# Use a dummy DataVar that is a subtype of BaseInternalAttrs, as required by the type var
from reformatters.common.config_models import (
    ROOT,
    BaseInternalAttrs,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVar,
    DataVarAttrs,
    Encoding,
)
from reformatters.common.storage import StoreFactory
from reformatters.common.template_config import (
    SPATIAL_REF_COORDS,
    TemplateConfig,
)
from reformatters.common.types import (
    AppendDim,
    DatetimeLike,
    Dims,
    Timedelta,
    Timestamp,
)
from reformatters.common.virtual_region_job import VirtualRegionJob
from reformatters.common.zarr import assert_fill_values_set


class ExampleDataVar(DataVar[BaseInternalAttrs]):
    pass


class ExampleCoordinate(Coordinate):
    pass


class ExampleDatasetAttributes(DatasetAttributes):
    pass


class ExampleConfig(TemplateConfig[ExampleDataVar]):
    """A minimal concrete implementation to test the happy-path logic."""

    dims: Dims = {ROOT: ("time",)}
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = pd.Timestamp("2000-01-01")
    append_dim_frequency: Timedelta = pd.Timedelta(days=1)

    @property
    def dataset_attributes(self) -> ExampleDatasetAttributes:
        return SimpleNamespace(dataset_id="simple_dataset")  # ty: ignore[invalid-return-type]

    @property
    def coords(self) -> list[Coordinate]:
        # no extra coords beyond dims
        return []

    @property
    def data_vars(self) -> list[ExampleDataVar]:
        return []

    def dimension_coordinates(self) -> dict[str, pd.DatetimeIndex]:
        # not used in these tests
        return {"time": self.append_dim_coordinates(self.append_dim_start)}

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        # exercise the base-class fallback (which only adds spatial_ref)
        return super().derive_coordinates(ds)


class BadCoordsConfig(ExampleConfig):
    """Injects a coord whose name isn't in dims to trigger the NotImplementedError."""

    @property
    def coords(self) -> list[Coordinate]:
        # name "bad" is not in self.dims
        return [SimpleNamespace(name="bad")]  # ty: ignore[invalid-return-type]


@pytest.fixture
def example_config() -> ExampleConfig:
    return ExampleConfig(
        append_dim="time",
        append_dim_start=pd.Timestamp("2000-01-01"),
        append_dim_frequency=pd.Timedelta(days=1),
    )


def test_dataset_id_property(example_config: ExampleConfig) -> None:
    assert example_config.dataset_id == "simple_dataset"


def test_append_dim_coordinates_left_inclusive_right_exclusive(
    example_config: ExampleConfig,
) -> None:
    # up to but not including 2000-01-05
    end = pd.Timestamp("2000-01-05")
    got = example_config.append_dim_coordinates(end)
    expected = pd.date_range("2000-01-01", end, freq="1D", inclusive="left")
    pd.testing.assert_index_equal(got, expected)


@pytest.mark.parametrize(
    ("start_year", "expected_years"),
    [
        (2000, max(2025 - 2000 + 15, 10)),
        (2024, max(2025 - 2024 + 15, 10)),
        (2030, max(2025 - 2030 + 15, 10)),
    ],
)
def test_append_dim_coordinate_chunk_size_varies_with_start(
    start_year: int, expected_years: int
) -> None:
    class C(ExampleConfig):
        append_dim_start: Timestamp = pd.Timestamp(f"{start_year}-01-01")

    inst = C(
        append_dim="time",
        append_dim_start=pd.Timestamp(f"{start_year}-01-01"),
        append_dim_frequency=pd.Timedelta(days=1),
    )
    # total days = 365 * expected_years, freq = 1 day
    result: float = pd.Timedelta(days=365 * expected_years) / inst.append_dim_frequency
    expected = int(result)
    assert inst.append_dim_coordinate_chunk_size() == expected


def test_default_derive_coordinates_returns_spatial_ref(
    example_config: ExampleConfig,
) -> None:
    ds = xr.Dataset()
    coords = example_config.derive_coordinates(ds)
    # only the spatial_ref key should be present
    assert set(coords) == {"spatial_ref"}
    assert coords["spatial_ref"] == SPATIAL_REF_COORDS


def test_derive_coordinates_raises_if_coords_not_returned() -> None:
    bad = BadCoordsConfig(
        append_dim="time",
        append_dim_start=pd.Timestamp("2000-01-01"),
        append_dim_frequency=pd.Timedelta(days=1),
    )
    with pytest.raises(
        NotImplementedError,
        match=r"Coordinates {'bad'} are defined.*derive_coordinates",
    ):
        bad.derive_coordinates(xr.Dataset())


STATISTICS = ["mean", "p10", "p25", "p50", "p75", "p90"]


class StringCoordinateConfig(TemplateConfig[DataVar[BaseInternalAttrs]]):
    dims: Dims = {ROOT: ("time", "statistic")}
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = pd.Timestamp("2024-01-01", tz="UTC")
    append_dim_frequency: Timedelta = pd.Timedelta(days=1)
    path: Path

    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="test-string-coordinate",
            dataset_version="1",
            name="String coordinate test",
            description="Test string dimension coordinates.",
            attribution="Synthetic data.",
            license="CC-BY-4.0",
            spatial_domain="Global",
            spatial_resolution="0.25 degrees (~20km)",
            time_domain="2024 onwards",
            time_resolution="Daily",
        )

    @property
    def coords(self) -> list[Coordinate]:
        return [
            Coordinate(
                name="time",
                encoding=Encoding(
                    dtype="int64",
                    chunks=100,
                    shards=None,
                    fill_value=0,
                    units="seconds since 1970-01-01 00:00:00",
                    calendar="proleptic_gregorian",
                ),
                attrs=CoordinateAttrs(units=None, statistics_approximate=None),
            ),
            Coordinate(
                name="statistic",
                encoding=Encoding(dtype="str", chunks=6, shards=None, fill_value=""),
                attrs=CoordinateAttrs(units=None, statistics_approximate=None),
            ),
        ]

    @property
    def data_vars(self) -> list[DataVar[BaseInternalAttrs]]:
        return [
            DataVar(
                name="temperature",
                encoding=Encoding(
                    dtype="float32", chunks=(1, 1), shards=None, fill_value=np.nan
                ),
                attrs=DataVarAttrs(
                    long_name="Temperature",
                    short_name="t",
                    units="K",
                    step_type="instant",
                ),
                internal_attrs=BaseInternalAttrs(keep_mantissa_bits="no-rounding"),
            )
        ]

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            "time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "statistic": STATISTICS,
        }

    def derive_coordinates(
        self,
        ds: xr.Dataset,  # noqa: ARG002
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        return {}

    def append_dim_coordinates(self, end: DatetimeLike) -> pd.DatetimeIndex:
        return super().append_dim_coordinates(end).tz_localize(None)

    def template_path(self) -> Path:
        return self.path


@pytest.fixture
def string_coordinate_config(tmp_path: Path) -> StringCoordinateConfig:
    config = StringCoordinateConfig(path=tmp_path / "latest.zarr")
    config.update_template()
    return config


def test_string_coordinate_template_roundtrip(
    string_coordinate_config: StringCoordinateConfig, tmp_path: Path
) -> None:
    config = string_coordinate_config
    metadata = json.loads((config.path / "statistic/zarr.json").read_text())
    assert metadata["data_type"] == "string"
    assert metadata["fill_value"] == ""
    assert metadata["codecs"][0] == {"name": "vlen-utf8", "configuration": {}}
    ds = xr.open_zarr(config.path)
    assert ds.sel(statistic="p50").statistic.item() == "p50"
    assert ds.statistic.values.tolist() == STATISTICS
    template = config.get_template(pd.Timestamp("2024-01-03", tz="UTC"))
    assert_fill_values_set(template)
    template_utils.write_metadata(template, tmp_path / "roundtrip.zarr")
    assert (
        json.loads((tmp_path / "roundtrip.zarr/statistic/zarr.json").read_text())
        == metadata
    )
    template_utils.assert_no_structural_drift_from_existing_store(
        template, xr.open_datatree(config.path), "time"
    )
    in_memory = xr.DataTree(
        config._build_node_dataset(ROOT, config.dimension_coordinates())
    )
    template_utils.assert_no_structural_drift_from_existing_store(
        in_memory, xr.open_datatree(config.path), "time"
    )


@pytest.mark.parametrize(
    "labels", [STATISTICS[::-1], [*STATISTICS[:-1], "p99"], STATISTICS[:-1]]
)
def test_string_coordinate_drift_rejected(
    string_coordinate_config: StringCoordinateConfig, labels: list[str]
) -> None:
    config = string_coordinate_config
    existing = xr.open_datatree(config.path)
    changed = existing.to_dataset().reindex(statistic=labels)
    changed.statistic.encoding = existing.statistic.encoding.copy()
    with pytest.raises(ValueError, match="coord statistic: values differ"):
        template_utils.assert_no_structural_drift_from_existing_store(
            xr.DataTree(changed), existing, "time"
        )


def test_string_coordinate_icechunk_append_and_refresh(
    string_coordinate_config: StringCoordinateConfig, tmp_path: Path
) -> None:
    config = string_coordinate_config
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    initial = config.get_template(pd.Timestamp("2024-01-02", tz="UTC"))
    session = repo.writable_session("main")
    template_utils.write_metadata(initial, session.store, mode="w-", consolidated=False)
    template = config.get_template(pd.Timestamp("2024-01-04", tz="UTC"))
    job = VirtualRegionJob(
        tmp_store=tmp_path / "unused.zarr",
        template_ds=template,
        data_vars=config.data_vars,
        append_dim="time",
        region=slice(0, 3),
        reformat_job_name="test",
        processing_mode="update",
    )
    session = repo.writable_session("main")
    before = zarr.core.sync.sync(
        session.store.get("statistic/c/0", prototype=default_buffer_prototype())
    )
    assert before is not None
    with patch.object(session.store, "set", wraps=session.store.set) as writes:
        job.sync_dims_to([session.store], 3)
        assert not any(
            call.args[0].startswith("statistic/") for call in writes.call_args_list
        )
    after = zarr.core.sync.sync(
        session.store.get("statistic/c/0", prototype=default_buffer_prototype())
    )
    assert after is not None
    assert before.to_bytes() == after.to_bytes()
    session.commit("Append time")
    reopened = xr.open_datatree(
        repo.readonly_session("main").store,  # ty: ignore[invalid-argument-type]
        engine="zarr",
        consolidated=False,
    )
    assert reopened.sizes["time"] == 3
    assert reopened.statistic.values.tolist() == STATISTICS
    template_utils.assert_no_structural_drift_from_existing_store(
        template, reopened, "time"
    )
    for index, label in enumerate(STATISTICS):
        assert job.chunk_key(
            {"time": pd.Timestamp("2024-01-03"), "statistic": label},
            config.data_vars[0],
        ) == (2, index)
    assert (
        job.chunk_key(
            {"time": pd.Timestamp("2024-01-03"), "statistic": "p99"},
            config.data_vars[0],
        )
        is None
    )
    factory = Mock(spec=StoreFactory)
    factory.open_primary_datatree.return_value = reopened
    factory.replica_stores.return_value = []
    refresh_session = repo.writable_session("main")
    factory.primary_store.return_value = refresh_session.store
    job.refresh_metadata(factory, tmp_path / "refresh.zarr")
    assert not refresh_session.has_uncommitted_changes
    assert repo.readonly_session("main").snapshot_id == session.snapshot_id
    assert (
        xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
        .sel(statistic="p50")
        .statistic.item()
        == "p50"
    )
