from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from gribberish.zarr import GribberishCodec
from zarr.core.metadata import ArrayV3Metadata

from reformatters.common.template_config import TemplateConfig
from reformatters.common.template_utils import write_metadata
from reformatters.common.types import (
    ROOT,
    AppendDim,
    DatetimeLike,
    Dims,
    Timedelta,
    Timestamp,
)
from reformatters.common.virtual_retention import (
    retention_start,
    store_template,
    trim_retention,
)
from tests.common.virtual_region_job_test import _create_template_ds

START = pd.Timestamp("2024-01-01")
FREQ = pd.Timedelta("6h")


class RetentionConfig(TemplateConfig[Any]):
    dims: Dims = {ROOT: ("init_time", "lead_time", "latitude", "longitude")}
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = START
    append_dim_frequency: Timedelta = FREQ
    grouped: bool = False
    ordered_axis: bool = False
    extra_var: bool = False
    sharded_coords: bool = False

    def get_template(
        self, end_time: DatetimeLike, *, start_time: DatetimeLike | None = None
    ) -> xr.DataTree:
        start = (
            self.append_dim_start if start_time is None else pd.Timestamp(start_time)
        )
        labels = pd.date_range(
            start, end_time, freq=self.append_dim_frequency, inclusive="left"
        )
        ds = _create_template_ds(len(labels)).to_dataset()
        encodings = {name: var.encoding.copy() for name, var in ds.coords.items()}
        ds = ds.assign_coords(
            init_time=labels,
            valid_time=(
                ("init_time", "lead_time"),
                labels.values[:, None] + ds.lead_time.values[None, :],
            ),
            ingested_forecast_length=("init_time", np.full(len(labels), np.nan)),
        )
        for name, encoding in encodings.items():
            ds[name].encoding = encoding
        ds.ingested_forecast_length.encoding = {
            "dtype": "float64",
            "fill_value": np.nan,
            "chunks": (8,),
        }
        if self.sharded_coords:
            for coord in ds.coords.values():
                if self.append_dim in coord.dims:
                    coord.encoding["shards"] = tuple(
                        2 * size for size in coord.encoding["chunks"]
                    )
        if self.ordered_axis:
            var = ds.temperature_2m.transpose(
                "latitude", "init_time", "lead_time", "longitude"
            )
            var.encoding["chunks"] = (2, 1, 1, 3)
            ds["temperature_2m"] = var
        if self.extra_var:
            ds["wind"] = ds.temperature_2m.copy(deep=False)
        nodes = {"/": ds}
        if self.grouped:
            nodes["/pressure_level"] = ds.copy(deep=False)
        return xr.DataTree.from_dict(nodes)


def _repo(
    tmp_path: Path, config: RetentionConfig, *, size: int = 4
) -> tuple[icechunk.Repository, Path]:
    source = tmp_path / "source.bin"
    blocks = [
        np.full((2, 3), i * 10 + lead, dtype="float64").tobytes()
        for i in range(size)
        for lead in range(2)
    ]
    source.write_bytes(b"".join(blocks))
    container = icechunk.VirtualChunkContainer(
        f"file://{tmp_path}/", icechunk.local_filesystem_store(str(tmp_path))
    )
    repository_config = icechunk.RepositoryConfig.default()
    repository_config.set_virtual_chunk_container(container)
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "repo")),
        config=repository_config,
        authorize_virtual_chunk_access={
            f"file://{tmp_path}/": icechunk.credentials.LocalFileSystemAccess
        },
    )
    session = repo.writable_session("main")
    template = config.get_template(START + size * FREQ)
    write_metadata(
        template,
        session.store,
        mode="w-",
        skip_icechunk_commit=True,
        consolidated=False,
    )
    for node in template.subtree:
        prefix = "" if node.path == "/" else node.path.strip("/") + "/"
        for i in range(size):
            for lead in range(2):
                index = f"0/{i}/{lead}/0" if config.ordered_axis else f"{i}/{lead}/0/0"
                session.store.set_virtual_ref(
                    f"{prefix}temperature_2m/c/{index}",
                    f"file://{source}",
                    offset=(i * 2 + lead) * 48,
                    length=48,
                )
        zarr.open_array(
            session.store, path=f"{prefix}ingested_forecast_length", mode="r+"
        )[:] = np.arange(size) + 1
    session.commit("Initialize test window")
    return repo, source


def _open(repo: icechunk.Repository) -> xr.DataTree:
    return xr.open_datatree(
        repo.readonly_session("main").store,  # ty: ignore[invalid-argument-type]
        engine="zarr",
        chunks=None,
        decode_timedelta=True,
        consolidated=False,
    )


@pytest.mark.parametrize("positions", [1, 3, 12])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("ordered_axis", [False, True])
def test_trim_preserves_labels_references_and_values(
    tmp_path: Path, positions: int, grouped: bool, ordered_axis: bool
) -> None:
    config = RetentionConfig(
        grouped=grouped, ordered_axis=ordered_axis, sharded_coords=True
    )
    size = positions + 2
    repo, _ = _repo(tmp_path, config, size=size)
    before = _open(repo).load()
    session = repo.writable_session("main")
    assert trim_retention(session, config, positions)
    session.commit("Trim", rebase_with=None)
    result = _open(repo)
    for node in result.subtree:
        xr.testing.assert_equal(
            node.to_dataset(),
            before[node.path].to_dataset().isel(init_time=slice(2, None)),
        )
        index = [0, 0, 0, 0]
        assert (
            repo.readonly_session("main").chunk_type(
                f"{node.path.rstrip('/')}/temperature_2m", index
            )
            == icechunk.ChunkType.virtual
        )
    assert (
        retention_start(repo.readonly_session("main").store, config) == START + 2 * FREQ
    )


@pytest.mark.parametrize("positions", [4, 12])
def test_noop_does_not_create_changes(tmp_path: Path, positions: int) -> None:
    config = RetentionConfig()
    repo, _ = _repo(tmp_path, config)
    session = repo.writable_session("main")
    assert not trim_retention(session, config, positions)
    assert not session.has_uncommitted_changes


def test_empty_store_uses_configured_origin(tmp_path: Path) -> None:
    config = RetentionConfig()
    repo, _ = _repo(tmp_path, config, size=0)
    store = repo.readonly_session("main").store
    assert retention_start(store, config) == START
    assert store_template(store, config).sizes["init_time"] == 0
    assert store_template(store, config, end=START + 2 * FREQ).sizes["init_time"] == 2
    assert not trim_retention(repo.writable_session("main"), config, 1)


def test_store_template_stays_bounded_and_never_contracts(tmp_path: Path) -> None:
    config = RetentionConfig()
    repo, _ = _repo(tmp_path, config)
    session = repo.writable_session("main")
    trim_retention(session, config, 1)
    session.commit("Trim")
    store = repo.readonly_session("main").store
    for end, size in [(None, 1), (START, 1), (START + 6 * FREQ, 3)]:
        template = store_template(store, config, end=end)
        assert template.sizes["init_time"] == size
        assert pd.Timestamp(template.init_time.values[0]) == START + 3 * FREQ


@pytest.mark.parametrize("kind", ["size", "duplicate", "grid"])
def test_invalid_alignment_fails_before_mutation(tmp_path: Path, kind: str) -> None:
    config = RetentionConfig(grouped=True)
    repo, _ = _repo(tmp_path, config)
    setup = repo.writable_session("main")
    if kind == "size":
        array = zarr.open_array(setup.store, path="temperature_2m", mode="r+")
        array.resize((3, *array.shape[1:]))
    elif kind in ("duplicate", "grid"):
        path = "pressure_level/init_time" if kind == "duplicate" else "init_time"
        array = zarr.open_array(setup.store, path=path, mode="r+")
        array[1] = array[0]
    setup.commit("Invalid alignment")
    session = repo.writable_session("main")
    with pytest.raises(AssertionError):
        trim_retention(session, config, 3)
    assert not session.has_uncommitted_changes


def test_trim_does_not_read_payloads_or_encode_grib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = RetentionConfig.get_template

    def codec_template(
        self: RetentionConfig,
        end_time: DatetimeLike,
        *,
        start_time: DatetimeLike | None = None,
    ) -> xr.DataTree:
        tree = original(self, end_time, start_time=start_time)
        tree.temperature_2m.encoding["serializer"] = GribberishCodec(
            var="TMP"
        ).to_dict()
        return tree

    monkeypatch.setattr(RetentionConfig, "get_template", codec_template)
    config = RetentionConfig()
    repo, source = _repo(tmp_path, config)
    source.unlink()
    session = repo.writable_session("main")
    assert trim_retention(session, config, 1)
    session.commit("Trim without source")
    assert (
        session.chunk_type("/temperature_2m", [0, 0, 0, 0])
        == icechunk.ChunkType.virtual
    )
    array = zarr.open_array(repo.readonly_session("main").store, path="temperature_2m")
    assert isinstance(array.metadata, ArrayV3Metadata)
    assert any(isinstance(codec, GribberishCodec) for codec in array.metadata.codecs)


def test_nonunit_append_chunks_fail_before_mutation(tmp_path: Path) -> None:
    config = RetentionConfig()
    repo, _ = _repo(tmp_path, config)
    setup = repo.writable_session("main")
    zarr.open_group(setup.store, mode="r+").create_array(
        "temperature_2m",
        shape=(4, 2, 2, 3),
        chunks=(2, 1, 2, 3),
        dtype="float64",
        dimension_names=("init_time", "lead_time", "latitude", "longitude"),
        overwrite=True,
    )
    setup.commit("Nonunit chunks")
    session = repo.writable_session("main")
    with pytest.raises(AssertionError, match="unit append chunks"):
        trim_retention(session, config, 3)
    assert not session.has_uncommitted_changes
