import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import icechunk
import numpy as np
import pytest
import xarray as xr
import zarr
from zarr.api import asynchronous
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import MemoryStore

from reformatters.common import template_utils
from reformatters.common.template_config import TemplateConfig
from reformatters.noaa.refs.forecast_3_hourly_virtual.template_config import (
    NoaaRefsForecast3HourlyVirtualTemplateConfig,
)
from reformatters.noaa.refs.forecast_hourly_virtual.template_config import (
    NoaaRefsForecastHourlyVirtualTemplateConfig,
)
from reformatters.noaa.rrfs.forecast_18_hour_virtual.template_config import (
    NoaaRrfsForecast18HourVirtualTemplateConfig,
)
from reformatters.noaa.rrfs.forecast_84_hour_virtual.template_config import (
    NoaaRrfsForecast84HourVirtualTemplateConfig,
)
from reformatters.noaa.rrfs.forecast_sub_hourly_virtual.template_config import (
    NoaaRrfsForecastSubHourlyVirtualTemplateConfig,
)
from reformatters.noaa.rrfs_ens.forecast_virtual.template_config import (
    NoaaRrfsEnsForecastVirtualTemplateConfig,
)
from scripts import migrate_rrfs_retired_arrays as migration

DATASET = "noaa-rrfs-forecast-18-hour-virtual"
CONFIGS = (
    NoaaRrfsForecast18HourVirtualTemplateConfig(),
    NoaaRrfsForecast84HourVirtualTemplateConfig(),
    NoaaRrfsForecastSubHourlyVirtualTemplateConfig(),
    NoaaRrfsEnsForecastVirtualTemplateConfig(),
    NoaaRefsForecastHourlyVirtualTemplateConfig(),
    NoaaRefsForecast3HourlyVirtualTemplateConfig(),
)


async def _store_bytes(store: icechunk.IcechunkStore) -> dict[str, bytes]:
    values: dict[str, bytes] = {}
    async for key in store.list():
        buffer = await store.get(key, default_buffer_prototype())
        assert buffer is not None
        values[key] = buffer.to_bytes()
    return values


async def _consolidate_metadata(store: icechunk.IcechunkStore) -> None:
    memory = MemoryStore()
    for key, value in (await _store_bytes(store)).items():
        if key.endswith("zarr.json"):
            await memory.set(key, default_buffer_prototype().buffer.from_bytes(value))
    with template_utils.ignore_consolidated_metadata_spec_warning():
        await asynchronous.consolidate_metadata(memory)
    root = await memory.get("zarr.json", default_buffer_prototype())
    assert root is not None
    await store.set("zarr.json", root)


type ChunkRecord = tuple[tuple[int, ...], int, str, int, int, bytes | None]


async def _chunk_records(store: icechunk.IcechunkStore) -> dict[str, list[ChunkRecord]]:
    records: dict[str, list[ChunkRecord]] = {}
    async for key in store.list():
        if not key.endswith("/zarr.json"):
            continue
        metadata = await migration._read_json(store, key)
        if metadata["node_type"] != "array":
            continue
        path = key.removesuffix("/zarr.json")
        records[path] = []
        async for (
            coords,
            kinds,
            locations,
            offsets,
            lengths,
            inlined,
        ) in store.array_chunk_iterator(path):
            records[path].extend(
                (
                    tuple(int(c) for c in coords[i]),
                    int(kinds[i]),
                    location,
                    int(offsets[i]),
                    int(lengths[i]),
                    inlined.get(i),
                )
                for i, location in enumerate(locations)
            )
        records[path].sort()
    return records


def _disk_bytes(path: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()
    }


def _repo(tmp_path: Path) -> icechunk.Repository:
    container = icechunk.VirtualChunkContainer(
        tmp_path.as_uri() + "/", icechunk.local_filesystem_store(str(tmp_path))
    )
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(container)
    return icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "repo")),
        config=config,
        authorize_virtual_chunk_access={
            container.url_prefix: icechunk.credentials.LocalFileSystemAccess
        },
    )


def _add_retired(group: zarr.Group, dataset: str) -> None:
    for path in migration.RETIRED_ARRAYS[dataset]:
        dims = ("init_time", "lead_time", "y", "x")
        if dataset == "noaa-rrfs-ens-forecast-virtual":
            dims = ("init_time", "ensemble_member", "lead_time", "y", "x")
        elif path == "composite_reflectivity":
            dims = ("init_time", "statistic", "lead_time", "y", "x")
        group.create_array(
            path,
            shape=(2,) * len(dims),
            chunks=(1,) * len(dims),
            dtype="float64",
            dimension_names=dims,
            compressors=None,
        )


def _simple_store(tmp_path: Path, *, consolidated: bool = False) -> icechunk.Repository:
    repo = _repo(tmp_path)
    session = repo.writable_session("main")
    group = zarr.group(
        session.store, attributes={"dataset_id": DATASET, "extent": "keep"}
    )
    _add_retired(group, DATASET)
    group.create_array(
        "pressure_level/retained",
        data=np.arange(4, dtype="int32"),
        chunks=(2,),
        dimension_names=("pressure_level",),
        attributes={"comment": "keep", "units": "K"},
    )
    if consolidated:
        asyncio.run(_consolidate_metadata(session.store))
    session.commit("old schema")
    return repo


def _tiny_template(config: TemplateConfig[Any]) -> xr.DataTree:
    coords = {dim: values[:2] for dim, values in config.dimension_coordinates().items()}
    coords[config.append_dim] = config.append_dim_coordinates(
        config.append_dim_start + 2 * config.append_dim_frequency
    )
    nodes = {
        config._group_node_path(group): config._build_node_dataset(group, coords)
        for group in config.groups
    }
    tree = xr.DataTree.from_dict(nodes)
    for node in tree.subtree:
        for array in node.variables.values():
            array.encoding["chunks"] = (1,) * array.ndim
            array.encoding["shards"] = None
    return tree


def _assert_retained_bytes(
    before: dict[str, bytes],
    after: dict[str, bytes],
    paths: tuple[str, ...],
    consolidated: bool,
) -> None:
    removed = {key for key in before if key.split("/")[0] in paths}
    assert set(after) == set(before) - removed
    for key, value in after.items():
        if key == "zarr.json" and consolidated:
            old_root = json.loads(before[key])
            for path in paths:
                del old_root["consolidated_metadata"]["metadata"][path]
            assert json.loads(value) == old_root
        else:
            assert value == before[key], key


@pytest.mark.parametrize("config", CONFIGS, ids=lambda config: config.dataset_id)
@pytest.mark.parametrize("consolidated", [False, True])
def test_old_schema_migration_preserves_retained_data_and_passes_new_template_guard(
    tmp_path: Path, config: TemplateConfig[Any], consolidated: bool
) -> None:
    dataset = config.dataset_id
    paths = migration.RETIRED_ARRAYS[dataset]
    assert {var.path for var in config.data_vars}.isdisjoint(paths)
    template = _tiny_template(config)
    repo = _repo(tmp_path)
    session = repo.writable_session("main")
    template_utils.write_metadata(
        template,
        session.store,
        mode="w-",
        skip_icechunk_commit=True,
        consolidated=False,
    )
    _add_retired(zarr.open_group(session.store, use_consolidated=False), dataset)
    message = tmp_path / "virtual.bin"
    message.write_bytes(b"padding!" + np.array([123.0], dtype="float64").tobytes())
    virtual_paths = [
        *(var.path for var in config.data_vars),
        *paths,
    ]
    for path in virtual_paths:
        ndim = len(zarr.open_array(session.store, path=path).shape)
        session.store.set_virtual_ref(
            path + "/c/" + "/".join("0" for _ in range(ndim)),
            message.as_uri(),
            offset=8,
            length=8,
        )
    if consolidated:
        asyncio.run(_consolidate_metadata(session.store))
    original = session.commit("old schema with virtual references")
    before = asyncio.run(_store_bytes(session.store))
    references = asyncio.run(_chunk_records(session.store))
    before_files = _disk_bytes(tmp_path)
    with (
        xr.open_datatree(session.store, engine="zarr", decode_timedelta=True) as old,  # ty: ignore[invalid-argument-type]
        pytest.raises(ValueError, match="missing from update template"),
    ):
        template_utils.assert_no_structural_drift_from_existing_store(
            template, old, "init_time"
        )

    plan = migration.migrate(repo, dataset, original)
    assert plan.status == "planned"
    assert _disk_bytes(tmp_path) == before_files
    result = migration.migrate(repo, dataset, original, apply=True)
    assert result.status == "applied"
    assert repo.lookup_tag(result.historical_tag) == original
    assert repo.lookup_snapshot(result.snapshot).parent_id == original
    assert len(list(repo.ancestry(branch="main"))) == 3
    current = repo.readonly_session("main")
    after = asyncio.run(_store_bytes(current.store))
    _assert_retained_bytes(before, after, paths, consolidated)
    assert asyncio.run(_chunk_records(current.store)) == {
        path: refs for path, refs in references.items() if path not in paths
    }
    for path in virtual_paths:
        if path not in paths:
            assert current.get_node_id("/" + path) == session.get_node_id("/" + path)
            ndim = len(zarr.open_array(current.store, path=path).shape)
            assert (
                current.chunk_type("/" + path, [0] * ndim) == icechunk.ChunkType.virtual
            )
    with xr.open_datatree(
        current.store,  # ty: ignore[invalid-argument-type]
        engine="zarr",
        decode_timedelta=True,
    ) as migrated:
        template_utils.assert_no_structural_drift_from_existing_store(
            template, migrated, "init_time"
        )
    historical = repo.readonly_session(tag=result.historical_tag)
    assert asyncio.run(_store_bytes(historical.store)) == before
    retired = zarr.open_array(historical.store, path=paths[0])
    assert retired[(0,) * retired.ndim] == 123.0
    assert set(historical.all_virtual_chunk_locations()) == {message.as_uri()}
    completed_files = _disk_bytes(tmp_path)
    assert (
        migration.migrate(repo, dataset, original, apply=True).status
        == "already-applied"
    )
    assert migration.migrate(repo, dataset, original).status == "already-applied"
    assert _disk_bytes(tmp_path) == completed_files


@pytest.mark.parametrize("consolidated", [False, True])
def test_local_cli_defaults_to_read_only_and_retains_readable_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, consolidated: bool
) -> None:
    repo = _simple_store(tmp_path, consolidated=consolidated)
    original = repo.lookup_branch("main")
    args = [
        "migrate_rrfs_retired_arrays.py",
        "--dataset",
        DATASET,
        "--repo",
        (tmp_path / "repo").as_uri(),
        "--expected-snapshot",
        original,
    ]
    monkeypatch.setattr(
        migration, "open_repo", Mock(side_effect=AssertionError("No auth"))
    )
    monkeypatch.setattr(sys, "argv", args)
    before = _disk_bytes(tmp_path)
    migration.main()
    assert _disk_bytes(tmp_path) == before
    monkeypatch.setattr(sys, "argv", [*args, "--apply"])
    migration.main()
    reopened = migration.open_repository(
        (tmp_path / "repo").as_uri(), auth_from_env=False
    )
    array = zarr.open_array(
        reopened.readonly_session("main").store, path="pressure_level/retained"
    )
    np.testing.assert_array_equal(array[:], np.arange(4))
    assert array.attrs.asdict() == {"comment": "keep", "units": "K"}


@pytest.mark.parametrize(
    "case",
    ["unauthorized", "identity", "missing", "group", "dims", "consolidated", "stale"],
)
def test_refuses_wrong_dataset_schema_or_head_without_mutations(
    tmp_path: Path, case: str
) -> None:
    repo = _simple_store(tmp_path)
    expected = repo.lookup_branch("main")
    dataset = DATASET
    if case == "unauthorized":
        dataset = "noaa-gfs-forecast"
    elif case == "identity":
        dataset = "noaa-rrfs-forecast-84-hour-virtual"
    else:
        session = repo.writable_session("main")
        group = zarr.open_group(session.store)
        path = migration.RETIRED_ARRAYS[DATASET][0]
        if case in ("missing", "group"):
            del group[path]
            if case == "group":
                group.create_group(path)
        elif case == "dims":
            metadata = asyncio.run(
                migration._read_json(session.store, f"{path}/zarr.json")
            )
            metadata["dimension_names"] = ["bad", "lead_time", "y", "x"]
            asyncio.run(
                session.store.set(
                    f"{path}/zarr.json",
                    default_buffer_prototype().buffer.from_bytes(
                        json.dumps(metadata).encode()
                    ),
                )
            )
        elif case == "consolidated":
            asyncio.run(_consolidate_metadata(session.store))
            group[path].attrs["stale"] = True
        else:
            group.attrs["concurrent"] = True
        changed = session.commit("unexpected state")
        if case != "stale":
            expected = changed
    before = _disk_bytes(tmp_path)
    with pytest.raises(AssertionError):
        migration.migrate(repo, dataset, expected, apply=True)
    assert _disk_bytes(tmp_path) == before


def test_tag_only_retry(tmp_path: Path) -> None:
    repo = _simple_store(tmp_path)
    expected = repo.lookup_branch("main")
    plan = migration.migrate(repo, DATASET, expected)
    repo.create_tag(plan.historical_tag, expected)
    assert migration.migrate(repo, DATASET, expected, apply=True).status == "applied"


def test_tag_mismatch_is_read_only(tmp_path: Path) -> None:
    repo = _simple_store(tmp_path)
    expected = repo.lookup_branch("main")
    plan = migration.migrate(repo, DATASET, expected)
    initial = list(repo.ancestry(branch="main"))[-1].id
    repo.create_tag(plan.historical_tag, initial)
    before = _disk_bytes(tmp_path)
    with pytest.raises(AssertionError, match="Historical tag mismatch"):
        migration.migrate(repo, DATASET, expected, apply=True)
    assert _disk_bytes(tmp_path) == before


def test_deleted_tag_cannot_be_recreated(tmp_path: Path) -> None:
    repo = _simple_store(tmp_path)
    expected = repo.lookup_branch("main")
    plan = migration.migrate(repo, DATASET, expected)
    repo.create_tag(plan.historical_tag, expected)
    repo.delete_tag(plan.historical_tag)
    before = _disk_bytes(tmp_path)
    with pytest.raises(icechunk.AlreadyExistsError):
        migration.migrate(repo, DATASET, expected, apply=True)
    assert _disk_bytes(tmp_path) == before


@pytest.mark.parametrize("change", ["missing-tag", "later-commit"])
def test_rerun_requires_tag_and_unchanged_migration_head(
    tmp_path: Path, change: str
) -> None:
    repo = _simple_store(tmp_path)
    expected = repo.lookup_branch("main")
    result = migration.migrate(repo, DATASET, expected, apply=True)
    if change == "missing-tag":
        repo.delete_tag(result.historical_tag)
    else:
        writer = repo.writable_session("main")
        zarr.open_group(writer.store).attrs["later"] = True
        writer.commit("later update")
    before = _disk_bytes(tmp_path)
    with pytest.raises(AssertionError, match=r"historical tag|Stale head"):
        migration.migrate(repo, DATASET, expected, apply=True)
    assert _disk_bytes(tmp_path) == before


def test_concurrent_commit_conflicts_without_rebase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _simple_store(tmp_path)
    expected = repo.lookup_branch("main")
    real_commit = icechunk.Session.commit
    concurrent: list[str] = []

    def interleave(
        session: icechunk.Session, message: str, metadata: dict[str, Any] | None = None
    ) -> str:
        assert metadata is not None
        writer = repo.writable_session("main")
        zarr.open_group(writer.store).attrs["concurrent"] = True
        concurrent.append(real_commit(writer, "concurrent writer"))
        return real_commit(session, message, metadata=metadata)

    monkeypatch.setattr(icechunk.Session, "commit", interleave)
    with pytest.raises(icechunk.ConflictError):
        migration.migrate(repo, DATASET, expected, apply=True)
    assert repo.lookup_branch("main") == concurrent[0]
    plan_tag = f"{migration.MIGRATION}-before-{DATASET}-{expected}"
    assert repo.lookup_tag(plan_tag) == expected
    group = zarr.open_group(repo.readonly_session("main").store, mode="r")
    assert all(path in group for path in migration.RETIRED_ARRAYS[DATASET])
    assert group.attrs["concurrent"]
    with pytest.raises(AssertionError, match="Stale head"):
        migration.migrate(repo, DATASET, expected, apply=True)


@pytest.mark.parametrize("point", ["session", "tag"])
def test_head_changes_before_deletion_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    repo = _simple_store(tmp_path)
    expected = repo.lookup_branch("main")
    method = "writable_session" if point == "session" else "create_tag"
    real_method = getattr(icechunk.Repository, method)

    def interleave(
        repository: icechunk.Repository, name: str, snapshot_id: str | None = None
    ) -> icechunk.Session | None:
        monkeypatch.setattr(icechunk.Repository, method, real_method)
        writer = repository.writable_session("main")
        zarr.open_group(writer.store).attrs["concurrent"] = True
        writer.commit("concurrent writer")
        if snapshot_id is None:
            return real_method(repository, name)
        return real_method(repository, name, snapshot_id)

    monkeypatch.setattr(icechunk.Repository, method, interleave)
    with pytest.raises(AssertionError, match="Head changed"):
        migration.migrate(repo, DATASET, expected, apply=True)
    assert all(
        path in zarr.open_group(repo.readonly_session("main").store, mode="r")
        for path in migration.RETIRED_ARRAYS[DATASET]
    )


def test_s3_auth_is_explicit_and_never_uses_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = Mock()
    monkeypatch.setattr(migration, "open_repo", opener)
    migration.open_repository("s3://bucket/prefix", auth_from_env=False)
    opener.assert_called_once_with("s3://bucket/prefix", auth="anonymous")
    migration.open_repository("s3://bucket/prefix", auth_from_env=True)
    opener.assert_called_with("s3://bucket/prefix", auth="env")


@pytest.mark.parametrize(
    "uri",
    [
        "relative",
        "https://bucket/repo",
        "s3://bucket",
        "file://host/tmp/repo",
        "s3://bucket/repo?x=1",
    ],
)
def test_rejects_ambiguous_repository_uri(uri: str) -> None:
    with pytest.raises(AssertionError):
        migration.open_repository(uri, auth_from_env=False)


def test_allowlist_is_exactly_six_datasets_and_32_arrays() -> None:
    assert set(migration.RETIRED_ARRAYS) == {cls.dataset_id for cls in CONFIGS}
    assert [len(migration.RETIRED_ARRAYS[cls.dataset_id]) for cls in CONFIGS] == [
        4,
        4,
        2,
        2,
        4,
        16,
    ]
    assert sum(map(len, migration.RETIRED_ARRAYS.values())) == 32
    assert all(
        "/" not in path for paths in migration.RETIRED_ARRAYS.values() for path in paths
    )
