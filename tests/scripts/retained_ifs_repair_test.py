import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal
from unittest.mock import Mock

import icechunk
import numpy as np
import pytest
import xarray as xr
import zarr
import zarr.storage
from pydantic import TypeAdapter
from zarr.abc.store import Store

from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
)
from scripts import retained_ifs_repair as repair


def artifact(root: Path, name: str, value: object) -> repair.Artifact:
    path = root / name
    path.write_text(json.dumps(value))
    return repair.Artifact(path=name, sha256=repair.file_digest(path))


@pytest.fixture
def example(tmp_path: Path) -> tuple[icechunk.Repository, repair.Manifest, Path]:
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "repo"))
    )
    session = repo.writable_session("main")
    values = np.arange(2 * 2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 2, 3, 4)
    ds = xr.Dataset(
        {"temperature": (repair.DIMS, values)},
        coords={
            "init_time": np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]"),
            "lead_time": np.array([0, 3], dtype="timedelta64[h]"),
            "ensemble_member": [0, 1],
            "latitude": [1, 0, -1],
            "longitude": [0, 1, 2, 3],
        },
        attrs={"dataset_id": repair.DATASET_ID},
    )
    ds.temperature.encoding.update(
        chunks=(1, 1, 1, 1, 2), shards=(1, 2, 2, 2, 2), fill_value=np.nan
    )
    ds.to_zarr(session.store, zarr_format=3, consolidated=False)
    base = session.commit("Baseline")
    repo.create_branch("retained", base)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    expected = values[1, 1, 1] + 100
    np.save(inputs / "expected.npy", expected)
    field = repair.ExpectedField(
        member=1,
        lead_hours=3,
        values=repair.Artifact(
            path="expected.npy", sha256=repair.file_digest(inputs / "expected.npy")
        ),
        source_evidence=artifact(
            inputs,
            "source.json",
            {
                "status": "pass",
                "independent": True,
                "init": "2000-01-02T00:00:00",
                "variable": "temperature",
                "member": 1,
                "lead_hours": 3,
                "values_sha256": repair.file_digest(inputs / "expected.npy"),
                "source_message_sha256s": ["0" * 64],
            },
        ),
    )
    unit = repair.Unit(
        variable="temperature",
        init="2000-01-02T00:00:00",
        requests=artifact(inputs, "requests.json", []),
        fields=(field,),
    )
    store = repo.readonly_session(snapshot_id=base).store
    tree = xr.open_datatree(
        store,  # ty: ignore[invalid-argument-type]
        engine="zarr",
        chunks={},
        decode_timedelta=True,
    )
    manifest = repair.Manifest(
        version=1,
        dataset_id=repair.DATASET_ID,
        repository="s3://example/store",
        base_snapshot=base,
        protected_branch="retained",
        candidate_branch="repair-candidate",
        runtime_sha256="0" * 64,
        metadata_sha256=repair.metadata_digest(repair.metadata_bytes(store, tree)),
        prerequisites=artifact(inputs, "prerequisites.json", {}),
        units=(unit,),
    )
    return repo, manifest, inputs


def regenerate(
    tree: xr.DataTree,
    variable: str,
    index: int,
    temporary: Path,
    stage: zarr.storage.LocalStore,
    requests: Path,
) -> dict[str, Any]:
    assert temporary.is_dir()
    assert requests.is_file()
    values = (
        tree.to_dataset()[variable]
        .isel(init_time=slice(index, index + 1))
        .values.copy()
    )
    values[0, 1, 1] += 100
    array = zarr.open_array(stage, path=variable, mode="r+")
    array[index : index + 1] = values
    return {
        "source_count": 4,
        "transform": "pass",
        "requests_sha256": repair.file_digest(requests),
    }


def test_stage_only_has_no_repository_writes(
    example: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, manifest, inputs = example
    create = Mock(side_effect=AssertionError("No branch creation"))
    writable = Mock(side_effect=AssertionError("No writable session"))
    monkeypatch.setattr(repo, "create_branch", create)
    monkeypatch.setattr(repo, "writable_session", writable)
    result = repair.run(
        repo, manifest, inputs, tmp_path / "work", regenerate, commit=False
    )
    assert result["status"] == "staged"
    assert result["units"][0]["comparison"] == {
        "compared_cells": 48,
        "changed_cells": 12,
        "outside_cells": 36,
    }
    assert len(result["changed_shards"]) == 4
    create.assert_not_called()
    writable.assert_not_called()


def test_data_only_descendant_commit(
    example: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, manifest, inputs = example
    original = repo.writable_session
    calls = []

    def writable(branch: str) -> icechunk.Session:
        calls.append(branch)
        return original(branch)

    monkeypatch.setattr(repo, "writable_session", writable)
    result = repair.run(
        repo, manifest, inputs, tmp_path / "work", regenerate, commit=True
    )
    assert result["status"] == "committed-and-verified"
    assert calls == ["repair-candidate"]
    assert (
        repo.lookup_branch("main")
        == repo.lookup_branch("retained")
        == manifest.base_snapshot
    )
    assert repo.lookup_branch("repair-candidate") == result["snapshot"]
    diff = repo.diff(
        from_snapshot_id=manifest.base_snapshot, to_snapshot_id=result["snapshot"]
    )
    assert set(diff.updated_chunks) == {"/temperature"}
    assert not diff.updated_arrays


@pytest.mark.parametrize(
    "name", ["main", "retained", "existing", "../bad", "_job_retry"]
)
def test_reject_candidate_names_before_staging(
    example: tuple, tmp_path: Path, name: str
) -> None:
    repo, manifest, inputs = example
    repo.create_branch("existing", manifest.base_snapshot)
    manifest = manifest.model_copy(update={"candidate_branch": name})
    with pytest.raises(ValueError, match=r"[Cc]andidate"):
        repair.run(repo, manifest, inputs, tmp_path / "work", regenerate, commit=True)
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize(
    "change", ["finite", "nan", "metadata", "source", "missing", "extra"]
)
def test_bad_staging_cannot_create_branch(
    example: tuple, tmp_path: Path, change: str
) -> None:
    repo, manifest, inputs = example

    def bad(
        tree: xr.DataTree,
        variable: str,
        index: int,
        temporary: Path,
        stage: zarr.storage.LocalStore,
        requests: Path,
    ) -> dict[str, Any]:
        result = regenerate(tree, variable, index, temporary, stage, requests)
        array = zarr.open_array(stage, path=variable, mode="r+")
        group = zarr.open_group(stage, mode="r+")
        match change:
            case "finite":
                array[1, 0, 0, 0, 0] = -99
            case "nan":
                array[1, 0, 0, 0, 0] = np.nan
            case "metadata":
                group.attrs["extra"] = True
            case "source":
                array[1, 1, 1, 0, 0] = 0
            case "missing":
                (stage.root / "temperature/c/1/0/0/0/0").unlink()
            case "extra":
                array[0, 0, 0, 0, 0] = -99
        return result

    with pytest.raises(ValueError, match=r"Unexpected|metadata|source|shard"):
        repair.run(repo, manifest, inputs, tmp_path / "work", bad, commit=True)
    assert "repair-candidate" not in repo.list_branches()
    assert repo.lookup_branch("retained") == manifest.base_snapshot


@pytest.mark.parametrize(
    "change", ["unknown_init", "duplicate", "member", "lead", "duplicate_field"]
)
def test_exact_pair_and_field_resolution(
    example: tuple, tmp_path: Path, change: str
) -> None:
    repo, manifest, inputs = example
    unit = manifest.units[0]
    match change:
        case "unknown_init":
            units = (unit.model_copy(update={"init": "1999-01-01"}),)
        case "duplicate":
            units = (unit, unit)
        case "member":
            units = (
                unit.model_copy(
                    update={
                        "fields": (unit.fields[0].model_copy(update={"member": 5}),)
                    }
                ),
            )
        case "lead":
            units = (
                unit.model_copy(
                    update={
                        "fields": (unit.fields[0].model_copy(update={"lead_hours": 6}),)
                    }
                ),
            )
        case "duplicate_field":
            units = (
                unit.model_copy(update={"fields": (unit.fields[0], unit.fields[0])}),
            )
        case _:
            raise AssertionError(change)
    manifest = manifest.model_copy(update={"units": units})
    with pytest.raises(ValueError, match=r"resolve exactly|Duplicate|Unknown"):
        repair.run(repo, manifest, inputs, tmp_path / "work", regenerate, commit=True)
    assert "repair-candidate" not in repo.list_branches()


def test_no_retry_after_ambiguous_commit(
    example: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, manifest, inputs = example
    real_session = repo.writable_session
    commit = Mock(side_effect=OSError("Ambiguous response"))

    def writable(branch: str) -> icechunk.Session:
        session = real_session(branch)
        monkeypatch.setattr(session, "commit", commit)
        return session

    monkeypatch.setattr(repo, "writable_session", writable)
    with pytest.raises(OSError, match="Ambiguous"):
        repair.run(repo, manifest, inputs, tmp_path / "work", regenerate, commit=True)
    stop = json.loads((tmp_path / "work/STOP.json").read_text())
    assert stop["candidate_branch"] == manifest.candidate_branch
    assert stop["base_snapshot"] == manifest.base_snapshot
    commit.assert_called_once()
    assert repo.lookup_branch("retained") == manifest.base_snapshot
    assert repo.lookup_branch("repair-candidate") == manifest.base_snapshot
    with pytest.raises(ValueError, match="already exists"):
        repair.run(repo, manifest, inputs, tmp_path / "retry", regenerate, commit=True)


def test_artifact_rejects_escape_and_modified_input(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    external = artifact(tmp_path, "outside.json", {})
    with pytest.raises(ValueError, match="escapes"):
        external.model_copy(update={"path": "../outside.json"}).checked_path(inputs)
    pin = artifact(inputs, "local.json", {})
    (inputs / "local.json").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        pin.checked_path(inputs)


def test_finite_bits_and_nan_masks() -> None:
    left = np.array([0, np.nan, 1], dtype=np.float32)
    right = np.array([-0.0, np.nan, np.nan], dtype=np.float32)
    assert repair.equivalent(left, right).tolist() == [False, True, False]


def test_prerequisite_receipt_is_bound_to_code_snapshot_and_scope(
    example: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest, inputs = example
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "uv.lock").write_text("package = []\n")
    monkeypatch.setattr(repair, "runtime_digest", lambda _: "0" * 64)
    (code_root / "src").mkdir()
    monkeypatch.setattr(
        repair.reformatters, "__file__", str(code_root / "src/reformatters/__init__.py")
    )
    monkeypatch.setattr(
        repair.scripts, "__file__", str(code_root / "src/scripts/__init__.py")
    )
    receipt = {
        "status": "pass",
        "base_snapshot": manifest.base_snapshot,
        "runtime_sha256": manifest.runtime_sha256,
        "units_sha256": repair.object_digest(
            [u.model_dump(mode="json") for u in manifest.units]
        ),
        "selector_parity": True,
        "config_parity": True,
        "decoder_transform_parity": True,
        "full_field_source_confirmation": True,
        "retention_review": True,
    }
    measurement = artifact(inputs, "measurement.json", {"synthetic": True})
    receipt["evidence"] = {
        gate: measurement.model_dump()
        for gate in (
            "selector_parity",
            "config_parity",
            "decoder_transform_parity",
            "full_field_source_confirmation",
            "retention_review",
        )
    }
    pin = artifact(inputs, "bound.json", receipt)
    manifest = manifest.model_copy(update={"prerequisites": pin})
    repair.verify_inputs(manifest, inputs, code_root)
    receipt["units_sha256"] = "f" * 64
    pin = artifact(inputs, "wrong-scope.json", receipt)
    with pytest.raises(ValueError, match="binding mismatch"):
        repair.verify_inputs(
            manifest.model_copy(update={"prerequisites": pin}), inputs, code_root
        )
    receipt["units_sha256"] = repair.object_digest(
        [u.model_dump(mode="json") for u in manifest.units]
    )
    receipt["decoder_transform_parity"] = False
    pin = artifact(inputs, "missing-parity.json", receipt)
    with pytest.raises(ValueError, match="decoder_transform_parity"):
        repair.verify_inputs(
            manifest.model_copy(update={"prerequisites": pin}), inputs, code_root
        )


def test_source_evidence_must_identify_expected_field(
    example: tuple, tmp_path: Path
) -> None:
    repo, manifest, inputs = example
    unit = manifest.units[0]
    pin = artifact(
        inputs,
        "wrong-source.json",
        {"status": "pass", "independent": True, "member": 0},
    )
    field = unit.fields[0].model_copy(update={"source_evidence": pin})
    manifest = manifest.model_copy(
        update={"units": (unit.model_copy(update={"fields": (field,)}),)}
    )
    with pytest.raises(ValueError, match="not bound to this field"):
        repair.run(repo, manifest, inputs, tmp_path / "work", regenerate, commit=True)
    assert "repair-candidate" not in repo.list_branches()


def test_protected_branch_move_blocks_before_write(
    example: tuple, tmp_path: Path
) -> None:
    repo, manifest, inputs = example
    session = repo.writable_session("retained")
    zarr.open_group(session.store, mode="r+").attrs["changed"] = True
    session.commit("External update")
    with pytest.raises(ValueError, match="Protected branch moved"):
        repair.run(repo, manifest, inputs, tmp_path / "work", regenerate, commit=True)
    assert "repair-candidate" not in repo.list_branches()


def test_retained_numerical_configuration_contract() -> None:
    expected = json.loads(
        (Path(__file__).parent / "fixtures/retained_ifs_variables.json").read_text()
    )
    variables = {
        v.name: v for v in EcmwfIfsEnsForecast15Day025DegreeTemplateConfig().data_vars
    }
    seconds = np.r_[np.arange(0, 145, 3), np.arange(150, 361, 6)] * 3600
    for name, historical in expected.items():
        actual = variables[name].model_dump(mode="json")
        assert actual["internal_attrs"].pop("source_units") is None
        old_reset = TypeAdapter(timedelta).validate_python(
            historical["internal_attrs"].pop("window_reset_frequency")
        )
        new_reset = variables[name].internal_attrs.window_reset_frequency
        actual["internal_attrs"].pop("window_reset_frequency")
        assert new_reset is not None
        np.testing.assert_array_equal(
            seconds % old_reset.total_seconds() == 0,
            seconds % new_reset.total_seconds() == 0,
        )
        assert repair.object_digest(actual) == repair.object_digest(historical), name


@pytest.mark.parametrize("prior_stop", [False, True])
def test_existing_work_directory_is_untouched(
    example: tuple, tmp_path: Path, prior_stop: bool
) -> None:
    repo, manifest, inputs = example
    work = tmp_path / "existing"
    work.mkdir()
    (work / "result.json").write_text("original result")
    if prior_stop:
        (work / "STOP.json").write_text("original stop")
    before = {p.name: p.read_bytes() for p in work.iterdir()}
    with pytest.raises(FileExistsError):
        repair.run(repo, manifest, inputs, work, regenerate, commit=False)
    assert {p.name: p.read_bytes() for p in work.iterdir()} == before


def test_failure_records_stop_only_in_owned_directory(
    example: tuple, tmp_path: Path
) -> None:
    repo, manifest, inputs = example
    work = tmp_path / "new"

    def fail(*args: object) -> dict[str, Any]:
        raise RuntimeError("source unavailable")

    with pytest.raises(RuntimeError, match="source unavailable"):
        repair.run(repo, manifest, inputs, work, fail, commit=False)
    assert json.loads((work / "STOP.json").read_text())["error_type"] == "RuntimeError"


def test_comparators_read_all_leads_in_bounded_spatial_blocks(
    example: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, manifest, inputs = example
    store = repo.readonly_session(snapshot_id=manifest.base_snapshot).store
    before = repair.open_tree(store)
    stage_path = tmp_path / "stage"
    repair.clone_metadata(repair.metadata_bytes(store, before), stage_path)
    stage = zarr.storage.LocalStore(stage_path)
    regenerate(before, "temperature", 1, stage_path, stage, inputs / "requests.json")
    after = repair.open_tree(stage)
    original_isel = xr.DataArray.isel
    selections = []

    def checked_isel(
        self: xr.DataArray,
        indexers: dict | None = None,
        drop: bool = False,
        missing_dims: Literal["raise", "warn", "ignore"] = "raise",
        **indexers_kwargs: int | slice,
    ) -> xr.DataArray:
        if self.name == "temperature" and indexers and "latitude" in indexers:
            assert "lead_time" not in indexers
            assert "ensemble_member" not in indexers
            assert indexers["latitude"].stop - indexers["latitude"].start <= 32
            assert indexers["longitude"].stop - indexers["longitude"].start <= 320
            selections.append(indexers)
        return original_isel(
            self, indexers, drop=drop, missing_dims=missing_dims, **indexers_kwargs
        )

    monkeypatch.setattr(xr.DataArray, "isel", checked_isel)
    repair.compare_unit(before, after, manifest.units[0], 1, inputs)
    assert len(selections) == 2
    selections.clear()
    assert len(repair.changed_shards(before, after, manifest.units[0], 1)) == 4
    assert len(selections) == 8


def test_changed_consumed_roster_cannot_create_branch(
    example: tuple, tmp_path: Path
) -> None:
    repo, manifest, inputs = example

    def changed(
        tree: xr.DataTree,
        variable: str,
        index: int,
        temporary: Path,
        stage: zarr.storage.LocalStore,
        requests: Path,
    ) -> dict[str, Any]:
        result = regenerate(tree, variable, index, temporary, stage, requests)
        result["requests_sha256"] = "0" * 64
        return result

    with pytest.raises(ValueError, match="Consumed request roster"):
        repair.run(repo, manifest, inputs, tmp_path / "work", changed, commit=True)
    assert "repair-candidate" not in repo.list_branches()


@pytest.mark.parametrize(
    ("latitude", "longitude"), [(0, 0), (31, 319), (32, 320), (720, 1439)]
)
@pytest.mark.parametrize("nan", [False, True])
def test_comparison_detects_block_edge_changes(
    tmp_path: Path, latitude: int, longitude: int, nan: bool
) -> None:
    values = np.zeros((1, 2, 2, 721, 1440), dtype=np.float32)
    ds = xr.Dataset(
        {"temperature": (repair.DIMS, values)},
        coords={
            "lead_time": np.array([0, 3], dtype="timedelta64[h]"),
            "ensemble_member": [0, 1],
        },
    )
    before = xr.DataTree(ds)
    after = xr.DataTree(ds.copy(deep=True))
    np.save(tmp_path / "expected.npy", values[0, 0, 0])
    pin = repair.file_digest(tmp_path / "expected.npy")
    field = repair.ExpectedField(
        member=0,
        lead_hours=0,
        values=repair.Artifact(path="expected.npy", sha256=pin),
        source_evidence=artifact(
            tmp_path,
            "source.json",
            {
                "status": "pass",
                "independent": True,
                "init": "2000-01-01T00:00:00",
                "variable": "temperature",
                "member": 0,
                "lead_hours": 0,
                "values_sha256": pin,
                "source_message_sha256s": ["0" * 64],
            },
        ),
    )
    unit = repair.Unit(
        variable="temperature",
        init="2000-01-01T00:00:00",
        requests=artifact(tmp_path, "request.json", []),
        fields=(field,),
    )
    expected = 2 * 2 * 721 * 1440
    assert repair.compare_unit(before, after, unit, 0, tmp_path) == {
        "compared_cells": expected,
        "changed_cells": 0,
        "outside_cells": expected - 721 * 1440,
    }
    after.to_dataset()["temperature"].values[0, 1, 1, latitude, longitude] = (
        np.nan if nan else 1
    )
    with pytest.raises(ValueError, match="outside permitted"):
        repair.compare_unit(before, after, unit, 0, tmp_path)


@pytest.mark.parametrize(
    ("gate", "message", "committed"),
    [
        ("staged_bytes", "Staged shard changed", False),
        ("status_chunks", "Transaction changed-key set", False),
        ("status_metadata", "Metadata or structure changed", False),
        ("candidate_base", "Candidate base changed", False),
        ("protected_before", "Protected branch moved before", False),
        ("protected_after", "Protected branch moved after", True),
        ("candidate_after", "Candidate moved after", True),
        ("parent", "Unexpected commit parent", True),
        ("diff_chunks", "Transaction changed-key set", True),
        ("diff_metadata", "Metadata or structure changed", True),
        ("metadata", "Committed metadata/coordinates changed", True),
        ("comparison", "Independent source mismatch", True),
        ("no_correction", "Unit has no correction", False),
    ],
)
def test_commit_and_postcommit_fail_closed(  # noqa: PLR0915
    example: tuple,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate: str,
    message: str,
    committed: bool,
) -> None:
    repo, manifest, inputs = example
    if gate == "no_correction":
        expected_path = inputs / "expected.npy"
        np.save(expected_path, np.load(expected_path) - 100)
        digest = repair.file_digest(expected_path)
        evidence = json.loads((inputs / "source.json").read_text())
        evidence["values_sha256"] = digest
        source = artifact(inputs, "source.json", evidence)
        unit = manifest.units[0]
        field = unit.fields[0].model_copy(
            update={
                "values": repair.Artifact(path="expected.npy", sha256=digest),
                "source_evidence": source,
            }
        )
        manifest = manifest.model_copy(
            update={"units": (unit.model_copy(update={"fields": (field,)}),)}
        )
    work = tmp_path / "work"
    real_writable = repo.writable_session
    commits = []

    def incorrect_diff() -> Mock:
        diff = Mock()
        for attr in (
            "new_groups",
            "new_arrays",
            "deleted_groups",
            "deleted_arrays",
            "updated_groups",
            "updated_arrays",
            "moved_nodes",
        ):
            setattr(diff, attr, [])
        diff.updated_arrays = ["/temperature"] if gate.endswith("metadata") else []
        diff.updated_chunks = {"/unexpected": [[0, 0, 0, 0, 0]]}
        return diff

    def writable(branch: str) -> icechunk.Session:
        session = real_writable(branch)
        real_commit = session.commit

        def commit(message: str) -> str:
            snapshot = real_commit(message)
            commits.append(snapshot)
            return snapshot

        monkeypatch.setattr(session, "commit", commit)
        if gate.startswith("status_"):
            monkeypatch.setattr(session, "status", incorrect_diff)
        if gate == "candidate_base":
            proxy = Mock(wraps=session)
            proxy.snapshot_id = "wrong-base"
            return proxy
        return session

    monkeypatch.setattr(repo, "writable_session", writable)
    if gate.startswith("diff_"):
        monkeypatch.setattr(repo, "diff", lambda **kwargs: incorrect_diff())
    if gate == "parent":
        monkeypatch.setattr(
            repo,
            "ancestry",
            lambda **kwargs: iter([Mock(id="child"), Mock(id="wrong-parent")]),
        )
    lookup = repo.lookup_branch
    protected_reads = 0

    def lookup_branch(branch: str) -> str:
        nonlocal protected_reads
        if branch == manifest.protected_branch:
            protected_reads += 1
            if (gate == "protected_before" and protected_reads == 3) or (
                gate == "protected_after" and protected_reads == 4
            ):
                return "moved"
        if gate == "candidate_after" and branch == manifest.candidate_branch:
            return "moved"
        return lookup(branch)

    monkeypatch.setattr(repo, "lookup_branch", lookup_branch)
    real_record = repair.write_record

    def record(path: Path, value: object) -> None:
        real_record(path, value)
        if gate == "staged_bytes" and path.name == "precommit.json":
            shard = work / "stage/temperature/c/1/0/0/0/0"
            shard.write_bytes(shard.read_bytes() + b"tampered")

    monkeypatch.setattr(repair, "write_record", record)
    real_metadata = repair.metadata_bytes
    metadata_reads = 0

    def metadata(store: Store, tree: xr.DataTree) -> dict[str, bytes]:
        nonlocal metadata_reads
        metadata_reads += 1
        result = real_metadata(store, tree)
        if gate == "metadata" and metadata_reads == 3:
            result["zarr.json"] += b" "
        return result

    monkeypatch.setattr(repair, "metadata_bytes", metadata)
    real_compare = repair.compare_unit
    comparisons = 0

    def compare(
        before: xr.DataTree,
        after: xr.DataTree,
        unit: repair.Unit,
        index: int,
        root: Path,
    ) -> dict[str, int]:
        nonlocal comparisons
        comparisons += 1
        if gate == "comparison" and comparisons == 2:
            after = after.copy(deep=True)
            changed = after.to_dataset()[unit.variable].load()
            changed.values[index, 1, 1, 0, 0] = -999
            after[unit.variable] = changed
        return real_compare(before, after, unit, index, root)

    monkeypatch.setattr(repair, "compare_unit", compare)

    def regenerate_checked(
        tree: xr.DataTree,
        variable: str,
        index: int,
        temporary: Path,
        stage: zarr.storage.LocalStore,
        requests: Path,
    ) -> dict[str, Any]:
        result = regenerate(tree, variable, index, temporary, stage, requests)
        if gate == "no_correction":
            zarr.open_array(stage, path=variable, mode="r+")[index : index + 1] = (
                tree.to_dataset()[variable]
                .isel(init_time=slice(index, index + 1))
                .values
            )
        return result

    with pytest.raises(ValueError, match=message):
        repair.run(repo, manifest, inputs, work, regenerate_checked, commit=True)
    assert len(commits) == int(committed)
    assert not (work / "result.json").exists()
    stop = json.loads((work / "STOP.json").read_text())
    assert stop["candidate_branch"] == manifest.candidate_branch
    assert stop["automatic_retry_allowed"] is False
    assert lookup("retained") == lookup("main") == manifest.base_snapshot


@pytest.mark.parametrize(
    ("gate", "message"),
    [
        ("optimized", "Optimized Python"),
        ("runtime", "Runtime source/lock hash"),
        ("import", "Imported runtime"),
        ("dependency", "Installed dependency differs"),
    ],
)
def test_runtime_identity_gates(
    example: tuple, monkeypatch: pytest.MonkeyPatch, gate: str, message: str
) -> None:
    _, manifest, inputs = example
    root = Path(repair.__file__).resolve().parents[2]
    manifest = manifest.model_copy(
        update={"runtime_sha256": repair.runtime_digest(root)}
    )
    with monkeypatch.context() as patch:
        if gate == "optimized":
            patch.setattr(repair.sys, "flags", Mock(wraps=repair.sys.flags, optimize=1))
        elif gate == "runtime":
            manifest = manifest.model_copy(update={"runtime_sha256": "f" * 64})
        elif gate == "import":
            patch.setattr(repair.scripts, "__file__", "/outside/scripts/__init__.py")
        else:
            patch.setattr(repair, "version", lambda name: "incorrect-version")
        with pytest.raises(ValueError, match=message):
            repair.verify_inputs(manifest, inputs, root)


@pytest.mark.parametrize("gate", ["manifest_hash", "repository"])
def test_cli_rejects_invalid_input_before_repository_open(
    example: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    _, manifest, inputs = example
    manifest = manifest.model_copy(update={"repository": "https://example/store"})
    path = inputs / "manifest.json"
    path.write_text(manifest.model_dump_json())
    digest = "f" * 64 if gate == "manifest_hash" else repair.file_digest(path)
    monkeypatch.setattr(
        repair.sys,
        "argv",
        [
            "repair",
            "--manifest",
            str(path),
            "--manifest-sha256",
            digest,
            "--work-dir",
            str(tmp_path / "work"),
        ],
    )
    monkeypatch.setattr(repair, "verify_inputs", lambda *args: None)
    opened = Mock(side_effect=AssertionError("Must not open repository"))
    monkeypatch.setattr(repair, "open_repo", opened)
    with pytest.raises(
        ValueError, match="Manifest hash" if gate == "manifest_hash" else "explicit S3"
    ):
        repair.main()
    opened.assert_not_called()
    assert not (tmp_path / "work").exists()
