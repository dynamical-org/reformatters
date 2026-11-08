import ast
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import pytest
import xarray as xr
import zarr

from scripts import retained_ifs_patch_verify as verify


def test_no_import_of_writer_or_shared_comparator() -> None:
    tree = ast.parse(Path(verify.__file__).read_text())
    imports = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    imports += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any(
        "retained_ifs_repair" in name or name.endswith("retained_ifs_patch")
        for name in imports
    )


@pytest.mark.parametrize("bits", [0x80000000, 0x7FC00001])
def test_outside_mask_preserves_zero_sign_and_nan_payload(bits: int) -> None:
    left = np.zeros((2, 2, 2, 2), dtype="<f4")
    if bits == 0x7FC00001:
        left[0, 0, 0, 0] = np.nan
    right = left.copy()
    right.view(np.uint32)[0, 0, 0, 0] = bits
    with pytest.raises(ValueError, match="Outside-mask"):
        verify.compare_shard(left, right, {(1, 1)})


def test_mask_is_exact_lead_member_pair() -> None:
    left = np.zeros((2, 2, 2, 2), dtype="<f4")
    right = left.copy()
    right[1, 1] = 1
    assert verify.compare_shard(left, right, {(1, 1)}) == (4, 12)
    right[1, 0] = 1
    with pytest.raises(ValueError, match="Outside-mask"):
        verify.compare_shard(left, right, {(1, 1)})


@pytest.fixture
def candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[icechunk.Repository, dict[str, Any], str]:
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "repo"))
    )
    units = sorted({(var, date) for var, date, _, _ in verify.SCOPE})
    dates = sorted({date for _, date in units})
    leads = sorted({lead for _, _, _, lead in verify.SCOPE})
    dims = verify.DIMS
    shape = (6, len(leads), 51, 3, 4)
    ds = xr.Dataset(
        {
            var: (dims, np.arange(np.prod(shape), dtype="<f4").reshape(shape))
            for var, _ in units
        },
        coords={
            "init_time": np.array(dates, dtype="datetime64[ns]"),
            "lead_time": np.array(leads, dtype="timedelta64[h]"),
            "ensemble_member": np.arange(51),
            "latitude": [1, 0, -1],
            "longitude": [0, 1, 2, 3],
        },
    )
    for var in ds.data_vars:
        ds[var].encoding.update(
            chunks=(1, len(leads), 51, 1, 2), shards=(1, len(leads), 51, 2, 2)
        )
    session = repo.writable_session("main")
    ds.to_zarr(session.store, zarr_format=3, consolidated=False)
    base = session.commit("base")
    monkeypatch.setattr(verify, "BASE", base)
    repo.create_branch(verify.PROTECTED, base)
    repo.create_branch("candidate", base)
    session = repo.writable_session("candidate")
    group = zarr.open_group(session.store, mode="r+")
    manifest: dict[str, Any] = {
        "version": 2,
        "dataset_id": "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
        "scientific_manifest_sha256": verify.SCIENCE_SHA256,
        "base_snapshot": base,
        "protected_branch": verify.PROTECTED,
        "repository": verify.REPOSITORY,
        "candidate_branch": "candidate",
        "units": [],
        "approved_shards": {},
    }
    for var, date in units:
        fields = []
        array = group[var]
        assert isinstance(array, zarr.Array)
        for v, d, member, lead in sorted(verify.SCOPE):
            if (v, d) != (var, date):
                continue
            field = np.asarray(
                array[dates.index(date), leads.index(lead), member]
            ) + np.float32(100)
            array[dates.index(date), leads.index(lead), member] = field
            fields.append(
                {
                    "member": member,
                    "lead_hours": lead,
                    "expected_sha256": verify.raw_digest(field),
                }
            )
        manifest["units"].append(
            {"variable": var, "init": date + "T00:00:00", "fields": fields}
        )
    monkeypatch.setattr(
        verify,
        "EXPECTED_HASHES",
        {
            (u["variable"], u["init"], f["member"], f["lead_hours"]): f[
                "expected_sha256"
            ]
            for u in manifest["units"]
            for f in u["fields"]
        },
    )
    snapshot = session.commit("repair")
    store = repo.readonly_session(snapshot_id=snapshot).store
    group = zarr.open_group(store, mode="r")
    for key in verify.diff_keys(
        repo.diff(from_snapshot_id=base, to_snapshot_id=snapshot)
    ):
        var, _, index, _, _, y, x = key.split("/")
        array = group[var]
        assert isinstance(array, zarr.Array)
        data = np.asarray(
            array[
                int(index),
                :,
                :,
                slice(int(y) * 2, (int(y) + 1) * 2),
                slice(int(x) * 2, (int(x) + 1) * 2),
            ]
        )
        manifest["approved_shards"][key] = {
            "encoded_sha256": hashlib.sha256(verify.read(store, key)).hexdigest(),
            "decoded_sha256": verify.raw_digest(data),
        }
    manifest["metadata_sha256"] = verify.metadata_sha256(
        verify.metadata(repo.readonly_session(snapshot_id=base).store)
    )
    return repo, manifest, snapshot


def test_full_independent_verification(candidate: tuple) -> None:
    repo, manifest, snapshot = candidate
    result = verify.verify(repo, manifest, snapshot)
    assert result["status"] == "pass"
    assert len(result["units"]) == 6
    assert sum(u["changed_cells"] for u in result["units"]) == 7 * 12
    assert sum(u["outside_cells"] for u in result["units"]) == 6 * 6 * 51 * 12 - 7 * 12


@pytest.mark.parametrize(
    "gate",
    ["encoded", "decoded", "field", "key", "metadata", "parent", "protected", "scope"],
)
def test_independent_verification_rejects_failed_gate(
    candidate: tuple, gate: str
) -> None:
    repo, manifest, snapshot = candidate
    if gate in {"encoded", "decoded"}:
        next(iter(manifest["approved_shards"].values()))[gate + "_sha256"] = "0" * 64
    elif gate == "field":
        manifest["units"][0]["fields"][0]["expected_sha256"] = "0" * 64
    elif gate == "key":
        manifest["approved_shards"].pop(next(iter(manifest["approved_shards"])))
    elif gate == "metadata":
        manifest["metadata_sha256"] = "0" * 64
    elif gate == "scope":
        manifest["units"][0]["fields"][0]["member"] = 50
    else:
        branch = "candidate" if gate == "parent" else verify.PROTECTED
        session = repo.writable_session(branch)
        group = zarr.open_group(session.store, mode="r+")
        group.attrs["unexpected"] = True
        updated = session.commit("unexpected")
        if gate == "parent":
            snapshot = updated
    with pytest.raises(
        ValueError,
        match={
            "encoded": "Encoded",
            "decoded": "Decoded",
            "field": "Frozen expected",
            "key": "changed-key",
            "metadata": "metadata",
            "parent": "one commit",
            "protected": "Protected",
            "scope": "scope",
        }[gate],
    ):
        verify.verify(repo, manifest, snapshot)


def test_manifest_hash_failure_creates_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({}))
    output = tmp_path / "out.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify",
            "--manifest",
            str(path),
            "--manifest-sha256",
            "0" * 64,
            "--snapshot",
            "x",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="Manifest hash"):
        verify.main()
    assert not output.exists()


def test_independent_local_stage_and_extra_key_rejection(
    candidate: tuple, tmp_path: Path
) -> None:
    repo, manifest, snapshot = candidate
    store = repo.readonly_session(snapshot_id=snapshot).store
    stage = tmp_path / "stage"
    stage.mkdir()
    permitted = set(verify.metadata(store)) | set(manifest["approved_shards"])
    for key in permitted:
        path = stage / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(verify.read(store, key))
    result = verify.verify_stage(repo, manifest, stage)
    assert result["status"] == "pass-local-stage"
    extra = next(
        key for key in asyncio.run(verify.list_keys(store)) if key not in permitted
    )
    path = stage / extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(verify.read(store, extra))
    with pytest.raises(ValueError, match="local stage keys"):
        verify.verify_stage(repo, manifest, stage)
