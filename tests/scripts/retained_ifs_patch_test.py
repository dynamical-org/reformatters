import copy
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import httpx
import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from pydantic import ValidationError
from zarr.storage import LocalStore

from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.source_file_coord import (
    IfsEnsSourceFileCoord,
)
from scripts import retained_ifs_patch as patch


def manifest_data() -> dict[str, Any]:
    units = []
    for (variable, init), fields in patch.SCOPE.items():
        unit = patch.Unit(variable=variable, init=init, fields=(), sources=())
        pairs = {(member, lead) for member, lead, _ in fields}
        if variable == "precipitation_surface":
            pairs.add((45, 108))
        sources = []
        for member, lead in sorted(pairs):
            source = patch.Source(
                member=member,
                lead_hours=lead,
                url="",
                start=0,
                end_exclusive=1,
                sha256="0" * 64,
                header={},
            )
            sources.append(
                source.model_copy(
                    update={
                        "url": patch.source_url(unit, source),
                        "header": patch.expected_header(unit, source),
                    }
                ).model_dump()
            )
        units.append(
            {
                "variable": variable,
                "init": init,
                "fields": [
                    {"member": member, "lead_hours": lead, "expected_sha256": sha}
                    for member, lead, sha in fields
                ],
                "sources": sources,
            }
        )
    return {
        "version": 2,
        "dataset_id": patch.helpers.DATASET_ID,
        "repository": patch.REPOSITORY,
        "base_snapshot": patch.BASE_SNAPSHOT,
        "protected_branch": patch.PROTECTED_BRANCH,
        "candidate_branch": "synthetic-patch-candidate",
        "runtime_sha256": "0" * 64,
        "metadata_sha256": "0" * 64,
        "scientific_manifest_sha256": patch.SCIENCE_SHA256,
        "units": units,
    }


def validate(data: dict[str, Any]) -> patch.Manifest:
    return patch.Manifest.model_validate_json(json.dumps(data))


def test_compact_manifest_scope_and_no_runtime_artifact_paths() -> None:
    manifest = validate(manifest_data())
    assert len(manifest.units) == 6
    assert sum(len(u.fields) for u in manifest.units) == 7
    assert sum(len(u.sources) for u in manifest.units) == 8
    assert manifest.approved_shards == {}
    assert "npy" not in manifest.model_dump_json()
    assert "requests" not in manifest.model_dump_json()


@pytest.mark.parametrize(
    "gate",
    [
        "repository",
        "base_snapshot",
        "protected_branch",
        "scientific_manifest_sha256",
        "candidate_branch",
        "unit_duplicate",
        "unit_missing",
        "field_duplicate",
        "field_wrong",
        "field_hash",
        "source_extra",
        "source_missing",
        "source_wrong",
        "source_url",
        "header",
        "range",
        "extra",
        "approved_path",
    ],
)
def test_manifest_rejects_scope_and_identity_drift(gate: str) -> None:  # noqa: PLR0912
    data = manifest_data()
    if gate in {
        "repository",
        "base_snapshot",
        "protected_branch",
        "scientific_manifest_sha256",
    }:
        data[gate] = "f" * 64
    elif gate == "candidate_branch":
        data[gate] = patch.PROTECTED_BRANCH
    elif gate == "unit_duplicate":
        data["units"][1] = data["units"][0]
    elif gate == "unit_missing":
        data["units"].pop()
    elif gate == "field_duplicate":
        data["units"][0]["fields"].append(copy.deepcopy(data["units"][0]["fields"][0]))
    elif gate == "field_wrong":
        data["units"][0]["fields"][0]["member"] = 44
    elif gate == "field_hash":
        data["units"][0]["fields"][0]["expected_sha256"] = "f" * 64
    elif gate == "source_extra":
        data["units"][0]["sources"].append(
            copy.deepcopy(data["units"][0]["sources"][0])
        )
    elif gate == "source_missing":
        data["units"][0]["sources"].pop()
    elif gate == "source_wrong":
        data["units"][0]["sources"][0]["lead_hours"] = 105
    elif gate == "source_url":
        data["units"][0]["sources"][0]["url"] += "?mirror=other"
    elif gate == "header":
        data["units"][0]["sources"][0]["header"]["parameter"] = 151
    elif gate == "range":
        data["units"][0]["sources"][0]["end_exclusive"] = 0
    elif gate == "extra":
        data["prerequisites"] = {"path": "external.npy"}
    else:
        data["approved_shards"] = {
            "../escape": {"encoded_sha256": "0" * 64, "decoded_sha256": "0" * 64}
        }
    with pytest.raises(ValidationError):
        validate(data)


def grib1() -> bytes:
    body = bytearray(100)
    body[:4], body[4:7], body[7] = b"GRIB", len(body).to_bytes(3, "big"), 1
    body[8:11] = (60).to_bytes(3, "big")
    for offset, value in {
        11: 128,
        12: 98,
        16: 167,
        17: 1,
        20: 24,
        21: 2,
        22: 3,
        25: 1,
        26: 90,
        28: 0,
        32: 21,
        48: 30,
        50: 11,
        57: 35,
        58: 51,
    }.items():
        body[offset] = value
    body[-4:] = b"7777"
    return bytes(body)


def grib2() -> bytes:
    reference, field = bytearray(21), bytearray(37)
    reference[5:7], reference[12:14], reference[14:19] = (
        (98).to_bytes(2, "big"),
        (2025).to_bytes(2, "big"),
        bytes([10, 13, 0, 0, 0]),
    )
    field[7:9], field[18:22] = (1).to_bytes(2, "big"), (258).to_bytes(4, "big")
    for offset, value in {9: 1, 10: 19, 17: 1, 22: 1, 34: 255, 35: 3, 36: 51}.items():
        field[offset] = value
    sections = []
    for number, payload in (
        (1, reference),
        (3, bytearray(5)),
        (4, field),
        (5, bytearray(5)),
        (6, bytearray(5)),
        (7, bytearray(5)),
    ):
        payload[:4], payload[4] = len(payload).to_bytes(4, "big"), number
        sections.append(bytes(payload))
    header = bytearray(16)
    header[:4], header[7] = b"GRIB", 2
    body = header + b"".join(sections) + b"7777"
    body[8:16] = len(body).to_bytes(8, "big")
    return bytes(body)


@pytest.mark.parametrize("edition", [1, 2])
def test_header_identity_decodes_synthetic_bytes(edition: int) -> None:
    body = grib1() if edition == 1 else grib2()
    variable = (
        "temperature_2m" if edition == 1 else "categorical_precipitation_type_surface"
    )
    source = next(
        u for u in validate(manifest_data()).units if u.variable == variable
    ).sources[0]
    assert patch.parse_header(body) == source.header


@pytest.mark.parametrize(
    "gate",
    [
        "framing",
        "length",
        "range_indicator",
        "p2",
        "section_size",
        "grib2_duplicate",
        "grib2_truncated",
    ],
)
def test_malformed_headers_fail(gate: str) -> None:
    body = bytearray(grib1())
    if gate == "framing":
        body[-1] = 0
    elif gate == "length":
        body[6] -= 1
    elif gate == "range_indicator":
        body[28] = 4
    elif gate == "p2":
        body[27] = 1
    elif gate == "section_size":
        body[8:11] = (10).to_bytes(3, "big")
    elif gate == "grib2_duplicate":
        body = bytearray(grib2())
        body[41 + 4] = 1
    else:
        body = bytearray(grib2()[:-6] + b"7777")
        body[8:16] = len(body).to_bytes(8, "big")
    with pytest.raises(ValueError, match="GRIB"):
        patch.parse_header(bytes(body))


@pytest.mark.parametrize(
    "gate",
    [
        "pass",
        "status",
        "redirect",
        "range",
        "range_total",
        "length",
        "truncated",
        "too_long",
        "sha",
        "header",
        "encoding",
    ],
)
def test_fetch_requires_anonymous_exact_range_and_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    body = grib1()
    headers = {
        "Content-Range": f"bytes 12-{11 + len(body)}/1000",
        "Content-Length": str(len(body)),
    }
    source = patch.Source(
        member=35,
        lead_hours=90,
        url="https://data.source.coop/only-source",
        start=12,
        end_exclusive=12 + len(body),
        sha256=hashlib.sha256(body).hexdigest(),
        header=patch.parse_header(body),
    )
    status = 206
    if gate == "status":
        status = 200
    elif gate == "redirect":
        status = 302
        headers["Location"] = "https://other.invalid/source"
    elif gate == "range":
        headers["Content-Range"] = "bytes 13-112/1000"
    elif gate == "range_total":
        headers["Content-Range"] = "bytes 12-111/*"
    elif gate == "length":
        headers["Content-Length"] = "99"
    elif gate == "truncated":
        body = body[:-1]
    elif gate == "too_long":
        body += b"x"
    elif gate == "sha":
        source = source.model_copy(update={"sha256": "f" * 64})
    elif gate == "header":
        source = source.model_copy(update={"header": source.header | {"member": 34}})
    elif gate == "encoding":
        headers["Content-Encoding"] = "gzip"
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Range"] == "bytes=12-111"
        assert request.headers["Accept-Encoding"] == "identity"
        assert "authorization" not in request.headers
        return httpx.Response(status, headers=headers, stream=httpx.ByteStream(body))

    client = httpx.Client

    def factory(
        *, trust_env: bool, follow_redirects: bool, timeout: float
    ) -> httpx.Client:
        assert trust_env is False
        assert follow_redirects is False
        return client(
            transport=httpx.MockTransport(handle),
            trust_env=trust_env,
            follow_redirects=follow_redirects,
            timeout=timeout,
        )

    monkeypatch.setattr(patch.httpx, "Client", factory)
    path = tmp_path / "source.grib"
    if gate == "pass":
        receipt = patch.fetch_source(source, path)
        assert path.read_bytes() == body
        assert receipt["http_status"] == 206
    else:
        with pytest.raises(ValueError, match=r"Source|source|Content-Range"):
            patch.fetch_source(source, path)
        assert not path.exists()
    assert len(requests) == 1


@pytest.fixture
def example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> tuple[icechunk.Repository, patch.Manifest, patch.Fields]:
    zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "repo"))
    )
    session = repo.writable_session("main")
    values = np.arange(2 * 2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 2, 3, 4)
    values.view("<u4")[1, 0, 0, 0, 0] = 0x7FC01234
    values.view("<u4")[1, 0, 0, 0, 1] = 0x80000000
    ds = xr.Dataset(
        {"temperature_2m": (patch.helpers.DIMS, values)},
        coords={
            "init_time": np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]"),
            "lead_time": np.array([0, 3], dtype="timedelta64[h]"),
            "ensemble_member": [0, 1],
            "latitude": [90, 0, -90],
            "longitude": [-180, -90, 0, 90],
        },
        attrs={"dataset_id": patch.helpers.DATASET_ID},
    )
    ds.temperature_2m.encoding.update(
        chunks=getattr(request, "param", (1, 2, 2, 1, 1)),
        shards=(1, 2, 2, 2, 2),
        fill_value=np.nan,
    )
    ds.to_zarr(session.store, zarr_format=3, consolidated=False)
    base = session.commit("Synthetic baseline")
    repo.create_branch(patch.PROTECTED_BRANCH, base)
    expected = values[1, 1, 1].copy()
    expected[0, 0] += 100
    expected.view("<u4")[0, 1] = 0x7FC12345
    fields = {(1, 3): expected}
    monkeypatch.setattr(patch, "BASE_SNAPSHOT", base)
    monkeypatch.setattr(
        patch,
        "SCOPE",
        {
            ("temperature_2m", "2000-01-02T00:00:00"): (
                (1, 3, patch.raw_digest(expected)),
            )
        },
    )
    for name, value in (
        ("LEADS", np.array([0, 3])),
        ("MEMBERS", np.array([0, 1])),
        ("LATITUDE", np.array([90, 0, -90])),
        ("LONGITUDE", np.array([-180, -90, 0, 90])),
    ):
        monkeypatch.setattr(patch, name, value)
    monkeypatch.setattr(patch, "verify_runtime", lambda *args: None)
    monkeypatch.setattr(
        patch,
        "reconstruct_fields",
        lambda unit, work: (fields, [source.model_dump() for source in unit.sources]),
    )
    metadata = patch.helpers.metadata_bytes(
        repo.readonly_session(snapshot_id=base).store,
        patch.helpers.open_tree(repo.readonly_session(snapshot_id=base).store),
    )
    data = manifest_data()
    data["metadata_sha256"] = patch.helpers.metadata_digest(metadata)
    return repo, validate(data), fields


def approved(
    example: tuple[icechunk.Repository, patch.Manifest, patch.Fields], tmp_path: Path
) -> patch.Manifest:
    repo, manifest, _ = example
    result = patch.run(repo, manifest, tmp_path / "prior-readonly-stage")
    return manifest.model_copy(
        update={
            "approved_shards": {
                key: patch.ShardHashes.model_validate(value)
                for key, value in result["changed_shards"].items()
            }
        }
    )


def test_prepare_no_writable_session_and_complete_comparison(
    example: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, manifest, _ = example
    create = Mock(side_effect=AssertionError("No branch creation"))
    writable = Mock(side_effect=AssertionError("No writable session"))
    monkeypatch.setattr(repo, "create_branch", create)
    monkeypatch.setattr(repo, "writable_session", writable)
    result = patch.run(repo, manifest, tmp_path / "prepare")
    assert result["status"] == "prepared"
    assert "snapshot" not in result
    assert result["units"][0]["comparison"] == {
        "compared_cells": 48,
        "changed_cells": 2,
        "outside_cells": 36,
    }
    assert len(result["changed_shards"]) == 1
    assert (
        len(list((tmp_path / "prepare/stage/temperature_2m/c").rglob("*/*/*/*/*"))) == 4
    )
    create.assert_not_called()
    writable.assert_not_called()


def test_commit_uses_c1_and_preserves_metadata_refs_and_all_uint32(
    example: tuple, tmp_path: Path
) -> None:
    repo, manifest, fields = example
    manifest = approved(example, tmp_path)
    result = patch.run(repo, manifest, tmp_path / "commit", commit=True)
    assert result["status"] == "committed-and-verified"
    assert repo.lookup_branch(manifest.candidate_branch) == result["snapshot"]
    assert (
        repo.lookup_branch("main")
        == repo.lookup_branch(patch.PROTECTED_BRANCH)
        == manifest.base_snapshot
    )
    assert result["post_comparisons"] == [result["units"][0]["comparison"]]
    assert result["changed_shards"] == {
        k: v.model_dump() for k, v in manifest.approved_shards.items()
    }
    before = repo.readonly_session(snapshot_id=manifest.base_snapshot).store
    after = repo.readonly_session(snapshot_id=result["snapshot"]).store
    assert patch.helpers.metadata_bytes(
        before, patch.helpers.open_tree(before)
    ) == patch.helpers.metadata_bytes(after, patch.helpers.open_tree(after))
    output = np.asarray(zarr.open_array(after, path="temperature_2m")[1])
    assert np.array_equal(output[1, 1].view("<u4"), fields[1, 3].view("<u4"))
    assert output.view("<u4")[0, 0, 0, 0] == 0x7FC01234
    assert output.view("<u4")[0, 0, 0, 1] == 0x80000000
    assert (
        len(list(repo.ancestry(snapshot_id=result["snapshot"])))
        == len(list(repo.ancestry(snapshot_id=manifest.base_snapshot))) + 1
    )


@pytest.mark.parametrize("gate", ["missing", "key", "encoded", "decoded"])
def test_c1_mismatch_cannot_create_branch(
    example: tuple, tmp_path: Path, gate: str
) -> None:
    repo, manifest, _ = example
    if gate != "missing":
        manifest = approved(example, tmp_path)
        key, hashes = next(iter(manifest.approved_shards.items()))
        if gate == "key":
            key = key[:-1] + "9"
        else:
            hashes = hashes.model_copy(update={f"{gate}_sha256": "f" * 64})
        manifest = manifest.model_copy(update={"approved_shards": {key: hashes}})
    work = tmp_path / "commit"
    with pytest.raises(ValueError, match="C1"):
        patch.run(repo, manifest, work, commit=True)
    assert manifest.candidate_branch not in repo.list_branches()
    assert (
        json.loads((work / "STOP.json").read_text())["automatic_retry_allowed"] is False
    )


@pytest.mark.parametrize("gate", ["nan_payload", "signed_zero", "finite", "expected"])
def test_comparator_has_no_nan_normalization(example: tuple, gate: str) -> None:
    _, _, fields = example
    left = np.zeros((2, 2, 3, 4), dtype=np.float32)
    left[1, 1] = fields[1, 3]
    right = left.copy()
    if gate == "nan_payload":
        left.view("<u4")[0, 0, 0, 0] = 0x7FC01234
        right.view("<u4")[0, 0, 0, 0] = 0x7FC05678
    elif gate == "signed_zero":
        left.view("<u4")[0, 0, 0, 0] = 0x80000000
    elif gate == "finite":
        right[0, 0, 0, 0] = 1
    else:
        right.view("<u4")[1, 1, 0, 1] = 0x7FC54321
    with pytest.raises(ValueError, match="uint32"):
        patch.compare_shard(
            left, right, fields, (1, slice(None), slice(None), slice(0, 3), slice(0, 4))
        )


def test_decoded_hash_excludes_padding_and_keeps_nan_payload_bits() -> None:
    values = np.arange(2 * 3 * 3 * 5, dtype="<f4").reshape(2, 3, 3, 5)
    values.view("<u4")[0, 0, 0, 0] = 0x7FC01234
    intersection = values[:, :, -1:, -1:]
    assert (
        patch.raw_digest(intersection)
        == hashlib.sha256(
            np.ascontiguousarray(intersection, dtype="<f4").tobytes(order="C")
        ).hexdigest()
    )
    old = patch.raw_digest(values)
    values.view("<u4")[0, 0, 0, 0] = 0x7FC05678
    assert patch.raw_digest(values) != old


@pytest.mark.parametrize(
    "gate",
    [
        "stage_bytes",
        "stage_metadata",
        "status",
        "ancestry",
        "diff",
        "post_hash",
        "protected_after",
        "candidate_after",
    ],
)
def test_commit_failure_stops_without_retry_or_ref_cleanup(
    example: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    repo, _, _ = example
    manifest = approved(example, tmp_path)
    work = tmp_path / "commit"
    record = patch.helpers.write_record
    key = next(iter(manifest.approved_shards))

    def write(path: Path, value: object) -> None:
        record(path, value)
        if path.name == "precommit.json" and gate.startswith("stage_"):
            target = work / "stage" / (key if gate == "stage_bytes" else "zarr.json")
            target.chmod(0o644)
            target.write_bytes(target.read_bytes() + b" ")

    monkeypatch.setattr(patch.helpers, "write_record", write)
    commits = []
    writable = repo.writable_session
    lookup = repo.lookup_branch

    def session(branch: str) -> icechunk.Session:
        assert branch == manifest.candidate_branch
        result = writable(branch)
        commit = result.commit

        def commit_once(message: str) -> str:
            snapshot = commit(message)
            commits.append(snapshot)
            return snapshot

        monkeypatch.setattr(result, "commit", commit_once)
        if gate == "status":
            monkeypatch.setattr(
                result, "status", lambda: Mock(new_groups=["unexpected"])
            )
        return result

    monkeypatch.setattr(repo, "writable_session", session)
    if gate == "ancestry":
        monkeypatch.setattr(repo, "ancestry", lambda **kwargs: iter([Mock(id="wrong")]))
    if gate == "diff":
        monkeypatch.setattr(
            repo, "diff", lambda **kwargs: Mock(new_groups=["unexpected"])
        )
    if gate == "post_hash":
        monkeypatch.setattr(
            patch,
            "post_compare",
            Mock(side_effect=ValueError("Committed decoded shard SHA256 mismatch")),
        )

    def checked_lookup(branch: str) -> str:
        if commits and (
            (gate == "protected_after" and branch == patch.PROTECTED_BRANCH)
            or (gate == "candidate_after" and branch == manifest.candidate_branch)
        ):
            return "moved"
        return lookup(branch)

    monkeypatch.setattr(repo, "lookup_branch", checked_lookup)
    with pytest.raises(ValueError, match=r"Sealed|Metadata|ancestry|Committed|moved"):
        patch.run(repo, manifest, work, commit=True)
    assert len(commits) == int(
        gate in {"ancestry", "diff", "post_hash", "protected_after", "candidate_after"}
    )
    assert not (work / "result.json").exists()
    assert (work / "STOP.json").exists()
    assert lookup("main") == lookup(patch.PROTECTED_BRANCH) == manifest.base_snapshot
    if commits:
        assert (
            json.loads((work / "committed.json").read_text())["snapshot"] == commits[0]
        )
        assert lookup(manifest.candidate_branch) == commits[0]
    elif gate.startswith("stage_"):
        assert manifest.candidate_branch not in repo.list_branches()


def test_existing_candidate_is_never_reset(example: tuple, tmp_path: Path) -> None:
    repo, manifest, _ = example
    repo.create_branch(manifest.candidate_branch, manifest.base_snapshot)
    with pytest.raises(ValueError, match="already exists"):
        patch.run(repo, manifest, tmp_path / "work")
    assert repo.lookup_branch(manifest.candidate_branch) == manifest.base_snapshot


def test_existing_work_directory_untouched(example: tuple, tmp_path: Path) -> None:
    repo, manifest, _ = example
    work = tmp_path / "existing"
    work.mkdir()
    (work / "STOP.json").write_text("original")
    before = {path.name: path.read_bytes() for path in work.iterdir()}
    with pytest.raises(FileExistsError):
        patch.run(repo, manifest, work)
    assert {path.name: path.read_bytes() for path in work.iterdir()} == before


def cli_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str]
) -> tuple[Path, dict[str, Any]]:
    data = manifest_data()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data, indent=2))
    work = tmp_path / "cli-work"
    monkeypatch.setattr(
        patch.sys,
        "argv",
        [
            "patch",
            "--manifest",
            str(path),
            "--manifest-sha256",
            patch.helpers.file_digest(path),
            "--work-dir",
            str(work),
            *extra,
        ],
    )
    return work, data


def test_source_only_checks_runtime_and_never_opens_repository_or_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, data = cli_args(tmp_path, monkeypatch, ["--check-sources-only"])
    runtime = Mock()
    opened = Mock(side_effect=AssertionError("No repository or credentials"))
    monkeypatch.setattr(patch, "verify_runtime", runtime)
    monkeypatch.setattr(patch, "open_repo", opened)
    monkeypatch.setattr(
        patch,
        "reconstruct_fields",
        lambda unit, work: (
            {
                (field.member, field.lead_hours): np.zeros((1, 1), dtype=np.float32)
                for field in unit.fields
            },
            [source.model_dump() for source in unit.sources],
        ),
    )
    patch.main()
    result = json.loads((work / "result.json").read_text())
    assert result["status"] == "sources-verified"
    assert result["source_count"] == 8
    assert result["field_count"] == 7
    assert result["machine"] == patch.platform.machine()
    assert result["manifest_sha256"] == patch.helpers.object_digest(
        validate(data).model_dump(mode="json")
    )
    assert result["runtime_sha256"] == data["runtime_sha256"]
    assert "snapshot" not in result
    opened.assert_not_called()
    runtime.assert_called_once()


@pytest.mark.parametrize(
    "extra",
    [
        ["--check-sources-only", "--auth", "env"],
        ["--check-sources-only", "--auth", "secret", "--k8s-secret", "forbidden"],
        ["--auth", "env"],
        ["--commit-to-new-branch"],
    ],
)
def test_cli_rejects_credentials_before_runtime_or_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str]
) -> None:
    work, _ = cli_args(tmp_path, monkeypatch, extra)
    opened = Mock(side_effect=AssertionError("Must not access credentials"))
    runtime = Mock(side_effect=AssertionError("Must not start runtime"))
    monkeypatch.setattr(patch, "open_repo", opened)
    monkeypatch.setattr(patch, "verify_runtime", runtime)
    with pytest.raises(ValueError, match=r"access|authentication"):
        patch.main()
    opened.assert_not_called()
    runtime.assert_not_called()
    assert (work / "STOP.json").exists()


def test_manifest_file_pin_checked_before_parse_or_remote_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, _ = cli_args(tmp_path, monkeypatch, ["--check-sources-only"])
    path = tmp_path / "manifest.json"
    path.write_text(path.read_text() + " ")
    opened = Mock(side_effect=AssertionError("Must not access remote"))
    monkeypatch.setattr(patch, "open_repo", opened)
    with pytest.raises(ValueError, match="Manifest file SHA256"):
        patch.main()
    opened.assert_not_called()
    assert (work / "STOP.json").exists()


def test_runtime_pin_rejects_changed_source_before_numerical_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = validate(manifest_data())
    monkeypatch.setattr(patch.helpers, "runtime_digest", lambda root: "f" * 64)
    with pytest.raises(ValueError, match="Runtime source/lock hash"):
        patch.verify_runtime(manifest, Path.cwd())


@pytest.mark.parametrize("example", [(1, 1, 1, 1, 1)], indirect=True)
def test_codec_normalizing_fill_only_nan_payload_stops_before_branch(
    example: tuple, tmp_path: Path
) -> None:
    repo, manifest, _ = example
    work = tmp_path / "nan-codec"
    with pytest.raises(ValueError, match="Codec roundtrip changed uint32 cells"):
        patch.run(repo, manifest, work)
    assert manifest.candidate_branch not in repo.list_branches()
    assert (work / "STOP.json").exists()


@pytest.mark.parametrize(
    "gate",
    [
        "pass",
        "negative",
        "grid",
        "forecast",
        "reference",
        "source_hash",
        "expected_hash",
    ],
)
def test_precipitation_uses_raw108_111_114_and_strict_production_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    unit = validate(manifest_data()).units[0]
    assert unit.variable == "precipitation_surface"
    body = b"synthetic-grib-for-decoder-stub"
    sha = hashlib.sha256(body).hexdigest()
    unit = unit.model_copy(
        update={
            "sources": tuple(
                source.model_copy(update={"sha256": sha}) for source in unit.sources
            ),
            "fields": tuple(
                field.model_copy(
                    update={
                        "expected_sha256": patch.raw_digest(
                            np.full(
                                (721, 1440),
                                1 if field.lead_hours == 111 else 2,
                                dtype=np.float32,
                            )
                        )
                    }
                )
                for field in unit.fields
            ),
        }
    )
    if gate == "expected_hash":
        unit = unit.model_copy(
            update={
                "fields": (
                    unit.fields[0].model_copy(update={"expected_sha256": "f" * 64}),
                    unit.fields[1],
                )
            }
        )
    values = {108: 0, 111: 10.8, 114: 32.4}
    if gate == "negative":
        values[114] = 0
    sources = sorted(unit.sources, key=lambda source: source.lead_hours)

    def fetch(source: patch.Source, path: Path) -> dict[str, Any]:
        path.write_bytes(body + (b"corrupt" if gate == "source_hash" else b""))
        return source.model_dump()

    def reader(path: Path) -> MagicMock:
        source = sources[int(path.stem.split("-")[-1])]
        opened = MagicMock()
        opened.__enter__.return_value = opened
        opened.shape, opened.height, opened.width = (721, 1440), 721, 1440
        opened.transform = patch.rasterio.transform.from_origin(
            -180.125 if gate != "grid" else -179.875, 90.125, 0.25, 0.25
        )
        opened.tags.return_value = {
            "GRIB_REF_TIME": str(
                int(pd.Timestamp(unit.init, tz="UTC").timestamp())
                + int(gate == "reference")
            ),
            "GRIB_FORECAST_SECONDS": str(
                (source.lead_hours + int(gate == "forecast")) * 3600
            ),
        }
        return opened

    def read_data(
        self: patch.ProductionJob, coord: IfsEnsSourceFileCoord, data_var: EcmwfDataVar
    ) -> patch.Array:
        return np.full(
            (721, 1440),
            values[int(coord.lead_time / pd.Timedelta(hours=1))],
            dtype=np.float32,
        )

    monkeypatch.setattr(patch, "fetch_source", fetch)
    monkeypatch.setattr(patch.rasterio, "open", reader)
    monkeypatch.setattr(patch.ProductionJob, "read_data", read_data)
    work = tmp_path / "sources"
    if gate == "pass":
        fields, receipts = patch.reconstruct_fields(unit, work)
        assert np.array_equal(fields[45, 111], np.ones((721, 1440), dtype=np.float32))
        assert np.array_equal(
            fields[45, 114], np.full((721, 1440), 2, dtype=np.float32)
        )
        assert len(receipts) == 3
        assert list(work.iterdir()) == []
    else:
        with pytest.raises(
            ValueError,
            match=r"below threshold|grid coordinates|time identity|Source changed|expected raw",
        ):
            patch.reconstruct_fields(unit, work)


def test_real_production_mars_read_scales_in_float64_before_float32_cast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    var = next(
        v for v in patch.ProductionConfig().data_vars if v.name == "temperature_2m"
    )
    ds = xr.Dataset(
        coords={
            "init_time": [pd.Timestamp("2024-02-03")],
            "lead_time": [pd.Timedelta(hours=90)],
            "ensemble_member": [35],
        },
        attrs={"dataset_id": patch.helpers.DATASET_ID},
    )
    job = patch.ProductionJob(
        tmp_store=tmp_path,
        template_ds=xr.DataTree(ds),
        data_vars=[var],
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="synthetic-cast-order",
    )
    coord = job.generate_source_file_coords(ds, [var])[0]
    coord = patch.replace(coord, downloaded_path=tmp_path / "synthetic.grib")
    reader = MagicMock()
    reader.__enter__.return_value = reader
    reader.count = 1
    reader.tags.return_value = {
        "GRIB_ELEMENT": "2T",
        "GRIB_COMMENT": "Synthetic temperature [K]",
    }
    temperature = 273.15 + 0.0123456789
    reader.read.return_value = np.full((721, 1440), temperature, dtype=np.float64)
    monkeypatch.setattr(patch.rasterio, "open", lambda path: reader)
    result = job.read_data(coord, var)
    reader.read.assert_called_once_with(1, out_dtype=np.float64)
    assert result.dtype == np.float32
    assert np.all(result == np.float32(temperature - 273.15))
    assert np.float32(temperature - 273.15) != np.float32(
        np.float32(temperature) - 273.15
    )


@pytest.mark.parametrize(
    "gate", ["pass", "outside_unchanged_shard", "encoded_hash", "decoded_hash"]
)
def test_post_reader_checks_unchanged_shards_and_both_hashes(
    example: tuple, tmp_path: Path, gate: str
) -> None:
    repo, manifest, fields = example
    work = tmp_path / "staged"
    result = patch.run(repo, manifest, work)
    base = repo.readonly_session(snapshot_id=manifest.base_snapshot).store
    stage = LocalStore(work / "stage")
    hashes = {
        key: patch.ShardHashes.model_validate(value)
        for key, value in result["changed_shards"].items()
    }
    if gate == "outside_unchanged_shard":
        array = zarr.open_array(stage, path="temperature_2m", mode="r+")
        array[1, 0, 0, 2, 3] = -99
    elif gate != "pass":
        key = next(iter(hashes))
        hashes[key] = hashes[key].model_copy(update={f"{gate[:7]}_sha256": "f" * 64})
    resolved = [(manifest.units[0], 1, fields)]
    if gate == "pass":
        assert patch.post_compare(base, stage, resolved, hashes) == [
            result["units"][0]["comparison"]
        ]
    else:
        with pytest.raises(
            ValueError, match=r"outside permitted|Committed .* shard SHA256"
        ):
            patch.post_compare(base, stage, resolved, hashes)


def test_candidate_conflict_is_not_rebased_or_retried(
    example: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _, _ = example
    manifest = approved(example, tmp_path)
    writable = repo.writable_session
    attempts = []
    competing = []

    def session(branch: str) -> icechunk.Session:
        result = writable(branch)
        commit = result.commit

        def conflict(message: str) -> str:
            attempts.append(message)
            other = writable(branch)
            zarr.open_array(other.store, path="temperature_2m", mode="r+")[
                0, 0, 0, 0, 0
            ] = -1
            competing.append(other.commit("External candidate commit"))
            return commit(message)

        monkeypatch.setattr(result, "commit", conflict)
        return result

    monkeypatch.setattr(repo, "writable_session", session)
    work = tmp_path / "conflict"
    with pytest.raises(icechunk.ConflictError):
        patch.run(repo, manifest, work, commit=True)
    assert len(attempts) == 1
    assert len(competing) == 1
    assert repo.lookup_branch(manifest.candidate_branch) == competing[0]
    assert not (work / "committed.json").exists()
    assert (work / "STOP.json").exists()
    assert repo.lookup_branch(patch.PROTECTED_BRANCH) == manifest.base_snapshot


def test_main_may_advance_without_changing_immutable_base(
    example: tuple, tmp_path: Path
) -> None:
    repo, manifest, _ = example
    other = repo.writable_session("main")
    zarr.open_array(other.store, path="temperature_2m", mode="r+")[0, 0, 0, 0, 0] = -1
    moving_main = other.commit("Operational main movement")
    result = patch.run(repo, manifest, tmp_path / "prepare")
    assert result["status"] == "prepared"
    assert repo.lookup_branch("main") == moving_main
    assert repo.lookup_branch(patch.PROTECTED_BRANCH) == manifest.base_snapshot
