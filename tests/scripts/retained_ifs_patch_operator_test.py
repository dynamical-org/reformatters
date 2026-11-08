import json
import tarfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from scripts import retained_ifs_patch_operator as operator
from scripts import retained_ifs_patch_verify as verifier


def bundle() -> dict[str, Any]:
    units = []
    for var, date in sorted({(v, d) for v, d, _, _ in verifier.SCOPE}):
        units.append(
            {
                "variable": var,
                "init": date + "T00:00:00",
                "fields": [
                    {
                        "member": m,
                        "lead_hours": lead,
                        "expected_sha256": verifier.EXPECTED_HASHES[
                            (v, d + "T00:00:00", m, lead)
                        ],
                    }
                    for v, d, m, lead in verifier.SCOPE
                    if (v, d) == (var, date)
                ],
            }
        )
    manifest = {
        "version": 2,
        "dataset_id": "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
        "scientific_manifest_sha256": verifier.SCIENCE_SHA256,
        "repository": verifier.REPOSITORY,
        "base_snapshot": verifier.BASE,
        "protected_branch": verifier.PROTECTED,
        "candidate_branch": "retained-repair-candidate",
        "runtime_sha256": "b" * 64,
        "units": units,
        "approved_shards": {
            "test/c/0": {"encoded_sha256": "a" * 64, "decoded_sha256": "b" * 64}
        },
    }
    return {
        "version": 1,
        "commit": "c" * 40,
        "architecture": "arm64",
        "capacity": "spot",
        "image_tag": "registry/reformatters:retained-repair",
        "manifest": manifest,
        "manifest_sha256": operator.sha(operator.canonical(manifest)),
        "implementation_review": {"path": "review.md", "sha256": "d" * 64},
        "independent_stage": {"path": "stage.json", "sha256": "d" * 64},
        "kube_context": "approved-context",
        "kube_server_sha256": operator.sha(b"https://approved-cluster"),
        "expected_seconds": 600,
        "deadline_seconds": 1200,
        "job_name": "retained-repair-job",
        "namespace": "default",
        "memory": "8Gi",
        "scratch": "20Gi",
        "storage_secret": "storage-options",
    }


def test_no_pvc_one_job_exact_inline_manifest_and_arm64() -> None:
    options = bundle()
    job = operator.render(options, "sha256:" + "f" * 64)
    pod = job["spec"]["template"]["spec"]
    assert pod["nodeSelector"]["kubernetes.io/arch"] == "arm64"
    assert pod["nodeSelector"]["karpenter.sh/capacity-type"] == "spot"
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["completions"] == job["spec"]["parallelism"] == 1
    assert not pod["automountServiceAccountToken"]
    assert pod["volumes"] == [
        {
            "name": "storage",
            "secret": {
                "secretName": "storage-options",
                "items": [{"key": "contents", "path": "storage-options.json"}],
            },
        }
    ]
    container = pod["containers"][0]
    env = {v["name"]: v["value"] for v in container["env"]}
    assert operator.sha(env["REPAIR_MANIFEST"].encode()) == options["manifest_sha256"]
    assert "--commit-to-new-branch" in container["command"]
    assert "--auth" in container["command"]


def test_dry_run_never_calls_external_command_or_makes_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        operator, "execute", lambda *a, **kw: pytest.fail("External command")
    )
    monkeypatch.setattr(
        operator, "anonymous_repo", lambda *a: pytest.fail("Network read")
    )
    work = tmp_path / "dry"
    assert (
        operator.launch(bundle(), work, tmp_path, dry_run=True)["external_commands"]
        == 0
    )
    assert not work.exists()


def test_source_check_has_no_credentials_or_commit_flag(tmp_path: Path) -> None:
    argv = operator.parity_command(bundle(), "sha256:" + "e" * 64, tmp_path)
    assert "--check-sources-only" in argv
    assert "--read-only" in argv
    assert "--commit-to-new-branch" not in argv
    assert "--auth" not in argv
    assert not any("secrets" in part or ".aws" in part for part in argv)
    assert argv[argv.index("--platform") + 1] == "linux/arm64"


@pytest.mark.parametrize(
    "failure",
    [
        "source",
        "source_binding",
        "machine",
        "counts",
        "tag",
        "registry",
        "fresh_gate",
        "none",
    ],
)
def test_launch_order_and_failure_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    options = bundle()
    work = tmp_path / "launch"
    work.mkdir()
    calls: list[list[str]] = []
    checks = []
    image_id = "sha256:" + "e" * 64
    digest = "sha256:" + "f" * 64

    def fresh(value: dict[str, Any]) -> None:
        checks.append(len(calls))
        if failure == "fresh_gate" and len(checks) == 2:
            raise ValueError("Fresh gate failed")

    def run(argv: list[str], **kwargs: object) -> str:
        calls.append(argv)
        if argv[:2] == ["docker", "info"]:
            return "aarch64\n"
        if argv[:3] == ["git", "cat-file", "-t"]:
            return "commit\n"
        if argv[:2] == ["git", "archive"]:
            with tarfile.open(work / "source.tar", "w"):
                pass
        if argv[:3] == ["docker", "image", "inspect"]:
            count = sum(c[:3] == argv[:3] for c in calls)
            return json.dumps(
                [
                    {
                        "Id": "sha256:" + "0" * 64
                        if failure == "tag" and count == 2
                        else image_id,
                        "Architecture": "arm64",
                        "Os": "linux",
                        "Config": {
                            "Labels": {
                                "org.opencontainers.image.revision": options["commit"]
                            }
                        },
                        "RepoDigests": ["registry/reformatters@" + digest],
                    }
                ]
            )
        if argv[:3] == ["docker", "buildx", "build"]:
            (work / "build-metadata.json").write_text(
                json.dumps(
                    {
                        "containerimage.config.digest": image_id,
                        "containerimage.digest": digest,
                    }
                )
            )
        if argv[:2] == ["docker", "run"]:
            if failure == "source":
                raise ValueError("source failure")
            (work / "source-check").mkdir()
            receipt = {
                "status": "sources-verified",
                "runtime_sha256": options["manifest"]["runtime_sha256"],
                "manifest_sha256": "0" * 64
                if failure == "source_binding"
                else options["manifest_sha256"],
                "machine": "x86_64" if failure == "machine" else "aarch64",
                "source_count": 7 if failure == "counts" else 8,
                "field_count": 7,
            }
            (work / "source-check/result.json").write_text(json.dumps(receipt))
        if argv[:4] == ["docker", "buildx", "imagetools", "inspect"]:
            return json.dumps(
                {
                    "config": {
                        "digest": "sha256:" + "0" * 64
                        if failure == "registry"
                        else image_id
                    }
                }
            )
        if argv[:2] == ["kubectl", "create"]:
            return json.dumps(
                {"metadata": {"name": options["job_name"], "uid": "job-uid"}}
            )
        return ""

    monkeypatch.setattr(operator, "execute", run)
    monkeypatch.setattr(operator, "fresh_launch_checks", fresh)
    if failure == "none":
        assert (
            operator.launch_in_directory(options, work, tmp_path)["status"]
            == "launched"
        )
        assert sum(c[:2] == ["kubectl", "create"] for c in calls) == 1
        assert (work / "launch-binding.json").exists()
        assert len(checks) == 2
    else:
        with pytest.raises(ValueError, match=r"source|Source|check|tag|Registry|Fresh"):
            operator.launch_in_directory(options, work, tmp_path)
        assert not any(c[:2] == ["kubectl", "create"] for c in calls)
        if failure not in {"registry", "fresh_gate"}:
            assert not any(c[:2] == ["docker", "push"] for c in calls)


def test_bundle_and_evidence_hashes_block_launch(tmp_path: Path) -> None:
    path = tmp_path / "bundle.json"
    value = bundle()
    review = tmp_path / "review.md"
    review.write_text("Reviewed implementation PASS")
    stage = tmp_path / "stage.json"
    stage.write_text(
        json.dumps(
            {
                "status": "pass-local-stage",
                "base_snapshot": verifier.BASE,
                "approved_shards": value["manifest"]["approved_shards"],
            }
        )
    )
    value["implementation_review"]["sha256"] = operator.sha(review.read_bytes())
    value["independent_stage"]["sha256"] = operator.sha(stage.read_bytes())
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hash mismatch"):
        operator.load_bundle(path, "0" * 64)
    assert operator.load_bundle(path, operator.sha(path.read_bytes())) == value
    review.write_text("different")
    with pytest.raises(ValueError, match="Evidence hash mismatch"):
        operator.load_bundle(path, operator.sha(path.read_bytes()))


def test_existing_work_dir_is_untouched(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        operator.launch(bundle(), tmp_path, tmp_path, dry_run=False)
    assert sentinel.read_text() == "keep"
    assert not (tmp_path / "STOP.json").exists()


def test_wrong_docker_host_stops_before_build_or_cloud_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def run(argv: list[str], **kwargs: object) -> str:
        calls.append(argv)
        return "x86_64\n"

    monkeypatch.setattr(operator, "execute", run)
    monkeypatch.setattr(
        operator,
        "fresh_launch_checks",
        lambda *args: pytest.fail("Premature cloud check"),
    )
    with pytest.raises(ValueError, match="native arm64 Docker host"):
        operator.launch_in_directory(bundle(), tmp_path, tmp_path)
    assert len(calls) == 1
    assert calls[0][:2] == ["docker", "info"]


@pytest.mark.parametrize("gate", ["protected", "candidate", "job", "pass"])
def test_fresh_launch_checks(gate: str, monkeypatch: pytest.MonkeyPatch) -> None:
    value = bundle()
    repo = Mock()
    repo.lookup_branch.return_value = "moved" if gate == "protected" else verifier.BASE
    repo.list_branches.return_value = (
        [value["manifest"]["candidate_branch"]]
        if gate == "candidate"
        else ["main", verifier.PROTECTED]
    )
    monkeypatch.setattr(operator, "anonymous_repo", lambda _: repo)
    calls = []

    def run(argv: list[str]) -> str:
        calls.append(argv)
        if argv[:2] == ["kubectl", "config"]:
            return "https://approved-cluster"
        return "{}" if gate == "job" else ""

    monkeypatch.setattr(operator, "execute", run)
    if gate == "pass":
        operator.fresh_launch_checks(value)
        assert calls[0][calls[0].index("--context") + 1] == "approved-context"
    else:
        with pytest.raises(
            ValueError,
            match={
                "protected": "Protected",
                "candidate": "Candidate",
                "job": "Job already",
            }[gate],
        ):
            operator.fresh_launch_checks(value)


@pytest.mark.parametrize("exists", [False, True])
def test_verify_timeout_is_indeterminate_and_never_writes(
    exists: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = bundle()
    repo = Mock()
    repo.lookup_branch.return_value = verifier.BASE
    repo.list_branches.return_value = (
        [value["manifest"]["candidate_branch"]] if exists else []
    )
    monkeypatch.setattr(operator, "anonymous_repo", lambda _: repo)
    monkeypatch.setattr(
        operator, "verify", lambda *args: pytest.fail("Uncommitted candidate")
    )
    with pytest.raises(ValueError, match="INDETERMINATE"):
        operator.verify_when_available(value, tmp_path / "result.json", 0)
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("indexed", [False, True])
def test_registry_uses_build_config_digest_not_local_image_id(
    indexed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = "sha256:" + "a" * 64
    image = "sha256:" + "b" * 64
    registry = "sha256:" + "c" * 64
    child = "sha256:" + "d" * 64

    def run(argv: list[str]) -> str:
        if argv[:3] == ["docker", "image", "inspect"]:
            assert argv[-1] == image
            return json.dumps([{"RepoDigests": ["registry/reformatters@" + registry]}])
        if argv[:4] == ["docker", "buildx", "imagetools", "inspect"]:
            if indexed and argv[-2].endswith(registry):
                return json.dumps(
                    {
                        "manifests": [
                            {
                                "digest": child,
                                "platform": {"architecture": "arm64", "os": "linux"},
                            }
                        ]
                    }
                )
            return json.dumps({"config": {"digest": config}})
        assert argv == ["docker", "push", "registry/reformatters:retained-repair"]
        return ""

    monkeypatch.setattr(operator, "execute", run)
    assert operator.push_checked_image(bundle(), image, config) == (
        child if indexed else registry
    )
