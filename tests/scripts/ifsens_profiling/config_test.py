import hashlib
import json
from datetime import timedelta
from pathlib import Path

import icechunk
import pytest

from reformatters.common import storage
from reformatters.common.config import Config, Env
from reformatters.common.kubernetes import Job, ReformatCronJob
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
)
from src.scripts.ifsens_profiling.__main__ import (
    build_manifest,
    preflight,
    verify_evidence,
)
from src.scripts.ifsens_profiling.config import (
    DATASET_ID,
    END,
    ROOT,
    START,
    VERSION,
    ProfilingStoreFactory,
    ProfilingTemplate,
    dataset,
)

RUN = "p20261029-000000000001"
IMAGE = "example.invalid/profiling@sha256:" + "0" * 64


@pytest.fixture(autouse=True)
def prod_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Config, "env", Env.prod)


@pytest.fixture
def factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ProfilingStoreFactory:
    def local_storage(*_args: object) -> icechunk.Storage:
        return icechunk.local_filesystem_storage(str(tmp_path / "repository"))

    monkeypatch.setattr(storage, "_get_icechunk_storage", local_storage)
    return dataset(RUN, "jpp1").store_factory


def test_create_only_and_owned_sessions(factory: ProfilingStoreFactory) -> None:
    factory.bootstrap(0)
    factory.bootstrap(1)
    assert factory.primary_store().session.read_only
    assert not factory.primary_store(writable=True).session.read_only
    with pytest.raises(icechunk.IcechunkError):
        factory.bootstrap(0)


def test_existing_unowned_repository_is_refused(factory: ProfilingStoreFactory) -> None:
    icechunk.Repository.create(factory._storage())
    with pytest.raises(icechunk.IcechunkError):
        factory.primary_store(writable=True)
    with pytest.raises(icechunk.IcechunkError):
        factory.bootstrap(0)


def test_nonzero_worker_does_not_create(factory: ProfilingStoreFactory) -> None:
    with pytest.raises(TimeoutError):
        factory.bootstrap(1, timeout_seconds=0)
    assert not icechunk.Repository.exists(factory._storage())


@pytest.mark.parametrize(
    "mutation",
    [
        {"template_config_version": "0.1.0"},
        {"dataset_id": "other"},
        {"run_token": "../../production"},
        {"leg": "other"},
    ],
)
def test_bad_destination_fails_before_storage(
    monkeypatch: pytest.MonkeyPatch, mutation: dict[str, str]
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Opened storage before validating the destination")

    monkeypatch.setattr(storage, "_get_icechunk_storage", forbidden)
    factory = dataset(RUN, "jpp1").store_factory.model_copy(update=mutation)
    with pytest.raises(AssertionError):
        factory.primary_store(writable=True)


def test_preflight_has_no_remote_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Offline preflight accessed remote storage or credentials")

    monkeypatch.setattr(storage.StorageConfig, "load_storage_options", forbidden)
    monkeypatch.setattr(storage, "_get_icechunk_storage", forbidden)
    info = preflight(RUN, "jpp10")
    assert info["units"] == 38
    assert [len(group) for group in info["assignments"]] == [10, 10, 9, 9]
    assert (
        len(
            {
                (unit["init_time"], unit["variable"])
                for group in info["assignments"]
                for unit in group
            }
        )
        == 38
    )
    assert info["repository"].startswith(f"{ROOT}/{RUN}/data/jpp10/")
    assert info["repository"].endswith(f"v{VERSION}.icechunk")
    assert "/data/" not in info["evidence"]


def test_manifest_keeps_resources_and_stops_retries() -> None:
    job = build_manifest(RUN, "jpp1", IMAGE, 21_600)
    spec = job["spec"]
    assert spec["completions"] == 38
    assert spec["parallelism"] == 2
    assert spec["backoffLimitPerIndex"] == 0
    assert spec["maxFailedIndexes"] == 0
    assert spec["activeDeadlineSeconds"] == 21_600
    assert all(rule["action"] != "Ignore" for rule in spec["podFailurePolicy"]["rules"])
    pod = spec["template"]["spec"]
    worker = pod["containers"][0]
    assert worker["resources"] == {"requests": {"cpu": "3", "memory": "35G"}}
    assert worker["securityContext"]["capabilities"] == {"drop": ["ALL"]}
    assert worker["image"] == IMAGE
    assert worker["command"][:4] == [
        "python",
        "-m",
        "src.scripts.ifsens_profiling",
        "run",
    ]
    assert "DYNAMICAL_SENTRY_DSN" not in {entry["name"] for entry in worker["env"]}
    assert pod["terminationGracePeriodSeconds"] == 60
    pvc = next(volume for volume in pod["volumes"] if volume["name"] == "ephemeral-vol")
    assert (
        pvc["ephemeral"]["volumeClaimTemplate"]["spec"]["resources"]["requests"][
            "storage"
        ]
        == "30G"
    )


@pytest.mark.parametrize(
    "image", ["example.invalid/profiling:latest", "", IMAGE + " extra"]
)
def test_manifest_requires_digest(image: str) -> None:
    with pytest.raises(AssertionError):
        build_manifest(RUN, "jpp1", image, 21_600)


def test_manifest_diff_is_limited_to_pilot_controls() -> None:
    cron = next(
        item
        for item in dataset(RUN, "jpp1").operational_kubernetes_resources(IMAGE)
        if isinstance(item, ReformatCronJob)
    )
    expected = Job(
        command=["backfill", END.tz_localize(None).isoformat()],
        image=IMAGE,
        dataset_id=DATASET_ID,
        workers_total=38,
        parallelism=2,
        active_deadline=timedelta(hours=6),
        **cron.model_dump(
            include={
                "cpu",
                "memory",
                "shared_memory",
                "ephemeral_storage",
                "pod_active_deadline",
                "secret_names",
            }
        ),
    ).as_kubernetes_object()
    actual = build_manifest(RUN, "jpp1", IMAGE, 21_600)
    assert actual["metadata"]["name"] == f"ifsens-profile-{RUN}-jpp1"
    assert actual["spec"]["podFailurePolicy"]["rules"] == [
        rule
        for rule in expected["spec"]["podFailurePolicy"]["rules"]
        if rule["action"] != "Ignore"
    ]
    actual["metadata"] = expected["metadata"]
    for key in ("backoffLimitPerIndex", "maxFailedIndexes", "podFailurePolicy"):
        actual["spec"][key] = expected["spec"][key]
    actual_pod = actual["spec"]["template"]["spec"]
    expected_pod = expected["spec"]["template"]["spec"]
    assert actual_pod["containers"][0]["env"] == [
        entry
        for entry in expected_pod["containers"][0]["env"]
        if entry["name"] != "DYNAMICAL_SENTRY_DSN"
    ] + [
        {"name": "POD_UID", "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}}},
        {"name": "PROFILING_IMAGE", "value": IMAGE},
    ]
    actual_pod["terminationGracePeriodSeconds"] = expected_pod[
        "terminationGracePeriodSeconds"
    ]
    for key in ("command", "env"):
        actual_pod["containers"][0][key] = expected_pod["containers"][0][key]
    assert actual == expected


def test_template_preserves_all_encodings() -> None:
    pilot = ProfilingTemplate()
    production = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    assert pilot.data_vars == production.data_vars
    assert [
        (coord.name, coord.encoding.model_dump_json()) for coord in pilot.coords
    ] == [(coord.name, coord.encoding.model_dump_json()) for coord in production.coords]
    assert pilot.append_dim_coordinate_chunk_size() == 5840
    assert pilot.template_path() != production.template_path()
    assert pilot.append_dim_start == START.tz_localize(None)
    assert pilot.version == VERSION
    template = pilot.get_template(END.tz_localize(None))
    assert template.sizes["init_time"] == 2


def test_backfill_never_uses_production_factory_openers(
    factory: ProfilingStoreFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Pilot used a production store opener")

    pilot = dataset(RUN, "jpp1")
    monkeypatch.setattr(storage.StoreFactory, "primary_store", forbidden)
    monkeypatch.setattr(storage.StoreFactory, "icechunk_repos", forbidden)
    monkeypatch.setattr(icechunk.Repository, "open_or_create", forbidden)
    monkeypatch.setattr(
        pilot.region_job_class, "process_worker_jobs", lambda *_args, **_kwargs: {}
    )
    factory.bootstrap(0)
    pilot.backfill(
        END.tz_localize(None).to_pydatetime(),
        "local-pilot-factory-test",
        worker_index=0,
        workers_total=1,
    )
    repo = factory.open_owned_repository()
    assert len(list(repo.ancestry(branch="main"))) > 1


def test_evidence_verification_rejects_corruption(tmp_path: Path) -> None:
    metadata = {"job_name": "test", "workers": 2, "worker_index": 0}
    (tmp_path / "manifest-initial.json").write_text(json.dumps({"metadata": metadata}))
    payload = tmp_path / "resource-000000.json"
    payload.write_text("[]")
    for name in ("py-spy.speedscope.json", "workload-exit.json"):
        (tmp_path / name).write_text("{}")
    final = {
        "metadata": metadata,
        "returncode": 0,
        "evidence_error": None,
        "signal": None,
        "profile_samples": 1,
        "py_spy_returncode": 0,
        "killed_after_drain": False,
        "sha256": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in (payload.name, "py-spy.speedscope.json", "workload-exit.json")
        },
    }
    (tmp_path / "manifest-final.json").write_text(json.dumps(final))
    assert verify_evidence(tmp_path) == {
        "attempts": 1,
        "complete": 1,
        "incomplete": 0,
        "files_verified": 3,
        "jobs": {
            "test": {"workers": 2, "complete_workers": [0], "missing_workers": [1]}
        },
    }
    payload.write_text("corrupted")
    with pytest.raises(AssertionError, match="checksum mismatch"):
        verify_evidence(tmp_path)
