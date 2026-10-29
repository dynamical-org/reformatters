import argparse
import hashlib
import json
import multiprocessing
import os
import re
import sys
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from reformatters.common.iterating import get_worker_jobs
from reformatters.common.kubernetes import Job, ReformatCronJob
from reformatters.common.region_job import RegionJob
from src.scripts.ifsens_profiling.config import (
    BASELINE,
    DATASET_ID,
    END,
    START,
    Leg,
    ProfilingDataset,
    ProfilingTemplate,
    dataset,
)


def jobs_for(pilot: ProfilingDataset) -> Sequence[RegionJob[Any, Any]]:
    template = pilot.template_config.get_template(END.tz_localize(None))
    jobs = pilot.region_job_class.get_jobs(
        tmp_store=Path("data/profiling-chunks"),
        template_ds=template,
        append_dim=pilot.template_config.append_dim,
        all_data_vars=pilot.template_config.data_vars,
        reformat_job_name="profiling-preflight",
    )
    assert len(jobs) == 38
    assert len(pilot.template_config.data_vars) == 19
    assert {job.region.start for job in jobs} == {0, 1}
    assert all(job.region.stop == job.region.start + 1 for job in jobs)
    assert all(len(job.data_vars) == 1 for job in jobs)
    return jobs


def job_name(run_token: str, leg: Leg) -> str:
    return f"ifsens-profile-{run_token}-{leg}"


def preflight(run_token: str, leg: Leg) -> dict[str, Any]:
    pilot = dataset(run_token, leg)
    jobs = jobs_for(pilot)
    workers = 38 if leg == "jpp1" else 4
    assignments = [
        [
            {
                "init_time": (START + timedelta(days=job.region.start)).isoformat(),
                "variable": job.data_vars[0].path,
            }
            for job in get_worker_jobs(
                jobs,
                index,
                workers,
                worker_assignment=pilot.region_job_class.worker_assignment,
            )
        ]
        for index in range(workers)
    ]
    return {
        "baseline": BASELINE,
        "run_token": run_token,
        "leg": leg,
        "job_name": job_name(run_token, leg),
        "repository": pilot.store_factory.primary_url(),
        "evidence": pilot.store_factory.evidence_url,
        "coordination": pilot.store_factory._coordination_base_path(),  # noqa: SLF001
        "start": START.isoformat(),
        "end_exclusive": END.isoformat(),
        "units": len(jobs),
        "workers": workers,
        "parallelism": 2,
        "assignments": assignments,
        "public_output": True,
    }


def assert_image(image: str) -> None:
    assert re.fullmatch(r"[a-zA-Z0-9./:_-]+@sha256:[a-f0-9]{64}", image), (
        "A registry image pinned by index digest is required"
    )


def verify_evidence(directory: Path) -> dict[str, Any]:
    initial = list(directory.rglob("manifest-initial.json"))
    assert initial, "No initial manifests found"
    complete = checked = 0
    jobs: dict[str, dict[str, Any]] = {}
    for path in initial:
        metadata = json.loads(path.read_text())["metadata"]
        name = metadata["job_name"]
        workers = metadata["workers"]
        index = metadata["worker_index"]
        assert 0 <= index < workers
        coverage = jobs.setdefault(
            name, {"workers": workers, "complete_workers": set()}
        )
        assert coverage["workers"] == workers
        final_path = path.with_name("manifest-final.json")
        if not final_path.exists():
            continue
        final = json.loads(final_path.read_text())
        assert final["metadata"] == metadata, "Evidence attempt metadata differs"
        for name, expected in final["sha256"].items():
            assert Path(name).name == name, "Evidence filename is not a basename"
            actual = hashlib.sha256((path.parent / name).read_bytes()).hexdigest()
            assert actual == expected, f"Evidence checksum mismatch: {name}"
            checked += 1
        if (
            final["returncode"] == 0
            and final["evidence_error"] is None
            and final["signal"] is None
            and final["profile_samples"] > 0
            and final["py_spy_returncode"] == 0
            and not final["killed_after_drain"]
        ):
            names = set(final["sha256"])
            assert {"py-spy.speedscope.json", "workload-exit.json"} <= names
            assert any(name.startswith("resource-") for name in names)
            complete += 1
            coverage["complete_workers"].add(index)
    for coverage in jobs.values():
        coverage["missing_workers"] = sorted(
            set(range(coverage["workers"])) - coverage["complete_workers"]
        )
        coverage["complete_workers"] = sorted(coverage["complete_workers"])
    return {
        "attempts": len(initial),
        "complete": complete,
        "incomplete": len(initial) - complete,
        "files_verified": checked,
        "jobs": jobs,
    }


def build_manifest(
    run_token: str, leg: Leg, image: str, seconds: int
) -> dict[str, Any]:
    assert_image(image)
    assert 300 <= seconds <= 21_600
    info = preflight(run_token, leg)
    pilot = dataset(run_token, leg)
    cron = next(
        item
        for item in pilot.operational_kubernetes_resources(image)
        if isinstance(item, ReformatCronJob)
    )
    job = Job(
        command=["backfill", END.tz_localize(None).isoformat()],
        image=image,
        dataset_id=DATASET_ID,
        workers_total=info["workers"],
        parallelism=2,
        active_deadline=timedelta(seconds=seconds),
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
    job["metadata"] = {
        "name": info["job_name"],
        "namespace": "default",
        "labels": {
            "dynamical.org/run-kind": "profiling-pilot",
            "dynamical.org/profiling-run": run_token,
        },
    }
    spec = job["spec"]
    spec["backoffLimitPerIndex"] = 0
    spec["maxFailedIndexes"] = 0
    spec["podFailurePolicy"]["rules"] = [
        rule for rule in spec["podFailurePolicy"]["rules"] if rule["action"] != "Ignore"
    ]
    pod = spec["template"]["spec"]
    pod["terminationGracePeriodSeconds"] = 60
    container = pod["containers"][0]
    container["command"] = [
        "python",
        "-m",
        "src.scripts.ifsens_profiling",
        "run",
        "--run-token",
        run_token,
        "--leg",
        leg,
        "--image",
        image,
    ]
    container["env"] = [
        env for env in container["env"] if env["name"] != "DYNAMICAL_SENTRY_DSN"
    ]
    container["env"].extend(
        [
            {
                "name": "POD_UID",
                "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
            },
            {"name": "PROFILING_IMAGE", "value": image},
        ]
    )
    return job


def run_worker(run_token: str, leg: Leg, image: str, output_dir: Path) -> None:
    from src.scripts.ifsens_profiling.profiling import install_hooks  # noqa: PLC0415

    assert_image(image)
    info = preflight(run_token, leg)
    index = int(os.environ["WORKER_INDEX"])
    assert 0 <= index < info["workers"]
    assert int(os.environ["WORKERS_TOTAL"]) == info["workers"]
    assert os.environ["JOB_NAME"] == info["job_name"]
    pilot = dataset(run_token, leg)
    with install_hooks(output_dir):
        pilot.store_factory.bootstrap(index)
        pilot.backfill(
            END.tz_localize(None).to_pydatetime(),
            info["job_name"],
            worker_index=index,
            workers_total=info["workers"],
        )


def run_supervisor(run_token: str, leg: Leg, image: str) -> int:
    from src.scripts.ifsens_profiling.profiling import run_profiled  # noqa: PLC0415

    assert_image(image)
    assert os.environ["PROFILING_IMAGE"] == image
    info = preflight(run_token, leg)
    assert os.environ["JOB_NAME"] == info["job_name"]
    index = int(os.environ["WORKER_INDEX"])
    assert 0 <= index < info["workers"]
    assert int(os.environ["WORKERS_TOTAL"]) == info["workers"]
    pod_uid = os.environ["POD_UID"]
    assert re.fullmatch(r"[a-f0-9-]{36}", pod_uid)
    attempt = uuid4().hex
    output_dir = Path("/app/data/ifsens-profiles") / attempt
    factory = dataset(run_token, leg).store_factory
    metadata = {
        **{key: value for key, value in info.items() if key != "assignments"},
        "assignment": info["assignments"][index],
        "image": image,
        "source_sha": os.environ["PROFILING_SOURCE_SHA"],
        "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "worker_index": index,
        "pod_uid": pod_uid,
        "attempt": attempt,
    }
    return run_profiled(
        command=[
            sys.executable,
            "-m",
            "src.scripts.ifsens_profiling",
            "worker",
            "--run-token",
            run_token,
            "--leg",
            leg,
            "--image",
            image,
            "--output-dir",
            str(output_dir),
        ],
        artifact_url=f"{factory.evidence_url}/{info['job_name']}/{pod_uid}/{attempt}",
        fs=factory._coordination_fs(),  # noqa: SLF001
        metadata=metadata,
        output_dir=output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "update-template",
            "preflight",
            "manifest",
            "run",
            "worker",
            "verify-evidence",
        ],
    )
    parser.add_argument("--run-token")
    parser.add_argument("--leg", choices=["jpp1", "jpp10"], default="jpp1")
    parser.add_argument("--image")
    parser.add_argument("--seconds", type=int, default=21_600)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.command == "update-template":
        ProfilingTemplate().update_template()
        return
    if args.command == "verify-evidence":
        assert args.output_dir is not None
        sys.stdout.write(json.dumps(verify_evidence(args.output_dir), indent=2) + "\n")
        return
    assert args.run_token is not None
    leg = cast("Leg", args.leg)
    if args.command == "preflight":
        sys.stdout.write(json.dumps(preflight(args.run_token, leg), indent=2) + "\n")
    else:
        assert args.image is not None
        if args.command == "manifest":
            sys.stdout.write(
                json.dumps(
                    build_manifest(args.run_token, leg, args.image, args.seconds),
                    indent=2,
                )
                + "\n"
            )
        elif args.command == "worker":
            assert args.output_dir is not None
            run_worker(args.run_token, leg, args.image, args.output_dir)
        else:
            raise SystemExit(run_supervisor(args.run_token, leg, args.image))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
