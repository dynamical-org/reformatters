import argparse
import hashlib
import multiprocessing
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from reformatters.common.iterating import get_worker_jobs
from reformatters.common.kubernetes import Job
from reformatters.common.logging import get_logger
from reformatters.common.region_job import RegionJob
from reformatters.ecmwf.ifs_ens.profiling.config import (
    END,
    START,
    ProfilingDataset,
    dataset,
)

log = get_logger(__name__)
MODULE = "reformatters.ecmwf.ifs_ens.profiling.worker"


def assert_image(image: str) -> None:
    assert re.fullmatch(r"[A-Za-z0-9./_-]+:[a-f0-9]{40}", image), (
        "Profiling requires the image tagged with the deployed main commit"
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
    assert len({(job.region.start, job.data_vars[0].path) for job in jobs}) == 38
    return jobs


class ProfilingJob(Job):
    run_id: str

    def as_kubernetes_object(self) -> dict[str, Any]:
        assert_image(self.image)
        factory = dataset(self.run_id).store_factory
        factory.assert_destination()
        assert self.dataset_id == "ifsens-profile"
        assert self.workers_total in {4, 38}
        assert self.parallelism == 2
        assert list(self.command) == [
            "backfill",
            END.tz_localize(None).isoformat(),
            f"--filter-start={START.tz_localize(None).isoformat()}",
            f"--filter-end={END.tz_localize(None).isoformat()}",
        ]
        result = super().as_kubernetes_object()
        result["metadata"]["labels"] = {
            "dynamical.org/run-kind": "profiling",
            "dynamical.org/profiling-run": self.run_id,
        }
        spec = result["spec"]
        spec.update(
            activeDeadlineSeconds=21600,
            backoffLimitPerIndex=0,
            maxFailedIndexes=0,
            podReplacementPolicy="Failed",
        )
        spec["podFailurePolicy"]["rules"] = [
            rule
            for rule in spec["podFailurePolicy"]["rules"]
            if rule["action"] != "Ignore"
        ]
        pod = spec["template"]["spec"]
        pod["terminationGracePeriodSeconds"] = 60
        container = pod["containers"][0]
        container["command"] = [
            "python",
            "-m",
            MODULE,
            "run",
            "--run-id",
            self.run_id,
            "--job-name",
            self.job_name,
            "--workers",
            str(self.workers_total),
            "--source-sha",
            self.image.rsplit(":", 1)[1],
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
                {"name": "PROFILING_IMAGE", "value": self.image},
            ]
        )
        return result

    def report_destination(self) -> None:
        factory = dataset(self.run_id).store_factory
        factory.assert_destination()
        text = (
            "### Profiling backfill submission\n\n"
            f"- Job: `{self.job_name}`\n"
            f"- Run: `{self.run_id}`\n"
            f"- Data: `{factory.primary_url()}`\n"
            f"- Evidence: `{factory.evidence_url}`\n"
            f"- Source commit: `{self.image.rsplit(':', 1)[1]}`\n\n"
            "Submission succeeds only if the backfill command exits successfully. "
            "A completed workflow does not mean the Kubernetes Job is complete.\n"
        )
        log.info(text)
        if path := os.getenv("GITHUB_STEP_SUMMARY"):
            with Path(path).open("a") as stream:
                stream.write(text)


def _context(
    run_id: str, job_name: str, workers: int, source_sha: str
) -> tuple[ProfilingDataset, dict[str, Any]]:
    image = os.environ["PROFILING_IMAGE"]
    assert_image(image)
    assert image.rsplit(":", 1)[1] == source_sha
    assert workers in {4, 38}
    assert os.environ["JOB_NAME"] == job_name
    assert job_name.startswith("ifsens-profile-backfill-")
    assert int(os.environ["WORKERS_TOTAL"]) == workers
    index = int(os.environ["WORKER_INDEX"])
    assert 0 <= index < workers
    pilot = dataset(run_id)
    jobs = jobs_for(pilot)
    assignment = get_worker_jobs(
        jobs,
        index,
        workers,
        worker_assignment=pilot.region_job_class.worker_assignment,
    )
    return pilot, {
        "run_id": run_id,
        "job_name": job_name,
        "workers": workers,
        "worker_index": index,
        "source_sha": source_sha,
        "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "repository": pilot.store_factory.primary_url(),
        "evidence": pilot.store_factory.evidence_url,
        "assignment": [
            {
                "region": [job.region.start, job.region.stop],
                "variable": job.data_vars[0].path,
            }
            for job in assignment
        ],
    }


def run_worker(
    run_id: str, job_name: str, workers: int, source_sha: str, output_dir: Path
) -> None:
    from reformatters.common.profiling import install_hooks  # noqa: PLC0415

    pilot, info = _context(run_id, job_name, workers, source_sha)
    with install_hooks(output_dir):
        pilot.store_factory.bootstrap(info["worker_index"])
        pilot.backfill(
            END.tz_localize(None).to_pydatetime(),
            job_name,
            worker_index=info["worker_index"],
            workers_total=workers,
            filter_start=START.tz_localize(None).to_pydatetime(),
            filter_end=END.tz_localize(None).to_pydatetime(),
        )


def run_supervisor(run_id: str, job_name: str, workers: int, source_sha: str) -> int:
    from reformatters.common.profiling import run_profiled  # noqa: PLC0415

    pilot, metadata = _context(run_id, job_name, workers, source_sha)
    pod_uid = os.environ["POD_UID"]
    assert re.fullmatch(r"[a-f0-9-]{36}", pod_uid)
    attempt = uuid4().hex
    output = Path("/app/data/profiling") / attempt
    metadata.update(pod_uid=pod_uid, attempt=attempt)
    factory = pilot.store_factory
    return run_profiled(
        command=[
            sys.executable,
            "-m",
            MODULE,
            "worker",
            "--run-id",
            run_id,
            "--job-name",
            job_name,
            "--workers",
            str(workers),
            "--source-sha",
            source_sha,
            "--output-dir",
            str(output),
        ],
        artifact_url=f"{factory.evidence_url}/{job_name}/{pod_uid}/{attempt}",
        fs=factory._coordination_fs(),  # noqa: SLF001
        metadata=metadata,
        output_dir=output,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "worker"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "worker":
        assert args.output_dir is not None
        run_worker(
            args.run_id, args.job_name, args.workers, args.source_sha, args.output_dir
        )
    else:
        raise SystemExit(
            run_supervisor(args.run_id, args.job_name, args.workers, args.source_sha)
        )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
