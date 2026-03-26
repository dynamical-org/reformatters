import contextlib
import json
import os
import pickle
import subprocess
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypedDict, TypeVar

import icechunk
import numpy as np
import pandas as pd
import sentry_sdk
import sentry_sdk.crons
import typer
import xarray as xr
from pydantic import Field, computed_field

from reformatters.common import docker, storage, template_utils, validation
from reformatters.common.config import Config
from reformatters.common.config_models import DataVar
from reformatters.common.iterating import digest, get_worker_jobs, item
from reformatters.common.kubernetes import (
    CronJob,
    Job,
    ReformatCronJob,
    ValidationCronJob,
)
from reformatters.common.logging import get_logger
from reformatters.common.pydantic import FrozenBaseModel
from reformatters.common.region_job import RegionJob, SourceFileCoord
from reformatters.common.storage import StorageConfig, StoreFactory, get_local_tmp_store
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import DatetimeLike
from reformatters.common.zarr import copy_zarr_metadata

DATA_VAR = TypeVar("DATA_VAR", bound=DataVar[Any])
SOURCE_FILE_COORD = TypeVar("SOURCE_FILE_COORD", bound=SourceFileCoord)

log = get_logger(__name__)


class _SetupInfo(TypedDict, total=False):
    branch_name: str
    repo_snapshots: dict[str, str]


class DynamicalDataset(FrozenBaseModel, Generic[DATA_VAR, SOURCE_FILE_COORD]):
    """Top level class managing a dataset configuration and processing."""

    template_config: TemplateConfig[DATA_VAR]
    region_job_class: type[RegionJob[DATA_VAR, SOURCE_FILE_COORD]]

    primary_storage_config: StorageConfig
    replica_storage_configs: Sequence[StorageConfig] = Field(default_factory=tuple)

    @computed_field
    @property
    def store_factory(self) -> StoreFactory:
        return StoreFactory(
            primary_storage_config=self.primary_storage_config,
            replica_storage_configs=self.replica_storage_configs,
            dataset_id=self.dataset_id,
            template_config_version=self.template_config.version,
        )

    def operational_kubernetes_resources(self, image_tag: str) -> Iterable[CronJob]:
        """
        Return the kubernetes cron job definitions to operationally
        update and validate this dataset.

        Implementions should look similar this:
        ```
        operational_update_cron_job = ReformatCronJob(
            name=f"{self.dataset_id}-update",
            schedule=_OPERATIONAL_CRON_SCHEDULE,
            pod_active_deadline=timedelta(minutes=30),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="14",
            memory="30G",
            shared_memory="12G",
            ephemeral_storage="30G",
            secret_names=self.store_factory.k8s_secret_names(),
        )
        validation_cron_job = ValidationCronJob(
            name=f"{self.dataset_id}-validate",
            schedule=_VALIDATION_CRON_SCHEDULE,
            pod_active_deadline=timedelta(minutes=10),
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="1.3",
            memory="7G",
            secret_names=self.store_factory.k8s_secret_names(),
        )

        return [operational_update_cron_job, validation_cron_job]
        ```
        """
        raise NotImplementedError(
            f"Implement `operational_kubernetes_resources` on {self.__class__.__name__}"
        )

    def validators(self) -> Sequence[validation.DataValidator]:
        """
        Return a sequence of DataValidators to run on this dataset.

        Implementions should look similar this:
        ```
        return (
            validation.check_analysis_current_data,
            validation.check_analysis_recent_nans,
        )
        ```
        """
        raise NotImplementedError(
            f"Implement `validators` on {self.__class__.__name__}"
        )

    # ----- Most subclasses will not need to override the attributes and methods below -----

    @computed_field
    @property
    def dataset_id(self) -> str:
        return self.template_config.dataset_id

    def update_template(self) -> None:
        """Generate and persist the dataset template using the template_config."""
        self.template_config.update_template()

    def update(
        self,
        reformat_job_name: Annotated[str, typer.Argument(envvar="JOB_NAME")],
        *,
        worker_index: Annotated[int, typer.Argument(envvar="WORKER_INDEX")] = 0,
        workers_total: Annotated[int, typer.Argument(envvar="WORKERS_TOTAL")] = 1,
    ) -> None:
        """Update an existing dataset with the latest data."""
        with self._monitor(ReformatCronJob, reformat_job_name):
            tmp_store = self._tmp_store()

            all_jobs, template_ds = self.region_job_class.operational_update_jobs(
                primary_store=self.store_factory.primary_store(),
                tmp_store=tmp_store,
                get_template_fn=self._get_template,
                append_dim=self.template_config.append_dim,
                all_data_vars=self.template_config.data_vars,
                reformat_job_name=reformat_job_name,
            )

            self._process_region_jobs(
                all_jobs=all_jobs,
                worker_index=worker_index,
                workers_total=workers_total,
                reformat_job_name=reformat_job_name,
                template_ds=template_ds,
                tmp_store=tmp_store,
                update_template_with_results=True,
            )

        log.info(
            f"Operational update complete. Wrote to primary store: {self.store_factory.primary_store()} and replicas {self.store_factory.replica_stores()} replicas"
        )

    def backfill_kubernetes(
        self,
        append_dim_end: datetime,
        jobs_per_pod: int,
        max_parallelism: int,
        filter_start: datetime | None = None,
        filter_end: datetime | None = None,
        filter_contains: list[datetime] | None = None,
        filter_variable_names: list[str] | None = None,
        docker_image: str | None = None,
        overwrite_existing: bool = False,
    ) -> None:
        """Run dataset reformatting using Kubernetes index jobs."""
        assert self._can_run_in_kubernetes(), (
            "backfill_kubernetes is only supported in prod environment"
        )

        image_tag = docker_image or docker.build_and_push_image()

        template_ds = self._get_template(append_dim_end)

        if overwrite_existing:
            assert self.store_factory.all_stores_exist(), (
                "Not all stores exist, cannot run with overwrite_existing=True"
            )
            log.info("Writing to existing stores, skipping metadata write.")
        else:
            template_utils.write_metadata(template_ds, self.store_factory)

        num_jobs = len(
            self.region_job_class.get_jobs(
                tmp_store=self._tmp_store(),
                template_ds=template_ds,
                append_dim=self.template_config.append_dim,
                all_data_vars=self.template_config.data_vars,
                # The jobs returned from this call are only counted,
                # the real job named will be filled in by kubernetes
                reformat_job_name="placeholder",
                filter_start=pd.Timestamp(filter_start) if filter_start else None,
                filter_end=pd.Timestamp(filter_end) if filter_end else None,
                filter_contains=(
                    [pd.Timestamp(t) for t in filter_contains]
                    if filter_contains
                    else None
                ),
                filter_variable_names=filter_variable_names,
            )
        )
        workers_total = int(np.ceil(num_jobs / jobs_per_pod))
        parallelism = min(workers_total, max_parallelism)

        command = [
            "backfill",
            pd.Timestamp(append_dim_end).isoformat(),
        ]
        if filter_start is not None:
            command.append(f"--filter-start={filter_start.isoformat()}")
        if filter_end is not None:
            command.append(f"--filter-end={filter_end.isoformat()}")
        if filter_contains is not None:
            command.extend(
                f"--filter-contains={timestamp.isoformat()}"
                for timestamp in filter_contains
            )
        if filter_variable_names is not None:
            command.extend(
                f"--filter-variable-names={variable_name}"
                for variable_name in filter_variable_names
            )

        # In an attempt to keep the subclassing API simpler, we are keeping
        # all resource needs defined right in `operational_kubernetes_resources`.
        # If for some reason there are _multiple_ ReformatCronJobs returned from
        # that we'll need to revisit the logic below or this approach.
        reformat_jobs = [
            r
            for r in self.operational_kubernetes_resources(image_tag)
            if isinstance(r, ReformatCronJob)
        ]
        assert len(reformat_jobs) == 1, (
            f"Can't infer kubernetes resources for backfill job from {reformat_jobs}."
        )
        reformat_job = reformat_jobs[0]

        kubernetes_job = Job(
            command=command,
            image=image_tag,
            dataset_id=self.dataset_id,
            workers_total=workers_total,
            parallelism=parallelism,
            **reformat_job.model_dump(
                include={
                    "cpu",
                    "memory",
                    "shared_memory",
                    "ephemeral_storage",
                    "pod_active_deadline",
                    "secret_names",
                }
            ),
        )
        subprocess.run(
            ["/usr/bin/kubectl", "apply", "-f", "-"],
            input=json.dumps(kubernetes_job.as_kubernetes_object()),
            text=True,
            check=True,
        )

        log.info(f"Submitted kubernetes job {kubernetes_job.job_name}")

    def backfill_local(
        self,
        append_dim_end: datetime,
        *,
        filter_start: datetime | None = None,
        filter_end: datetime | None = None,
        filter_contains: list[datetime] | None = None,
        filter_variable_names: list[str] | None = None,
    ) -> None:
        """Run dataset reformatting locally in this process."""
        assert Config.is_dev or Config.is_test, (
            "backfill_local is only supported in dev or test environments"
        )

        template_ds = self._get_template(append_dim_end)
        template_utils.write_metadata(template_ds, self.store_factory)

        self.backfill(
            append_dim_end,
            reformat_job_name="local",
            worker_index=0,
            workers_total=1,
            filter_start=filter_start,
            filter_end=filter_end,
            filter_contains=filter_contains,
            filter_variable_names=filter_variable_names,
        )
        log.info(f"Done writing to {self.store_factory.primary_store()}")

    def backfill(
        self,
        append_dim_end: datetime,
        reformat_job_name: Annotated[str, typer.Argument(envvar="JOB_NAME")],
        *,
        worker_index: Annotated[int, typer.Argument(envvar="WORKER_INDEX")],
        workers_total: Annotated[int, typer.Argument(envvar="WORKERS_TOTAL")],
        filter_start: datetime | None = None,
        filter_end: datetime | None = None,
        filter_contains: list[datetime] | None = None,
        filter_variable_names: list[str] | None = None,
    ) -> None:
        """Orchestrate running RegionJob instances."""
        template_ds = self._get_template(append_dim_end)
        tmp_store = self._tmp_store()

        all_jobs = self.region_job_class.get_jobs(
            tmp_store=tmp_store,
            template_ds=template_ds,
            append_dim=self.template_config.append_dim,
            all_data_vars=self.template_config.data_vars,
            reformat_job_name=reformat_job_name,
            filter_start=pd.Timestamp(filter_start) if filter_start else None,
            filter_end=pd.Timestamp(filter_end) if filter_end else None,
            filter_contains=(
                [pd.Timestamp(t) for t in filter_contains] if filter_contains else None
            ),
            filter_variable_names=filter_variable_names,
        )

        self._process_region_jobs(
            all_jobs=all_jobs,
            worker_index=worker_index,
            workers_total=workers_total,
            reformat_job_name=reformat_job_name,
            template_ds=template_ds,
            tmp_store=tmp_store,
            update_template_with_results=False,
        )

    def _process_region_jobs(
        self,
        all_jobs: Sequence[RegionJob[DATA_VAR, SOURCE_FILE_COORD]],
        worker_index: int,
        workers_total: int,
        reformat_job_name: str,
        template_ds: xr.Dataset,
        tmp_store: Path,
        *,
        update_template_with_results: bool,
    ) -> None:
        """Shared processing loop for both updates and backfills.

        Coordinates parallel writes across multiple workers:
        - Icechunk stores: uses a temp branch so readers on "main" never see partial data
        - Zarr v3 stores: defers metadata write until all workers finish
        """
        is_first = worker_index == 0
        is_last = worker_index == workers_total - 1
        my_jobs = get_worker_jobs(all_jobs, worker_index, workers_total)

        jobs_summary = ", ".join(repr(j) for j in my_jobs)
        log.info(
            f"This is {worker_index = }, {workers_total = }, {len(my_jobs)} jobs, {jobs_summary}"
        )

        icechunk_repos = self.store_factory.icechunk_repos(sort="primary-first")
        has_icechunk = len(icechunk_repos) > 0

        # ── SETUP / WAIT ─────────────────────────────────────────────
        setup_info = self._parallel_setup(
            is_first,
            workers_total,
            reformat_job_name,
            template_ds,
            tmp_store,
            icechunk_repos,
        )
        branch_name = setup_info.get("branch_name", "main")

        # ── GET STORES AND PROCESS JOBS ───────────────────────────────
        icechunk_branch = branch_name if has_icechunk else "main"
        primary_store = self.store_factory.primary_store(
            writable=True, branch=icechunk_branch
        )
        replica_stores = self.store_factory.replica_stores(
            writable=True, branch=icechunk_branch
        )

        all_results: dict[str, Sequence[SOURCE_FILE_COORD]] = {}
        for job in my_jobs:
            template_utils.write_metadata(job.template_ds, job.tmp_store)
            results = job.process(
                primary_store=primary_store, replica_stores=replica_stores
            )
            all_results.update(results)

        storage.commit_if_icechunk(
            f"Worker {worker_index} at {pd.Timestamp.now(tz='UTC').isoformat()}",
            primary_store,
            replica_stores,
        )

        # ── WRITE RESULTS ─────────────────────────────────────────────
        if workers_total > 1:
            self.store_factory.write_coordination_file(
                reformat_job_name,
                f"results/worker-{worker_index}.pkl",
                pickle.dumps(all_results),
            )

        # ── FINALIZE (last worker) ────────────────────────────────────
        if is_last:
            merged_results = self._collect_results(
                all_results, reformat_job_name, workers_total
            )
            self._finalize(
                all_jobs=all_jobs,
                merged_results=merged_results,
                reformat_job_name=reformat_job_name,
                template_ds=template_ds,
                tmp_store=tmp_store,
                setup_info=setup_info,
                workers_total=workers_total,
                update_template_with_results=update_template_with_results,
            )

    def _parallel_setup(
        self,
        is_first: bool,
        workers_total: int,
        reformat_job_name: str,
        template_ds: xr.Dataset,
        tmp_store: Path,
        icechunk_repos: list[tuple[str, Any]],
    ) -> _SetupInfo:
        has_icechunk = len(icechunk_repos) > 0

        if is_first:
            template_utils.write_metadata(template_ds, tmp_store)

            setup_info: _SetupInfo = {}
            # Icechunk: create temp branch, write full metadata, commit on branch.
            # This expands the dataset on the temp branch while readers on "main" are unaffected.
            # Branch name is deterministic so worker 0 retries reuse the same branch.
            if has_icechunk:
                branch_name = f"_job_{reformat_job_name}"
                setup_info["branch_name"] = branch_name
                setup_info["repo_snapshots"] = {}
                for role, repo in icechunk_repos:
                    snapshot = repo.lookup_branch("main")
                    setup_info["repo_snapshots"][role] = snapshot
                    try:
                        repo.create_branch(branch_name, snapshot)
                    except (icechunk.IcechunkError, ValueError):
                        # Branch already exists from a previous worker 0 attempt — reuse it.
                        # Safe because write_metadata(mode="w") overwrites the branch state.
                        log.info(
                            f"Branch {branch_name} already exists on {role}, reusing"
                        )
                # Write metadata to all icechunk stores on the temp branch.
                # Use write_metadata (not copy_zarr_metadata) so to_zarr creates/expands
                # the full zarr structure including arrays and coordinates.
                icechunk_primary = self.store_factory.primary_store(
                    writable=True, branch=branch_name
                )
                icechunk_replicas = self.store_factory.replica_stores(
                    writable=True, branch=branch_name
                )
                for ic_store in [icechunk_primary, *icechunk_replicas]:
                    template_utils.write_metadata(
                        template_ds, ic_store, mode="w", skip_icechunk_commit=True
                    )
                storage.commit_if_icechunk(
                    "expand metadata for parallel update",
                    icechunk_primary,
                    icechunk_replicas,
                )
            # Zarr v3: do NOT expand (readers would see empty holes)

            if workers_total > 1:
                self.store_factory.write_coordination_file(
                    reformat_job_name, "setup/ready.pkl", pickle.dumps(setup_info)
                )
            return setup_info

        # Poll until worker 0 completes setup. Rely on kubernetes pod_active_deadline for timeout.
        if workers_total > 1:
            while not self.store_factory.read_all_coordination_files(
                reformat_job_name, "setup"
            ):
                log.info("Waiting for worker 0 to complete setup...")
                time.sleep(5)
            return pickle.loads(  # noqa: S301
                self.store_factory.read_all_coordination_files(
                    reformat_job_name, "setup"
                )[0]
            )

        return _SetupInfo()

    def _collect_results(
        self,
        local_results: dict[str, Sequence[SOURCE_FILE_COORD]],
        reformat_job_name: str,
        workers_total: int,
    ) -> Mapping[str, Sequence[SOURCE_FILE_COORD]]:
        if workers_total == 1:
            return local_results

        # Poll until all workers write results. Rely on kubernetes pod_active_deadline for timeout.
        while (
            len(
                self.store_factory.read_all_coordination_files(
                    reformat_job_name, "results"
                )
            )
            < workers_total
        ):
            log.info("Waiting for all workers to complete...")
            time.sleep(10)

        merged: dict[str, Sequence[SOURCE_FILE_COORD]] = {}
        for data in self.store_factory.read_all_coordination_files(
            reformat_job_name, "results"
        ):
            merged.update(pickle.loads(data))  # noqa: S301
        return merged

    def _finalize(
        self,
        all_jobs: Sequence[RegionJob[DATA_VAR, SourceFileCoord]],
        merged_results: Mapping[str, Sequence[SOURCE_FILE_COORD]],
        reformat_job_name: str,
        template_ds: xr.Dataset,
        tmp_store: Path,
        setup_info: _SetupInfo,
        workers_total: int,
        *,
        update_template_with_results: bool,
    ) -> None:
        if update_template_with_results:
            assert len(all_jobs) > 0
            updated_template = all_jobs[0].update_template_with_results(merged_results)
            template_utils.write_metadata(updated_template, tmp_store)
        else:
            updated_template = template_ds

        # Icechunk: write final (possibly trimmed) metadata on temp branch, commit, reset main.
        # Uses copy_zarr_metadata (not write_metadata) because the zarr arrays already exist
        # on the temp branch from _parallel_setup — we only need to update the metadata files.
        # Process replicas before primary so primary (which drives future work) is last to update.
        # Only runs when setup created a temp branch (branch_name != "main").
        branch_name = setup_info.get("branch_name", "main")
        if branch_name != "main":
            replicas_first = self.store_factory.icechunk_repos(sort="primary-last")
            # First pass: commit final metadata and reset main on each repo.
            # If a previous attempt already reset main for a repo, skip it.
            for role, repo in replicas_first:
                original_snapshot = setup_info.get("repo_snapshots", {}).get(role)
                current_main = repo.lookup_branch("main")
                if current_main != original_snapshot:
                    log.info(
                        f"Skipping {role}: main already moved past original snapshot"
                    )
                    continue
                session = repo.writable_session(branch_name)
                copy_zarr_metadata(
                    updated_template, tmp_store, session.store, icechunk_only=True
                )
                new_snapshot = session.commit(
                    "finalize parallel update",
                    rebase_with=icechunk.ConflictDetector(),
                )
                repo.reset_branch(
                    "main", new_snapshot, from_snapshot_id=original_snapshot
                )
            # Second pass: clean up temp branches.
            for _role, repo in replicas_first:
                with contextlib.suppress(icechunk.IcechunkError, ValueError):
                    repo.delete_branch(branch_name)

        # Zarr v3: copy metadata now (makes data visible to readers).
        # Only needed for updates where metadata was deferred during setup.
        # For backfills, metadata is written before workers start.
        if update_template_with_results:
            zarr3_primary = self.store_factory.primary_store(writable=True)
            zarr3_replicas = self.store_factory.replica_stores(writable=True)
            copy_zarr_metadata(
                updated_template,
                tmp_store,
                zarr3_primary,
                replica_stores=zarr3_replicas,
                zarr3_only=True,
            )

        if workers_total > 1:
            self.store_factory.clear_coordination_files(reformat_job_name)

    def validate_dataset(
        self,
        reformat_job_name: Annotated[str, typer.Argument(envvar="JOB_NAME")],
    ) -> None:
        """Validate the dataset, raising an exception if it is invalid."""
        with self._monitor(ValidationCronJob, reformat_job_name):
            primary_store = self.store_factory.primary_store()
            primary_store_validators = list(self.validators())
            primary_store_validators.append(
                partial(validation.check_for_expected_shards, primary_store)
            )

            validation.validate_dataset(
                primary_store,
                validators=primary_store_validators,
            )
            log.info(f"Done validating {primary_store}")

            for replica_store in self.store_factory.replica_stores():
                replica_store_validators = list(self.validators())
                replica_store_validators.append(
                    partial(validation.check_for_expected_shards, replica_store)
                )
                replica_store_validators.append(
                    partial(
                        validation.compare_replica_and_primary,
                        self.template_config.append_dim,
                        xr.open_zarr(replica_store, chunks=None),
                    )
                )

                validation.validate_dataset(
                    replica_store,
                    validators=replica_store_validators,
                )
                log.info(f"Done validating {replica_store}")

    def get_cli(
        self,
    ) -> typer.Typer:
        """Create a CLI app with dataset commands"""
        app = typer.Typer()
        app.command()(self.update_template)
        app.command()(self.update)
        app.command()(self.backfill_kubernetes)
        app.command()(self.backfill_local)
        app.command()(self.backfill)
        # Avoid method name conflict with pydantic's validate while keeping cli commands consistent
        app.command("validate")(self.validate_dataset)
        return app

    def _tmp_store(self) -> Path:
        return get_local_tmp_store()

    def _get_template(self, append_dim_end: DatetimeLike) -> xr.Dataset:
        return self.template_config.get_template(pd.Timestamp(append_dim_end))

    def _can_run_in_kubernetes(self) -> bool:
        # This is a method to support testing without changing the Config.env
        return Config.is_prod

    @contextmanager
    def _monitor(
        self,
        cron_type: type[CronJob],
        reformat_job_name: str,
        cron_job_name: str | None = None,
    ) -> Iterator[None]:
        # Don't require operational_kubernetes_resources to be defined unless sentry reporting is enabled
        if not Config.is_sentry_enabled:
            yield
            return

        # Find the cron job that matches the type (and name if provided). There should be exactly one.
        cron_jobs = self.operational_kubernetes_resources("placeholder-image-tag")
        if cron_job_name:
            cron_jobs = (c for c in cron_jobs if c.name == cron_job_name)
        cron_jobs = (c for c in cron_jobs if isinstance(c, cron_type))
        cron_job = item(cron_jobs)

        # Use the actual cronjob name from k8s env when available. This ensures
        # staging cronjobs report to their own Sentry monitor, not production's.
        monitor_slug = os.getenv("CRON_JOB_NAME") or cron_job.name

        def capture_checkin(status: Literal["ok", "in_progress", "error"]) -> None:
            sentry_sdk.crons.capture_checkin(
                monitor_slug=monitor_slug,
                check_in_id=digest([reformat_job_name], length=32),
                status=status,
                monitor_config={
                    "schedule": {"type": "crontab", "value": cron_job.schedule},
                    "timezone": "UTC",
                    "checkin_margin": 10,
                    "max_runtime": int(
                        cron_job.pod_active_deadline.total_seconds() / 60
                    ),
                    "failure_issue_threshold": 1,
                    "recovery_threshold": 1,
                },
            )

        capture_checkin("in_progress")
        try:
            yield
        except Exception:
            capture_checkin("error")
            raise
        else:
            capture_checkin("ok")
