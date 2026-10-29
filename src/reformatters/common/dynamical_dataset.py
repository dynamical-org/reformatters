import json
import math
import os
import subprocess
import time
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, ClassVar, Generic, Literal, Self, TypeVar

import icechunk
import numpy as np
import pandas as pd
import typer
import xarray as xr
import zarr.errors
from icechunk.store import IcechunkStore
from kubernetes import client
from pydantic import Field, computed_field, model_validator

from reformatters.common import (
    kubernetes,
    parallel_coordination,
    template_utils,
    validation,
)
from reformatters.common.config import Config
from reformatters.common.config_models import DataVar
from reformatters.common.iterating import get_worker_jobs, item
from reformatters.common.kubernetes import (
    CronJob,
    Job,
    ReformatCronJob,
    get_deployed_cronjob_image,
)
from reformatters.common.logging import get_logger
from reformatters.common.materialized_region_job import MaterializedRegionJob
from reformatters.common.materialized_update_trigger import MaterializedUpdateTrigger
from reformatters.common.operational import OperationalResources
from reformatters.common.region_job import (
    RegionJob,
    SourceFileCoord,
    SourceFileResult,
)
from reformatters.common.staging import staging_cronjob_name
from reformatters.common.storage import (
    DatasetFormat,
    IcechunkVirtualConfig,
    StorageConfig,
    StoreFactory,
    get_local_tmp_store,
)
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import DatetimeLike, Timestamp
from reformatters.common.virtual_region_job import VirtualRegionJob

DATA_VAR = TypeVar("DATA_VAR", bound=DataVar[Any])
SOURCE_FILE_COORD = TypeVar("SOURCE_FILE_COORD", bound=SourceFileCoord)

log = get_logger(__name__)


class DynamicalDataset(OperationalResources, Generic[DATA_VAR, SOURCE_FILE_COORD]):
    """Top level class managing a dataset configuration and processing."""

    template_config: TemplateConfig[DATA_VAR]
    region_job_class: type[
        MaterializedRegionJob[DATA_VAR, SOURCE_FILE_COORD]
        | VirtualRegionJob[DATA_VAR, SOURCE_FILE_COORD]
    ]

    primary_storage_config: StorageConfig
    replica_storage_configs: Sequence[StorageConfig] = Field(default_factory=tuple)
    icechunk_virtual_config: IcechunkVirtualConfig | None = None
    materialized_update_triggers: ClassVar[tuple[MaterializedUpdateTrigger, ...]] = ()
    virtual_poll_deadline_grace: ClassVar[timedelta] = timedelta(seconds=30)

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """
        Return the kubernetes cron job definitions to operationally
        update and validate this dataset.

        Implementations should look similar to this:
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

        return [operational_update_cron_job]
        ```
        """
        raise NotImplementedError(
            f"Implement `operational_kubernetes_resources` on {self.__class__.__name__}"
        )

    def validators(self) -> Sequence[validation.Validator]:
        """
        Return the operational validation checks to run on this dataset.

        Implementations should look similar to this:
        ```
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=2)),
            validation.CheckRecentNans(),
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

    @computed_field
    @property
    def store_factory(self) -> StoreFactory:
        return StoreFactory(
            primary_storage_config=self.primary_storage_config,
            replica_storage_configs=self.replica_storage_configs,
            dataset_id=self.dataset_id,
            template_config_version=self.template_config.version,
            icechunk_virtual_config=self.icechunk_virtual_config,
        )

    def num_variable_groups(self) -> int:
        """Number of variable groups for parallel updates."""
        return self.region_job_class.num_variable_groups(self.template_config.data_vars)

    def update(
        self,
        reformat_job_name: Annotated[str, typer.Argument(envvar="JOB_NAME")],
        *,
        worker_index: Annotated[int, typer.Argument(envvar="WORKER_INDEX")] = 0,
        workers_total: Annotated[int, typer.Argument(envvar="WORKERS_TOTAL")] = 1,
    ) -> None:
        """Update an existing dataset with the latest data."""
        is_first = worker_index == 0
        is_last = worker_index == workers_total - 1
        is_virtual = issubclass(self.region_job_class, VirtualRegionJob)
        job_fire_time: Timestamp | None = None
        if (
            is_virtual
            and (
                cron_job := self._operational_cron_job(ReformatCronJob)
            ).concurrency_policy
            == "Allow"
        ):
            job_fire_time = cron_job.previous_fire_time(pd.Timestamp.now())
        materialized_job = None
        job_reference_time = (
            None if is_virtual else pd.Timestamp.now(tz="UTC").tz_localize(None)
        )
        job_uid: str | None = None
        with self._monitor(
            ReformatCronJob,
            reformat_job_name,
            send_in_progress=is_first,
            send_result=is_last,
        ):
            if not is_virtual and Config.is_prod:
                cron = self._operational_cron_job(ReformatCronJob)
                if cron.concurrency_policy == "Allow":
                    materialized_job = kubernetes.get_job(reformat_job_name)
                    assert materialized_job.metadata is not None
                    assert materialized_job.metadata.creation_timestamp is not None
                    assert materialized_job.metadata.uid is not None
                    job_uid = materialized_job.metadata.uid
                    job_reference_time = pd.Timestamp(
                        materialized_job.metadata.creation_timestamp
                    ).tz_convert(None)
                    if not self._prepare_and_wait_materialized_update(
                        materialized_job, cron, is_first=is_first
                    ):
                        return
            tmp_store = self._tmp_store()

            all_jobs, template_ds = self._operational_update_jobs(
                reformat_job_name,
                tmp_store,
                job_fire_time=job_fire_time,
                job_reference_time=job_reference_time,
            )

            if is_virtual:
                # Virtual operational updates stream one worker per Job (see docs/parallel_processing.md).
                if is_first:
                    self._assert_no_structural_drift(template_ds)
                self._run_virtual_operational_update(
                    all_jobs, worker_index, workers_total, job_fire_time=job_fire_time
                )
            else:
                self._process_region_jobs(
                    all_jobs=all_jobs,
                    worker_index=worker_index,
                    workers_total=workers_total,
                    reformat_job_name=reformat_job_name,
                    template_ds=template_ds,
                    tmp_store=tmp_store,
                    update_template_with_results=True,
                    job_uid=job_uid,
                )

        if is_last:
            self._validate_after_update(
                reformat_job_name,
                materialized_job=materialized_job,
                job_fire_time=job_fire_time,
            )

        log.info(
            f"Operational update complete. Wrote to primary store {self.store_factory.primary_store()} and replicas {self.store_factory.replica_stores()}"
        )

    def _prepare_and_wait_materialized_update(
        self, job: client.V1Job, cron: CronJob, *, is_first: bool
    ) -> bool:
        """Return whether this Job should write, after recording or reusing its decision and, if it runs, waiting for older writers."""
        factory = self.store_factory
        assert job.metadata is not None
        name, uid = job.metadata.name, job.metadata.uid
        assert name is not None
        assert uid is not None
        cronjob_name = os.environ["CRON_JOB_NAME"]
        state = parallel_coordination.read_job_state(factory, job)
        if is_first and state is None:
            fire = cron.previous_fire_time(pd.Timestamp(kubernetes.job_order(job)[0]))
            fire_interval = cron.next_fire_time(fire) - fire
            is_backstop = kubernetes.job_kind(job) == "scheduled" and any(
                kubernetes.job_kind(other) == "trigger"
                and timedelta(0)
                <= kubernetes.job_order(job)[0] - kubernetes.job_order(other)[0]
                <= fire_interval / 2
                for other in kubernetes.list_cronjob_jobs(cronjob_name)
            )
            if is_backstop:
                parallel_coordination.wait_for_older_jobs(factory, cronjob_name, job)
                if self._current_data_is_published():
                    state = "skip"
            state = state or "run"
            parallel_coordination.write_job_state(factory, name, uid, state)
            log.info("Job %s decision: %s", name, state)
        while state is None:
            time.sleep(5)
            state = parallel_coordination.read_job_state(factory, job)
        if state != "run":
            return False
        parallel_coordination.wait_for_older_jobs(factory, cronjob_name, job)
        return True

    def _current_data_is_published(self) -> bool:
        check = item(
            check
            for check in self.validators()
            if isinstance(check, validation.CheckCurrentData)
        )
        store = self.store_factory.primary_store()
        with validation.open_flattened_dataset(
            store, consolidated=not isinstance(store, IcechunkStore)
        ) as ds:
            return check.check(
                validation.ValidationContext(
                    store=store,
                    ds=ds,
                    append_dim=self.template_config.append_dim,
                    append_dim_frequency=self.template_config.append_dim_frequency,
                )
            ).passed

    def _validate_after_update(
        self,
        reformat_job_name: str,
        *,
        materialized_job: client.V1Job | None = None,
        job_fire_time: Timestamp | None = None,
    ) -> None:
        try:
            self._validate_with_monitor(reformat_job_name, job_fire_time=job_fire_time)
        except Exception as error:
            log.exception("Operational validation failed for %s", reformat_job_name)
            if (
                (
                    materialized_job is not None
                    or isinstance(error, validation.OperationalValidationError)
                )
                and Config.is_prod
                and not (
                    issubclass(self.region_job_class, VirtualRegionJob)
                    and self._operational_cron_job(ReformatCronJob).concurrency_policy
                    == "Allow"
                )
            ):
                should_retry, retry_job_name = kubernetes.retry_job_name(
                    reformat_job_name,
                    max_retries=16 if materialized_job is not None else 1,
                )
                if should_retry and (
                    materialized_job is None
                    or parallel_coordination.backoff_before_repass(
                        self.store_factory,
                        os.environ["CRON_JOB_NAME"],
                        materialized_job,
                        int(retry_job_name.rsplit("-r", 1)[1]),
                    )
                ):
                    try:
                        kubernetes.create_job_from_cronjob(
                            os.environ["CRON_JOB_NAME"],
                            retry_job_name,
                            skip_if_next_run_within_deadline=materialized_job is None,
                        )
                    except Exception:
                        log.exception("Could not submit %s", retry_job_name)
            # Stop with exit code that fails the job rather than having this pod retry
            raise typer.Exit(kubernetes.VALIDATION_FAILURE_EXIT_CODE) from error

    def backfill_kubernetes(
        self,
        append_dim_end: datetime | None = None,
        jobs_per_pod: int = 2,
        max_parallelism: int = 10,
        filter_start: datetime | None = None,
        filter_end: datetime | None = None,
        filter_contains: list[datetime] | None = None,
        filter_variable_names: list[str] | None = None,
        docker_image: str | None = None,
        overwrite_chunks: bool = False,
        overwrite_metadata: bool = False,
    ) -> None:
        """Run dataset reformatting using Kubernetes index jobs.

        See docs/backfill.md for usage.
        """
        assert self._can_run_in_kubernetes(), (
            "backfill_kubernetes is only supported in prod environment"
        )

        overwrite = overwrite_chunks or overwrite_metadata
        existing_ds = self._open_existing_store()
        template_ds, resolved_end = self._resolve_backfill_template(
            existing_ds,
            append_dim_end,
            overwrite_chunks=overwrite_chunks,
            overwrite_metadata=overwrite_metadata,
        )

        if overwrite_metadata and not overwrite_chunks:
            no_effect_options = [
                filter_start,
                filter_end,
                filter_contains,
                filter_variable_names,
                docker_image,
            ]
            assert not any(no_effect_options), (
                "--overwrite-metadata without --overwrite-chunks refreshes metadata "
                "in place without launching workers; the filter and --docker-image "
                "options would have no effect"
            )
            template_utils.refresh_store_metadata(
                self.store_factory,
                template_ds,
                self.template_config.append_dim,
                self._tmp_store(),
                consolidated=self.region_job_class.consolidated_metadata,
            )
            log.info("Metadata refresh complete, no chunk data written.")
            return

        reformat_jobs = [
            r
            for r in self.operational_kubernetes_resources(image_tag="placeholder")
            if isinstance(r, ReformatCronJob)
        ]
        assert len(reformat_jobs) == 1, (
            f"Can't infer kubernetes resources for backfill job from {reformat_jobs}."
        )
        reformat_job = reformat_jobs[0]

        image_tag = docker_image or get_deployed_cronjob_image(reformat_job.name)
        log.info(f"Using image {image_tag}")

        if overwrite:
            assert self.store_factory.all_stores_exist(), (
                "Not all stores exist, cannot run an overwrite backfill"
            )
            log.info("Writing into existing stores.")
        elif not self.store_factory.all_stores_icechunk():
            # Write metadata to final store. Required for Zarr v3 only, Icechunk metadata is written in parallel_setup.
            template_utils.write_metadata(template_ds, self.store_factory)

        num_jobs = len(
            self.region_job_class.get_jobs(
                tmp_store=self._tmp_store(),
                template_ds=template_ds,
                append_dim=self.template_config.append_dim,
                all_data_vars=self.template_config.data_vars,
                # The jobs returned from this call are only counted,
                # the real job name will be filled in by kubernetes
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
            resolved_end.isoformat(),
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
        if overwrite_chunks:
            command.append("--overwrite-chunks")
        if overwrite_metadata:
            command.append("--overwrite-metadata")

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
        # Write metadata to final store. Required for Zarr v3 only, Icechunk metadata is written in parallel_setup.
        if not self.store_factory.all_stores_icechunk():
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
        overwrite_chunks: bool = False,
        overwrite_metadata: bool = False,
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
            overwrite_chunks=overwrite_chunks,
            overwrite_metadata=overwrite_metadata,
        )

    def _process_region_jobs(
        self,
        all_jobs: Sequence[RegionJob[DATA_VAR, SOURCE_FILE_COORD]],
        worker_index: int,
        workers_total: int,
        reformat_job_name: str,
        template_ds: xr.DataTree,
        tmp_store: Path,
        *,
        update_template_with_results: bool,
        overwrite_chunks: bool = False,
        overwrite_metadata: bool = False,
        job_uid: str | None = None,
    ) -> None:
        """Shared processing loop for both updates and backfills.

        Coordinates parallel writes across multiple workers (see docs/parallel_processing.md):
        - Icechunk stores: uses a temp branch so readers on "main" never see partial data
        - Zarr v3 stores: defers metadata write until all workers finish
        """
        is_first = worker_index == 0
        is_last = worker_index == workers_total - 1
        worker_jobs = get_worker_jobs(
            all_jobs,
            worker_index,
            workers_total,
            worker_assignment=self.region_job_class.worker_assignment,
        )

        jobs_summary = ", ".join(repr(j) for j in worker_jobs)
        log.info(
            f"This is {worker_index = }, {workers_total = }, "
            f"{len(worker_jobs)} of {len(all_jobs)} total jobs, {jobs_summary}"
        )

        icechunk_repos = self.store_factory.icechunk_repos(sort="primary-first")
        has_icechunk = len(icechunk_repos) > 0
        branch_name = f"_job_{reformat_job_name}" if has_icechunk else "main"

        # 0. Guards run on worker 0 before any writes; failing here leaves the live
        # archive untouched. An operational update must not change the structure of
        # the already-published store; an overwrite backfill must not corrupt it
        # (structural drift, trimming, or unrequested new arrays / expansion).
        overwrite = overwrite_chunks or overwrite_metadata
        if is_first and update_template_with_results:
            self._assert_no_structural_drift(template_ds)
        if is_first and overwrite:
            template_utils.assert_safe_overwrite(
                template_ds,
                self.store_factory.open_primary_datatree(),
                self.template_config.append_dim,
                allow_new_arrays=overwrite_metadata,
                allow_expansion=overwrite_chunks and overwrite_metadata,
            )

        # Overwrite backfills write template metadata into a store that has live,
        # job-written coordinate values (e.g. ingested_forecast_length); never
        # overwrite those with the template's nulls.
        exclude_coord_value_chunks = (
            template_utils.store_written_coords(template_ds) if overwrite else set()
        )

        # 1. Set up / wait for setup
        setup_info = parallel_coordination.parallel_setup(
            self.store_factory,
            is_first=is_first,
            workers_total=workers_total,
            reformat_job_name=reformat_job_name,
            branch_name=branch_name,
            template_ds=template_ds,
            tmp_store=tmp_store,
            icechunk_repos=icechunk_repos,
            consolidated=self.region_job_class.consolidated_metadata,
            exclude_coord_value_chunks=exclude_coord_value_chunks,
        )

        # 2. Process jobs. Each region job variant owns its own store/session
        # lifecycle and commit cadence behind this one call.
        worker_results: dict[str, list[SourceFileResult]] = (
            self.region_job_class.process_worker_jobs(
                worker_jobs,
                self.store_factory,
                branch_name,
                worker_index,
                overwrite_chunks=overwrite_chunks
                or self.region_job_class.rewrites_whole_region,
            )
            if worker_jobs
            else {}
        )

        # 3. Write results and finalize
        if workers_total > 1:
            self.store_factory.write_coordination_file(
                reformat_job_name,
                f"results/worker-{worker_index}.json",
                parallel_coordination.dump_worker_results_json(worker_results),
            )

        if is_last:
            if update_template_with_results:
                if workers_total > 1:
                    merged_results = parallel_coordination.collect_results(
                        self.store_factory, reformat_job_name, workers_total
                    )
                else:
                    merged_results = worker_results
            else:
                parallel_coordination.wait_for_workers(
                    self.store_factory, reformat_job_name, workers_total
                )
                merged_results = {}
            parallel_coordination.finalize(
                self.store_factory,
                all_jobs=all_jobs,
                merged_results=merged_results,
                reformat_job_name=reformat_job_name,
                branch_name=branch_name,
                template_ds=template_ds,
                tmp_store=tmp_store,
                setup_info=setup_info,
                workers_total=workers_total,
                update_template_with_results=update_template_with_results,
                consolidated=self.region_job_class.consolidated_metadata,
                publish_zarr3_metadata=update_template_with_results or overwrite,
                exclude_coord_value_chunks=exclude_coord_value_chunks,
                job_uid=job_uid,
            )

    def _run_virtual_operational_update(
        self,
        all_jobs: Sequence[RegionJob[DATA_VAR, SOURCE_FILE_COORD]],
        worker_index: int,
        workers_total: int,
        *,
        job_fire_time: Timestamp | None = None,
    ) -> None:
        """One-worker virtual dataset operational update: commit one or more
        whole source files straight to the "main" icechunk branch as they arrive
        (no parallel_coordination), so readers see each file within seconds.
        See "Operational updates" in docs/virtual_datasets.md."""
        assert workers_total == 1, "Virtual operational updates use one worker per Job"
        assert worker_index == 0, "Virtual operational updates use one worker per Job"
        # A single active-window job whose generator polls the union of all
        # still-missing files; multiple jobs would run sequentially and the first
        # one's polling could consume the pod's deadline and starve the rest.
        assert len(all_jobs) == 1, (
            f"Virtual operational updates run a single active-window job, got {len(all_jobs)}"
        )
        (job,) = all_jobs
        assert isinstance(job, VirtualRegionJob)
        assert job.processing_mode == "update", (
            "operational_update_jobs must construct jobs with processing_mode='update'"
        )
        job = job.model_copy(
            update={
                "poll_deadline": self._virtual_poll_deadline(
                    job_fire_time if job_fire_time is not None else pd.Timestamp.now()
                )
            }
        )
        # Deploy checked-in template metadata fixes (attrs, coordinate values) before
        # ingesting. No-op commit-wise when the store already matches the template.
        job.refresh_metadata(self.store_factory, self._tmp_store())
        window_became_complete = job.process_virtual_worker_jobs(
            [job], self.store_factory, "main", overwrite_chunks=False
        )
        if window_became_complete and Config.is_prod:
            source_cronjob = self._operational_cron_job(ReformatCronJob).name
            if os.getenv("CRON_JOB_NAME") == source_cronjob:
                unit = job.template_ds.to_dataset().get_index(job.append_dim)[-1]
                for trigger in self.materialized_update_triggers:
                    trigger.dispatch(pd.Timestamp(unit))

    def validate_dataset(
        self,
        reformat_job_name: Annotated[str, typer.Argument(envvar="JOB_NAME")],
    ) -> None:
        """Validate the dataset, raising an exception if it is invalid.

        Runs the dataset's validators() plus the always-on structural checks:
        CheckExpectedShards on every materialized store (a virtual store holds chunk
        references rather than shards), and CheckReplicaMatchesPrimary on every
        replica.
        """
        self._validate_with_monitor(reformat_job_name)

    def _validate_with_monitor(
        self, reformat_job_name: str, *, job_fire_time: Timestamp | None = None
    ) -> None:
        with self._monitor(
            ReformatCronJob,
            reformat_job_name,
            monitor_name=self._validation_monitor_name,
            # Validate shares the same cron schedule as update, but we give it margin
            # so it's not expected to start until after the update work completes.
            checkin_margin=lambda cron: (
                math.ceil(cron.pod_active_deadline.total_seconds() / 60) + 10
            ),
        ):
            self._validate_dataset(reformat_job_name, job_fire_time=job_fire_time)

    def _validation_monitor_name(self) -> str:
        if os.getenv("CRON_JOB_NAME") in {
            staging_cronjob_name(self.dataset_id, self.template_config.version, command)
            for command in ("update", "validate")
        }:
            return staging_cronjob_name(
                self.dataset_id, self.template_config.version, "validate"
            )
        update_name = self._operational_cron_job(ReformatCronJob).name
        assert update_name.endswith("-update")
        return update_name.removesuffix("-update") + "-validate"

    def _validate_dataset(
        self, reformat_job_name: str, *, job_fire_time: Timestamp | None = None
    ) -> None:
        is_virtual = issubclass(self.region_job_class, VirtualRegionJob)
        validators = list(self.validators())
        if not is_virtual:
            validators.append(validation.CheckExpectedShards())
        region_job = self._virtual_validation_region_job(
            validators, reformat_job_name, job_fire_time=job_fire_time
        )

        primary_store = self.store_factory.primary_store()
        validation.validate_dataset(
            validators,
            store=primary_store,
            append_dim=self.template_config.append_dim,
            append_dim_frequency=self.template_config.append_dim_frequency,
            data_vars=self.template_config.data_vars,
            dataset_id=self.dataset_id,
            region_job=region_job,
        )
        log.info(f"Done validating {primary_store}")

        replica_stores = self.store_factory.replica_stores()
        if not replica_stores:
            return

        replica_validators = [
            *validators,
            validation.CheckReplicaMatchesPrimary(),
        ]
        primary_ds = validation.open_flattened_dataset(
            primary_store,
            consolidated=not isinstance(primary_store, IcechunkStore),
        )
        for replica_store in replica_stores:
            validation.validate_dataset(
                replica_validators,
                store=replica_store,
                append_dim=self.template_config.append_dim,
                append_dim_frequency=self.template_config.append_dim_frequency,
                data_vars=self.template_config.data_vars,
                dataset_id=self.dataset_id,
                region_job=region_job,
                primary_ds=primary_ds,
            )
            log.info(f"Done validating {replica_store}")

    def _operational_update_jobs(
        self,
        reformat_job_name: str,
        tmp_store: Path,
        *,
        job_fire_time: Timestamp | None = None,
        job_reference_time: Timestamp | None = None,
    ) -> tuple[Sequence[RegionJob[DATA_VAR, SOURCE_FILE_COORD]], xr.DataTree]:
        """The jobs an operational update runs, and the template they write against."""
        job_class = self.region_job_class
        if issubclass(job_class, MaterializedRegionJob):
            assert job_reference_time is not None, (
                "Materialized updates require job_reference_time"
            )
            return job_class.operational_update_jobs(
                primary_store=self.store_factory.primary_store(),
                tmp_store=tmp_store,
                get_template_fn=self._get_template,
                append_dim=self.template_config.append_dim,
                all_data_vars=self.template_config.data_vars,
                reformat_job_name=reformat_job_name,
                job_reference_time=job_reference_time,
            )
        return job_class.operational_update_jobs(
            primary_store=self.store_factory.primary_store(),
            tmp_store=tmp_store,
            get_template_fn=self._get_template,
            append_dim=self.template_config.append_dim,
            all_data_vars=self.template_config.data_vars,
            reformat_job_name=reformat_job_name,
            job_fire_time=job_fire_time
            if job_fire_time is not None
            else self._operational_cron_job(ReformatCronJob).previous_fire_time(
                pd.Timestamp.now()
            ),
        )

    def _virtual_validation_region_job(
        self,
        validators: Sequence[validation.Validator],
        reformat_job_name: str,
        *,
        job_fire_time: Timestamp | None = None,
    ) -> VirtualRegionJob[DATA_VAR, SOURCE_FILE_COORD] | None:
        """The operational-window job a manifest-probing check runs against, or None if
        none of the validators need it. Built once and shared across primary + replica
        (the job is store-independent; validate_dataset passes each check the store)."""
        if not any(v.requires_virtual_dataset for v in validators):
            return None
        jobs, _template_ds = self._operational_update_jobs(
            reformat_job_name, self._tmp_store(), job_fire_time=job_fire_time
        )
        job = item(jobs)
        assert isinstance(job, VirtualRegionJob), (
            f"validators() includes a check that requires a virtual dataset but "
            f"{self.region_job_class.__name__} is not a VirtualRegionJob"
        )
        return job

    def dataset_urls(
        self,
        output_format: Annotated[
            Literal["text", "json"],
            typer.Option("--format", help="Output format"),
        ] = "text",
    ) -> None:
        """Print the canonical production URLs for this dataset's primary and replica stores."""
        primary = self.store_factory.primary_url()
        replicas = self.store_factory.replica_urls()

        match output_format:
            case "json":
                typer.echo(
                    json.dumps({"primary": primary, "replicas": replicas}, indent=2)
                )
            case "text":
                typer.echo("Primary:")
                typer.echo(primary)
                typer.echo("")
                typer.echo("Replicas:")
                if replicas:
                    for url in replicas:
                        typer.echo(url)
                else:
                    typer.echo("(none)")

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
        app.command()(self.dataset_urls)
        # Avoid method name conflict with pydantic's validate while keeping cli commands consistent
        app.command("validate")(self.validate_dataset)
        return app

    def _tmp_store(self) -> Path:
        return get_local_tmp_store()

    def _get_template(self, append_dim_end: DatetimeLike) -> xr.DataTree:
        return self.template_config.get_template(pd.Timestamp(append_dim_end))

    def _open_existing_store(self) -> xr.DataTree | None:
        """The primary store's current contents, or None if no dataset exists there yet.

        Only store-missing errors mean None; anything else (auth, network) raises —
        misclassifying an existing store as absent would route a backfill past the
        overwrite guards."""
        try:
            return self.store_factory.open_primary_datatree()
        except (
            FileNotFoundError,  # zarr3 store path absent
            zarr.errors.GroupNotFoundError,  # store path exists but holds no dataset
            icechunk.NotFoundError,  # icechunk repository not created yet
        ):
            return None

    def _resolve_backfill_template(
        self,
        existing_ds: xr.DataTree | None,
        append_dim_end: datetime | None,
        *,
        overwrite_chunks: bool,
        overwrite_metadata: bool,
    ) -> tuple[xr.DataTree, pd.Timestamp]:
        """Resolve a backfill's template and append_dim_end and validate it is safe
        to write. The end defaults to now for a new store and to the store's current
        end for an existing store. A store is only created when no overwrite flag is
        passed, only written into with one, never trimmed, and only extended by an
        explicit append_dim_end with both overwrite flags."""
        overwrite = overwrite_chunks or overwrite_metadata
        append_dim = self.template_config.append_dim

        if existing_ds is None:
            assert not overwrite, (
                f"No existing store found for {self.dataset_id}. Remove "
                "--overwrite-chunks/--overwrite-metadata to create a new store."
            )
            resolved_end = (
                pd.Timestamp(append_dim_end)
                if append_dim_end is not None
                # Floored so the isoformat in the worker command parses as a CLI datetime
                else pd.Timestamp.now(tz="UTC").tz_localize(None).floor("s")
            )
            return self._get_template(resolved_end), resolved_end

        assert overwrite, (
            f"A store already exists for {self.dataset_id}. Pass --overwrite-chunks "
            "to rewrite chunk data and/or --overwrite-metadata to update metadata "
            "from the template. Creating a store from scratch requires deleting the "
            "existing store first."
        )
        if append_dim_end is not None:
            resolved_end = pd.Timestamp(append_dim_end)
        else:
            resolved_end = self.template_config.append_dim_frequency + max(
                pd.Timestamp(node[append_dim].max().item())
                for node in existing_ds.subtree
                if append_dim in node.dims
            )
        template_ds = self._get_template(resolved_end)
        template_utils.assert_safe_overwrite(
            template_ds,
            existing_ds,
            append_dim,
            allow_new_arrays=overwrite_metadata,
            allow_expansion=overwrite_chunks and overwrite_metadata,
        )
        return template_ds, resolved_end

    def _assert_no_structural_drift(self, template_ds: xr.DataTree) -> None:
        template_utils.assert_no_structural_drift_from_existing_store(
            template_ds,
            self.store_factory.open_primary_datatree(),
            self.template_config.append_dim,
        )

    def _can_run_in_kubernetes(self) -> bool:
        # This is a method to support testing without changing the Config.env
        return Config.is_prod

    def _virtual_poll_deadline(self, now: Timestamp) -> Timestamp:
        """Polling cutoff of the latest scheduled fire, reserving time for validation."""
        cron_job = self._operational_cron_job(ReformatCronJob)
        assert (
            timedelta(0)
            < self.virtual_poll_deadline_grace
            < cron_job.pod_active_deadline
        )
        return (
            cron_job.previous_fire_time(now)
            + cron_job.pod_active_deadline
            - self.virtual_poll_deadline_grace
        )

    @model_validator(mode="after")
    def _validate_virtual_storage(self) -> Self:
        # A virtual region job emits chunk refs into icechunk and needs the source
        # containers registered, so the virtual config is mandatory for it.
        if (
            issubclass(self.region_job_class, VirtualRegionJob)
            and self.icechunk_virtual_config is None
        ):
            raise ValueError(
                f"{self.region_job_class.__name__} is a VirtualRegionJob but no "
                "icechunk_virtual_config was provided; virtual datasets require it."
            )
        # The default virtual operational_update_jobs reads this classvar; catch a
        # missing declaration at dataset construction, not at the first cron fire.
        if (
            issubclass(self.region_job_class, VirtualRegionJob)
            and self.region_job_class.operational_update_jobs.__func__
            is VirtualRegionJob.operational_update_jobs.__func__
            and not hasattr(self.region_job_class, "operational_update_window")
        ):
            raise ValueError(
                f"{self.region_job_class.__name__} uses the default virtual "
                "operational_update_jobs but does not set operational_update_window."
            )
        # Virtual datasets store icechunk metadata + virtual chunk refs, so every store must be icechunk.
        if self.icechunk_virtual_config is not None:
            non_icechunk = [
                config
                for config in (
                    self.primary_storage_config,
                    *self.replica_storage_configs,
                )
                if config.format != DatasetFormat.ICECHUNK
            ]
            if non_icechunk:
                raise ValueError(
                    "icechunk_virtual_config requires every storage config to use "
                    f"the ICECHUNK format, but found: {non_icechunk}"
                )
        # A virtual chunk is exactly one source object or message: no shards (get_jobs
        # would partition by them and zarr would shard source bytes), and an explicit
        # compressor pipeline. None would serialize away and silently select Zarr's
        # default compressor rather than matching the referenced bytes.
        if issubclass(self.region_job_class, VirtualRegionJob):
            for var in self.template_config.data_vars:
                assert var.encoding.shards is None, (
                    f"virtual data var {var.name} must not declare shards"
                )
                assert var.encoding.compressors is not None, (
                    f"virtual data var {var.name} must explicitly declare compressors"
                )
        return self
