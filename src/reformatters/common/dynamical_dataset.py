import json
import subprocess
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, Protocol, Self, TypeVar

import icechunk
import numpy as np
import pandas as pd
import typer
import xarray as xr
import zarr.errors
from icechunk.store import IcechunkStore
from pydantic import Field, computed_field, model_validator

from reformatters.common import (
    parallel_coordination,
    template_utils,
    validation,
)
from reformatters.common.config import Config
from reformatters.common.config_models import ROOT, DataVar
from reformatters.common.iterating import get_worker_jobs, item
from reformatters.common.kubernetes import (
    CronJob,
    Job,
    ReformatCronJob,
    ValidationCronJob,
    get_deployed_cronjob_image,
)
from reformatters.common.logging import get_logger
from reformatters.common.pydantic import FrozenBaseModel
from reformatters.common.region_job import (
    RegionJob,
    SourceFileCoord,
    SourceFileResult,
)
from reformatters.common.storage import (
    DatasetFormat,
    IcechunkVirtualConfig,
    StorageConfig,
    StoreFactory,
    get_local_tmp_store,
)
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import DatetimeLike
from reformatters.common.virtual_region_job import VirtualRegionJob

DATA_VAR = TypeVar("DATA_VAR", bound=DataVar[Any])
SOURCE_FILE_COORD = TypeVar("SOURCE_FILE_COORD", bound=SourceFileCoord)

log = get_logger(__name__)


class DynamicalDataset(FrozenBaseModel, Generic[DATA_VAR, SOURCE_FILE_COORD]):
    """Top level class managing a dataset configuration and processing."""

    template_config: TemplateConfig[DATA_VAR]
    region_job_class: type[RegionJob[DATA_VAR, SOURCE_FILE_COORD]]

    primary_storage_config: StorageConfig
    replica_storage_configs: Sequence[StorageConfig] = Field(default_factory=tuple)
    icechunk_virtual_config: IcechunkVirtualConfig | None = None

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

        Implementations should look similar to this:
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
        with self._monitor(
            ReformatCronJob,
            reformat_job_name,
            send_in_progress=is_first,
            send_result=is_last,
        ):
            tmp_store = self._tmp_store()

            all_jobs, template_ds = self.region_job_class.operational_update_jobs(
                primary_store=self.store_factory.primary_store(),
                tmp_store=tmp_store,
                get_template_fn=self._get_template,
                append_dim=self.template_config.append_dim,
                all_data_vars=self.template_config.data_vars,
                reformat_job_name=reformat_job_name,
            )

            if issubclass(self.region_job_class, VirtualRegionJob):
                # Virtual operational updates are single-writer streaming (see docs/parallel_processing.md)
                if is_first:
                    self._assert_no_structural_drift(template_ds)
                self._run_virtual_operational_update(
                    all_jobs, worker_index, workers_total
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
                )

        log.info(
            f"Operational update complete. Wrote to primary store {self.store_factory.primary_store()} and replicas {self.store_factory.replica_stores()}"
        )

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
                self._open_primary_datatree(),
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
                worker_jobs, self.store_factory, branch_name, worker_index
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
            )

    def _run_virtual_operational_update(
        self,
        all_jobs: Sequence[RegionJob[DATA_VAR, SOURCE_FILE_COORD]],
        worker_index: int,
        workers_total: int,
    ) -> None:
        """Single-writer virtual dataset operational update: commit one or more
        whole source files straight to the "main" icechunk branch as they arrive
        (no parallel_coordination), so readers see each file within seconds.
        See "Operational updates" in docs/virtual_datasets.md."""
        assert workers_total == 1, "Virtual operational updates run single-writer"
        assert worker_index == 0, "Virtual operational updates run single-writer"
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
        # Deploy checked-in template metadata fixes (attrs, coordinate values) before
        # ingesting. No-op commit-wise when the store already matches the template.
        job.refresh_metadata(self.store_factory, self._tmp_store())
        self.region_job_class.process_worker_jobs(
            all_jobs, self.store_factory, "main", worker_index
        )

    def validate_dataset(
        self,
        reformat_job_name: Annotated[str, typer.Argument(envvar="JOB_NAME")],
    ) -> None:
        """Validate the dataset, raising an exception if it is invalid."""
        # validators() lists both validator kinds; validate_dataset dispatches by type.
        # check_for_expected_shards / compare_replica_and_primary are materialized-only
        # and appended below.
        is_virtual = issubclass(self.region_job_class, VirtualRegionJob)
        base_validators = list(self.validators())
        with self._monitor(ValidationCronJob, reformat_job_name):
            region_job = self._virtual_validation_region_job(
                base_validators, reformat_job_name
            )

            primary_store = self.store_factory.primary_store()
            primary_store_validators = list(base_validators)
            if not is_virtual:
                primary_store_validators.append(
                    partial(validation.check_for_expected_shards, primary_store)
                )

            validation.validate_dataset(
                primary_store,
                validators=primary_store_validators,
                region_job=region_job,
            )
            log.info(f"Done validating {primary_store}")

            for replica_store in self.store_factory.replica_stores():
                replica_store_validators = list(base_validators)
                if not is_virtual:
                    replica_store_validators.append(
                        partial(validation.check_for_expected_shards, replica_store)
                    )
                    replica_store_validators.append(
                        partial(  # ty: ignore[invalid-argument-type]
                            validation.compare_replica_and_primary,
                            self.template_config.append_dim,
                            validation.open_flattened_dataset(
                                replica_store,
                                consolidated=not isinstance(
                                    replica_store, IcechunkStore
                                ),
                            ),
                        )
                    )

                validation.validate_dataset(
                    replica_store,
                    validators=replica_store_validators,
                    region_job=region_job,
                )
                log.info(f"Done validating {replica_store}")

    def _virtual_validation_region_job(
        self,
        validators: Sequence[validation.DataValidator],
        reformat_job_name: str,
    ) -> VirtualRegionJob[DATA_VAR, SOURCE_FILE_COORD] | None:
        """The operational-window job a VirtualDataValidator probes against, or None if
        none of the validators need it. Built once and shared across primary + replica
        (the job is store-independent; validate_dataset passes each validator the store)."""
        if not any(isinstance(v, validation.VirtualDataValidator) for v in validators):
            return None
        jobs, _template_ds = self.region_job_class.operational_update_jobs(
            primary_store=self.store_factory.primary_store(),
            tmp_store=self._tmp_store(),
            get_template_fn=self._get_template,
            append_dim=self.template_config.append_dim,
            all_data_vars=self.template_config.data_vars,
            reformat_job_name=reformat_job_name,
        )
        job = item(jobs)
        assert isinstance(job, VirtualRegionJob), (
            f"validators() returned a VirtualDataValidator but {self.region_job_class.__name__} "
            "is not a VirtualRegionJob"
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

    def _open_primary_datatree(self) -> xr.DataTree:
        return xr.open_datatree(
            self.store_factory.primary_store(),  # ty: ignore[invalid-argument-type]
            engine="zarr",
            decode_timedelta=True,
            chunks=None,
        )

    def _open_existing_store(self) -> xr.DataTree | None:
        """The primary store's current contents, or None if no dataset exists there yet.

        Only store-missing errors mean None; anything else (auth, network) raises —
        misclassifying an existing store as absent would route a backfill past the
        overwrite guards."""
        try:
            return self._open_primary_datatree()
        except (
            FileNotFoundError,  # zarr3 store path absent
            zarr.errors.GroupNotFoundError,  # store path exists but holds no dataset
            icechunk.IcechunkError,  # icechunk repository not created yet
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
            template_ds, self._open_primary_datatree(), self.template_config.append_dim
        )

    def _can_run_in_kubernetes(self) -> bool:
        # This is a method to support testing without changing the Config.env
        return Config.is_prod

    @contextmanager
    def _monitor(
        self,
        cron_type: type[CronJob],
        reformat_job_name: str,
        cron_job_name: str | None = None,
        *,
        send_in_progress: bool = True,
        send_result: bool = True,
    ) -> Iterator[None]:
        # No registered monitors -> nothing to report to, and no need to require
        # operational_kubernetes_resources to be defined.
        if not _RUN_MONITORS:
            yield
            return

        # Find the cron job that matches the type (and name if provided). There should be exactly one.
        cron_jobs = self.operational_kubernetes_resources("placeholder-image-tag")
        if cron_job_name:
            cron_jobs = (c for c in cron_jobs if c.name == cron_job_name)
        cron_jobs = (c for c in cron_jobs if isinstance(c, cron_type))
        cron_job = item(cron_jobs)

        with ExitStack() as stack:
            for monitor in _RUN_MONITORS:
                stack.enter_context(
                    monitor(
                        cron_job,
                        reformat_job_name,
                        send_in_progress=send_in_progress,
                        send_result=send_result,
                    )
                )
            yield

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
        # A virtual chunk is exactly one source message: no shards (get_jobs would
        # partition by them and zarr would shard raw source bytes) and no
        # compressors (a None would serialize away and zarr would stack its
        # default compressor on the raw source bytes).
        if issubclass(self.region_job_class, VirtualRegionJob):
            for var in self.template_config.data_vars:
                assert var.encoding.shards is None, (
                    f"virtual data var {var.name} must not declare shards"
                )
                assert var.encoding.compressors == (), (
                    f"virtual data var {var.name} must declare compressors=()"
                )
        else:
            # The materialized chunk-write path (zarr.copy_data_var, write_shards) is
            # not yet group-aware; only virtual datasets support vertical groups today.
            grouped = [
                v.path for v in self.template_config.data_vars if v.group is not ROOT
            ]
            assert not grouped, (
                f"materialized datasets do not yet support vertical groups: {grouped}"
            )
        return self


class RunMonitor(Protocol):
    """Wraps a single operational cron run to report it to a monitoring service.

    The application registers monitors (see `register_run_monitor`); `DynamicalDataset._monitor`
    enters every registered one around each update/validate run. This keeps
    DynamicalDataset agnostic of any specific monitoring service (Better Stack,
    Sentry, ...) — a different deployment registers whatever it uses, or nothing.
    """

    def __call__(
        self,
        cron_job: CronJob,
        reformat_job_name: str,
        *,
        send_in_progress: bool,
        send_result: bool,
    ) -> AbstractContextManager[None]: ...


_RUN_MONITORS: list[RunMonitor] = []


def register_run_monitor(monitor: RunMonitor) -> None:
    """Register a monitor to wrap every operational cron run. With none registered,
    monitoring is a no-op."""
    _RUN_MONITORS.append(monitor)
