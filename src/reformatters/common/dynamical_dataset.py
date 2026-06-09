import json
import os
import subprocess
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, Self, TypeVar

import numpy as np
import pandas as pd
import sentry_sdk
import sentry_sdk.crons
import typer
import xarray as xr
from icechunk.store import IcechunkStore
from pydantic import Field, computed_field, model_validator

from reformatters.common import (
    parallel_coordination,
    template_utils,
    validation,
)
from reformatters.common.config import Config
from reformatters.common.config_models import DataVar
from reformatters.common.iterating import digest, get_worker_jobs, item
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

    @model_validator(mode="after")
    def _validate_virtual_storage(self) -> Self:
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
        return self

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

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
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

        # In an attempt to keep the subclassing API simpler, we are keeping
        # all resource needs defined right in `operational_kubernetes_resources`.
        # If for some reason there are _multiple_ ReformatCronJobs returned from
        # that we'll need to revisit the logic below or this approach.
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

        template_ds = self._get_template(append_dim_end)

        if overwrite_existing:
            assert self.store_factory.all_stores_exist(), (
                "Not all stores exist, cannot run with overwrite_existing=True"
            )
            log.info("Writing to existing stores, skipping metadata write.")
        else:
            # Write metadata to final store. Required for Zarr v3 only, Icechunk metadata is written in parallel_setup.
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
        worker_jobs = get_worker_jobs(all_jobs, worker_index, workers_total)

        jobs_summary = ", ".join(repr(j) for j in worker_jobs)
        log.info(
            f"This is {worker_index = }, {workers_total = }, "
            f"{len(worker_jobs)} of {len(all_jobs)} total jobs, {jobs_summary}"
        )

        icechunk_repos = self.store_factory.icechunk_repos(sort="primary-first")
        has_icechunk = len(icechunk_repos) > 0
        branch_name = f"_job_{reformat_job_name}" if has_icechunk else "main"

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
            )

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
                    partial(  # ty: ignore[invalid-argument-type]
                        validation.compare_replica_and_primary,
                        self.template_config.append_dim,
                        xr.open_zarr(
                            replica_store,
                            chunks=None,
                            consolidated=not isinstance(replica_store, IcechunkStore),
                        ),
                    )
                )

                validation.validate_dataset(
                    replica_store,
                    validators=replica_store_validators,
                )
                log.info(f"Done validating {replica_store}")

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
        *,
        send_in_progress: bool = True,
        send_result: bool = True,
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

        if send_in_progress:
            capture_checkin("in_progress")
        try:
            yield
        except Exception:
            if send_result:
                capture_checkin("error")
            raise
        else:
            if send_result:
                capture_checkin("ok")
