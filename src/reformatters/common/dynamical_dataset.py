import json
import subprocess
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeVar

import numpy as np
import pandas as pd
import sentry_sdk
import typer
import xarray as xr
from pydantic import Field, computed_field

from reformatters.common import docker, storage, template_utils, validation
from reformatters.common.config import Config
from reformatters.common.config_models import DataVar
from reformatters.common.iterating import digest, item
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
from reformatters.common.update_progress_tracker import UpdateProgressTracker
from reformatters.common.zarr import copy_zarr_metadata

DATA_VAR = TypeVar("DATA_VAR", bound=DataVar[Any])
SOURCE_FILE_COORD = TypeVar("SOURCE_FILE_COORD", bound=SourceFileCoord)

log = get_logger(__name__)


class DynamicalDataset(FrozenBaseModel, Generic[DATA_VAR, SOURCE_FILE_COORD]):
    """Top level class managing a dataset configuration and processing."""

    template_config: TemplateConfig[DATA_VAR]
    region_job_class: type[RegionJob[DATA_VAR, SOURCE_FILE_COORD]]

    primary_storage_config: StorageConfig
    replica_storage_configs: Sequence[StorageConfig] = Field(default_factory=tuple)

    use_progress_tracker: bool = False

    @computed_field  # type: ignore[prop-decorator]
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
            name=f"{self.dataset_id}-operational-update",
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
            name=f"{self.dataset_id}-validation",
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_id(self) -> str:
        return self.template_config.dataset_id

    def update_template(self) -> None:
        """Generate and persist the dataset template using the template_config."""
        self.template_config.update_template()

    def update(
        self,
        reformat_job_name: Annotated[str, typer.Argument(envvar="JOB_NAME")],
    ) -> None:
        """Update an existing dataset with the latest data."""
        with self._monitor(ReformatCronJob, reformat_job_name):
            tmp_store = self._tmp_store()

            jobs, template_ds = self.region_job_class.operational_update_jobs(
                primary_store=self.store_factory.primary_store(),
                tmp_store=tmp_store,
                get_template_fn=self._get_template,
                append_dim=self.template_config.append_dim,
                all_data_vars=self.template_config.data_vars,
                reformat_job_name=reformat_job_name,
            )

            for job in jobs:
                assert template_ds.equals(job.template_ds), (
                    "template_ds doesn't match job.template_ds"
                )

                # New stores to ensure that, if any are Icechunk stores, we have
                # an uncomitted Icechunk session for each job.
                primary_store = self.store_factory.primary_store()
                replica_stores = self.store_factory.replica_stores()

                # This will expand the tmp store dimensions. We do this for each job
                # because the tmp store will also be potentially trimmed when we
                # call update_template_with_results on the job, and each job needs to
                # start off with the fully expanded dimensions.
                template_utils.write_metadata(job.template_ds, job.tmp_store)

                # Icechunk stores metadata needs to be updated to
                # expand the dataset dimensions before we write the actual data
                copy_zarr_metadata(
                    job.template_ds,
                    job.tmp_store,
                    primary_store,
                    replica_stores=replica_stores,
                    icechunk_only=True,
                )

                progress_tracker = None
                if self.use_progress_tracker:
                    progress_tracker = UpdateProgressTracker(
                        reformat_job_name,
                        job.region.start,
                        self.store_factory,
                    )

                process_results = job.process(
                    primary_store=primary_store,
                    replica_stores=replica_stores,
                    progress_tracker=progress_tracker,
                )
                updated_template = job.update_template_with_results(process_results)
                # overwrite the tmp store metadata with updated template
                template_utils.write_metadata(updated_template, tmp_store)
                copy_zarr_metadata(
                    updated_template,
                    tmp_store,
                    primary_store,
                    replica_stores=replica_stores,
                )

                storage.commit_if_icechunk(
                    f"Automated update at {pd.Timestamp.now(tz='UTC').isoformat()}",
                    primary_store,
                    replica_stores,
                )

                if progress_tracker is not None:
                    progress_tracker.close()

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
            log.info("Writing to existing stores, skipping metadata write.")
            assert self.store_factory.all_stores_exist(), (
                "Not all stores exist, cannot run with overwrite_existing=True"
            )
        else:
            template_utils.write_metadata(template_ds, self.store_factory)

        num_jobs = len(
            self.region_job_class.get_jobs(
                kind="backfill",
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
            "process-backfill-region-jobs",
            pd.Timestamp(append_dim_end).isoformat(),
        ]
        if filter_start is not None:
            command.append(f"--filter-start={filter_start.isoformat()}")
        if filter_end is not None:
            command.append(f"--filter-end={filter_end.isoformat()}")
        if filter_contains is not None:
            for timestamp in filter_contains:
                command.append(f"--filter-contains={timestamp.isoformat()}")
        if filter_variable_names is not None:
            for variable_name in filter_variable_names:
                command.append(f"--filter-variable-names={variable_name}")

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
                }
            ),
            secret_names=self.store_factory.k8s_secret_names(),
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

        self.process_backfill_region_jobs(
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

    def process_backfill_region_jobs(
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

        region_jobs = self.region_job_class.get_jobs(
            kind="backfill",
            tmp_store=self._tmp_store(),
            template_ds=self._get_template(append_dim_end),
            append_dim=self.template_config.append_dim,
            all_data_vars=self.template_config.data_vars,
            reformat_job_name=reformat_job_name,
            worker_index=worker_index,
            workers_total=workers_total,
            filter_start=pd.Timestamp(filter_start) if filter_start else None,
            filter_end=pd.Timestamp(filter_end) if filter_end else None,
            filter_contains=(
                [pd.Timestamp(t) for t in filter_contains] if filter_contains else None
            ),
            filter_variable_names=filter_variable_names,
        )

        jobs_summary = ", ".join(repr(j) for j in region_jobs)
        log.info(
            f"This is {worker_index = }, {workers_total = }, {len(region_jobs)} jobs, {jobs_summary}"
        )
        for region_job in region_jobs:
            # New stores to ensure that, if any are Icechunk stores, we have
            # an uncomitted Icechunk session for each job.
            primary_store = self.store_factory.primary_store()
            replica_stores = self.store_factory.replica_stores()

            template_utils.write_metadata(region_job.template_ds, region_job.tmp_store)

            progress_tracker = None
            if self.use_progress_tracker:
                progress_tracker = UpdateProgressTracker(
                    reformat_job_name,
                    region_job.region.start,
                    self.store_factory,
                )

            region_job.process(
                primary_store, replica_stores, progress_tracker=progress_tracker
            )

            storage.commit_if_icechunk(
                f"Backfill completed at {pd.Timestamp.now(tz='UTC').isoformat()}",
                primary_store,
                replica_stores,
            )

            if progress_tracker is not None:
                progress_tracker.close()

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
        app.command()(self.process_backfill_region_jobs)
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
    ) -> Iterator[None]:
        # Don't require operational_kubernetes_resources to be defined unless sentry reporting is enabled
        if not Config.is_sentry_enabled:
            yield
            return

        cron_jobs = self.operational_kubernetes_resources("placeholder-image-tag")
        cron_job = item(c for c in cron_jobs if isinstance(c, cron_type))

        def capture_checkin(status: Literal["ok", "in_progress", "error"]) -> None:
            sentry_sdk.crons.capture_checkin(
                monitor_slug=cron_job.name,
                check_in_id=digest(reformat_job_name, length=32),
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
