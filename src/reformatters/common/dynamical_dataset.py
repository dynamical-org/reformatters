from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

import pandas as pd
import typer
import xarray as xr
import zarr
from pydantic import computed_field

from reformatters.common import template_utils
from reformatters.common.config_models import DataVar
from reformatters.common.logging import get_logger
from reformatters.common.pydantic import FrozenBaseModel
from reformatters.common.region_job import RegionJob, SourceFileCoord
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import DatetimeLike
from reformatters.common.zarr import (
    copy_zarr_metadata,
    get_local_tmp_store,
    get_mode,
    get_zarr_store,
)

DATA_VAR = TypeVar("DATA_VAR", bound=DataVar[Any])
SOURCE_FILE_COORD = TypeVar("SOURCE_FILE_COORD", bound=SourceFileCoord)

logger = get_logger(__name__)


class DynamicalDataset(FrozenBaseModel, Generic[DATA_VAR, SOURCE_FILE_COORD]):
    """Top level class managing a dataset configuration and processing."""

    template_config: TemplateConfig[DATA_VAR]
    region_job_class: type[RegionJob[DATA_VAR, SOURCE_FILE_COORD]]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_id(self) -> str:
        return self.template_config.dataset_id

    def update_template(self) -> None:
        """Generate and persist the dataset template using the template_config."""
        self.template_config.update_template()

    def reformat_kubernetes(self) -> None:
        """Run dataset reformatting using Kubernetes index jobs."""
        pass

    def reformat_operational_update(self) -> None:
        """Run operational update of the dataset locally."""
        end = self.region_job_class.operational_update_append_dim_end()
        template_ds = self._template_ds(end)
        final_store = self._final_store()
        tmp_store = self._tmp_store()
        # Write initial metadata to tmp_store
        template_utils.write_metadata(template_ds, tmp_store, get_mode(tmp_store))
        jobs = self.region_job_class.operational_update_jobs(
            final_store=final_store,
            tmp_store=tmp_store,
            template_ds=template_ds,
            append_dim=self.template_config.append_dim,
            all_data_vars=self.template_config.data_vars,
            kubernetes_job_name="operational-update",
        )
        for job in jobs:
            process_results = job.process()
            updated_template = job.update_template_with_results(process_results)
            template_utils.write_metadata(
                updated_template, tmp_store, get_mode(tmp_store)
            )
            copy_zarr_metadata(updated_template, tmp_store, final_store)
        logger.info(f"Done operational update writing to {final_store}")

    def reformat_local(
        self,
        append_dim_end: datetime,
        *,
        filter_start: datetime | None = None,
        filter_end: datetime | None = None,
        filter_variable_names: list[str] | None = None,
    ) -> None:
        """Run dataset reformatting locally in this process."""
        template_ds = self._template_ds(append_dim_end)
        final_store = self._final_store()

        template_utils.write_metadata(template_ds, final_store, get_mode(final_store))

        self.process_region_jobs(
            append_dim_end,
            kubernetes_job_name="local",
            worker_index=0,
            workers_total=1,
            filter_start=filter_start,
            filter_end=filter_end,
            filter_variable_names=filter_variable_names,
        )
        logger.info(f"Done writing to {final_store}")

    def process_region_jobs(
        self,
        append_dim_end: DatetimeLike,
        kubernetes_job_name: str,
        *,
        worker_index: int,
        workers_total: int,
        filter_start: DatetimeLike | None = None,
        filter_end: DatetimeLike | None = None,
        filter_variable_names: list[str] | None = None,
    ) -> None:
        """Orchestrate running RegionJob instances."""

        region_jobs = self.region_job_class.get_jobs(
            kind="backfill",
            final_store=self._final_store(),
            tmp_store=self._tmp_store(),
            template_ds=self._template_ds(append_dim_end),
            append_dim=self.template_config.append_dim,
            all_data_vars=self.template_config.data_vars,
            kubernetes_job_name=kubernetes_job_name,
            worker_index=worker_index,
            workers_total=workers_total,
            filter_start=pd.Timestamp(filter_start) if filter_start else None,
            filter_end=pd.Timestamp(filter_end) if filter_end else None,
            filter_variable_names=filter_variable_names,
        )

        jobs_summary = ", ".join(j.summary() for j in region_jobs)
        logger.info(
            f"This is {worker_index = }, {workers_total = }, {len(region_jobs)} jobs, {jobs_summary}"
        )
        for region_job in region_jobs:
            region_job.process()

    def get_cli(
        self,
    ) -> typer.Typer:
        """Create a CLI app with dataset commands"""
        app = typer.Typer()
        app.command()(self.update_template)
        app.command()(self.reformat_local)
        app.command()(self.reformat_kubernetes)
        return app

    def _final_store(self) -> zarr.abc.store.Store:
        return get_zarr_store(
            self.template_config.dataset_id, self.template_config.version
        )

    def _tmp_store(self) -> Path:
        return get_local_tmp_store()

    def _template_ds(self, append_dim_end: DatetimeLike) -> xr.Dataset:
        return self.template_config.get_template(pd.Timestamp(append_dim_end))
