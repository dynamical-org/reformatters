from typing import Any, Generic, TypeVar

import pandas as pd
import pydantic

from reformatters.common.config_models import DataVar
from reformatters.common.logging import get_logger
from reformatters.common.region_job import RegionJob, SourceFileCoord
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import DatetimeLike
from reformatters.common.zarr import get_zarr_store

DATA_VAR = TypeVar("DATA_VAR", bound=DataVar[Any])
SOURCE_FILE_COORD = TypeVar("SOURCE_FILE_COORD", bound=SourceFileCoord)

logger = get_logger(__name__)


class DynamicalDataset(pydantic.BaseModel, Generic[DATA_VAR, SOURCE_FILE_COORD]):
    """Top level class managing a dataset configuration and processing."""

    template_config: TemplateConfig[DATA_VAR]
    region_job_class: type[RegionJob[DATA_VAR, SOURCE_FILE_COORD]]

    def update_template(self) -> None:
        """Generate and persist the dataset template using the template_config."""
        self.template_config.update_template()

    def reformat_kubernetes(self) -> None:
        """Run dataset reformatting using Kubernetes index jobs."""
        pass

    def reformat_local(self) -> None:
        """Run dataset reformatting locally in this process."""
        pass

    def process_region_jobs(
        self,
        append_dim_end: DatetimeLike,
        *,
        worker_index: int,
        workers_total: int,
        filter_start: DatetimeLike | None = None,
        filter_end: DatetimeLike | None = None,
        filter_variable_names: list[str] | None = None,
    ) -> None:
        """Orchestrate running RegionJob instances."""

        store = get_zarr_store(
            self.template_config.dataset_id, self.template_config.version
        )

        region_jobs = self.region_job_class.get_backfill_jobs(
            store=store,
            template_ds=self.template_config.get_template(pd.Timestamp(append_dim_end)),
            append_dim=self.template_config.append_dim,
            all_data_vars=self.template_config.data_vars,
            worker_index=worker_index,
            workers_total=workers_total,
            filter_start=pd.Timestamp(filter_start) if filter_start else None,
            filter_end=pd.Timestamp(filter_end) if filter_end else None,
            filter_variable_names=filter_variable_names,
        )

        jobs_summary = ", ".join(j.summary() for j in region_jobs)
        logger.info(f"This is {worker_index = }, {workers_total = }, {jobs_summary}")
        for region_job in region_jobs:
            region_job.process()
