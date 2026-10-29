from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import computed_field

from reformatters.common import validation
from reformatters.common.kubernetes import CronJob
from reformatters.common.pydantic import replace
from reformatters.common.region_job import RegionJob
from reformatters.common.storage import StoreFactory
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day025DegreeDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.source_file_coord import (
    IfsEnsSourceFileCoord,
)

from .region_job import EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3RegionJob
from .template_config import (
    EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3TemplateConfig,
)


class EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3Dataset(
    EcmwfIfsEnsForecast15Day025DegreeDataset
):
    """The plain Zarr v3 store of this dataset, processed independently of the
    Icechunk store so it can keep its own append_dim_start."""

    template_config: EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3TemplateConfig = (
        EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3TemplateConfig()
    )
    region_job_class: type[
        EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3RegionJob
    ] = EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3RegionJob

    @computed_field
    @property
    def dataset_id(self) -> str:
        return "ecmwf-ifs-ens-15-day-deprecated-zarr-v3"

    @computed_field
    @property
    def store_factory(self) -> StoreFactory:
        # The store keeps the published dataset_id; dataset_id above only names the
        # CLI command and kubernetes resources.
        return StoreFactory(
            primary_storage_config=self.primary_storage_config,
            replica_storage_configs=self.replica_storage_configs,
            dataset_id=self.template_config.dataset_id,
            template_config_version=self.template_config.version,
            icechunk_virtual_config=self.icechunk_virtual_config,
        )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        (update_cron_job,) = super().operational_kubernetes_resources(image_tag)
        return [
            replace(
                update_cron_job,
                # f360 lands on GCS ~08:20-08:35 UTC
                schedule="45 8 * * *",
                concurrency_policy="Replace",
                pod_active_deadline=timedelta(minutes=45),
                active_deadline=None,
            )
        ]

    def validators(self) -> Sequence[validation.Validator]:
        return (
            validation.CheckCurrentData(max_delay=timedelta(hours=9, minutes=20)),
            validation.CheckRecentNans(append_dim_window=3),
        )

    def _resolve_backfill_template(
        self,
        existing_ds: xr.DataTree | None,
        append_dim_end: datetime | None,
        *,
        overwrite_chunks: bool,
        overwrite_metadata: bool,
    ) -> tuple[xr.DataTree, pd.Timestamp]:
        template_ds, resolved_end = super()._resolve_backfill_template(
            existing_ds,
            append_dim_end,
            overwrite_chunks=overwrite_chunks,
            overwrite_metadata=overwrite_metadata,
        )
        if existing_ds is not None:
            self._assert_stored_labels_match_template(existing_ds, template_ds)
        return template_ds, resolved_end

    def _process_region_jobs(
        self,
        all_jobs: Sequence[RegionJob[EcmwfDataVar, IfsEnsSourceFileCoord]],
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
        if (existing_ds := self._open_existing_store()) is not None:
            self._assert_stored_labels_match_template(existing_ds, template_ds)
        super()._process_region_jobs(
            all_jobs,
            worker_index,
            workers_total,
            reformat_job_name,
            template_ds,
            tmp_store,
            update_template_with_results=update_template_with_results,
            overwrite_chunks=overwrite_chunks,
            overwrite_metadata=overwrite_metadata,
            job_uid=job_uid,
        )

    def _assert_stored_labels_match_template(
        self, existing_ds: xr.DataTree, template_ds: xr.DataTree
    ) -> None:
        append_dim = self.template_config.append_dim
        stored: np.ndarray[Any, Any] = existing_ds.dataset[append_dim].values
        expected = template_ds.dataset[append_dim].values[: stored.size]
        assert np.array_equal(stored, expected), (
            f"Stored {append_dim} labels do not match the template; refusing to write"
        )
