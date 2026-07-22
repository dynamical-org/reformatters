from collections.abc import Sequence
from datetime import timedelta  # noqa: F401
from functools import partial  # noqa: F401

import icechunk  # noqa: F401
from pydantic import Field  # noqa: F401

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import (  # noqa: F401
    CronJob,
    ReformatCronJob,
    ValidationCronJob,
)
from reformatters.common.storage import (  # noqa: F401
    IcechunkVirtualConfig,
    manifest_append_dim_split,
)

from .region_job import ExampleSpatialRegionJob, ExampleSpatialSourceFileCoord

# from .region_job import _SOURCE_PREFIX, _SOURCE_REGION
from .template_config import ExampleDataVar, ExampleSpatialTemplateConfig


class ExampleSpatialDynamicalDataset(
    DynamicalDataset[ExampleDataVar, ExampleSpatialSourceFileCoord]
):
    template_config: ExampleSpatialTemplateConfig = ExampleSpatialTemplateConfig()
    region_job_class: type[ExampleSpatialRegionJob] = ExampleSpatialRegionJob

    # A virtual dataset MUST set icechunk_virtual_config (a materialized dataset
    # leaves it None). It declares which source buckets the refs are allowed to point
    # into and how to split the chunk-reference manifest. Use default_factory because
    # icechunk's container objects can't be deep-copied as a plain pydantic default.
    #
    # icechunk_virtual_config: IcechunkVirtualConfig = Field(
    #     default_factory=lambda: IcechunkVirtualConfig(
    #         containers=(
    #             # One per source bucket; the prefix must match SourceFileCoord.get_url().
    #             icechunk.VirtualChunkContainer(
    #                 _SOURCE_PREFIX, icechunk.s3_store(region=_SOURCE_REGION)
    #             ),
    #         ),
    #         # Every commit rewrites each touched array's manifest split(s) and readers
    #         # download whole manifests, so split size bounds both. Size per array
    #         # group (a single int, or a {path_regex: size} mapping for datasets whose
    #         # groups have very different refs per append step); see "Manifest
    #         # splitting" in docs/virtual_datasets.md.
    #         manifest_split=manifest_append_dim_split(split_size=28, dim="init_time"),
    #     )
    # )

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron jobs that operationally update and validate this dataset.

        A virtual update is SINGLE-WRITER (it commits to the icechunk branch directly),
        so there is no workers_total/parallelism fan-out - one pod that polls through the
        source's publication window and exits once the manifest is complete.
        """
        # suspend = True  # Keeps updates and validation off until the store is backfilled; once the backfill is complete, remove via a PR so operational updates run.
        # operational_update_cron_job = ReformatCronJob(
        #     name=f"{self.dataset_id}-update",
        #     schedule="0 6 * * *",
        #     # Bound how long the single writer polls for files that never publish;
        #     # keep it well under the gap between fires so runs never overlap.
        #     pod_active_deadline=timedelta(hours=2),
        #     image=image_tag,
        #     dataset_id=self.dataset_id,
        #     cpu="1.7",
        #     memory="7G",
        #     secret_names=self.store_factory.k8s_secret_names(),
        #     suspend=suspend,
        # )
        # validation_cron_job = ValidationCronJob(
        #     name=f"{self.dataset_id}-validate",
        #     # After the update's fire + its deadline.
        #     schedule="0 8 * * *",
        #     pod_active_deadline=timedelta(minutes=30),
        #     image=image_tag,
        #     dataset_id=self.dataset_id,
        #     cpu="1.3",
        #     memory="7G",
        #     secret_names=self.store_factory.k8s_secret_names(),
        #     suspend=suspend,
        # )

        # return [operational_update_cron_job, validation_cron_job]
        raise NotImplementedError(
            f"Implement `operational_kubernetes_resources` on {self.__class__.__name__}"
        )

    def validators(self) -> Sequence[validation.DataValidator]:
        """Return the DataValidators to run on this dataset.

        Mix the generic xarray validators (which read the opened dataset) with the two
        virtual-specific ones, which need manifest/store access:
        - CheckVirtualManifestCompleteness: re-runs the operational filter to assert
          recent append-dim positions are sufficiently ingested (refs exist).
        - CheckVirtualDecodeHealth: decodes a sample of the references that exist to
          confirm the serializer and virtual-container authorization work end to end.
        """
        # return (
        #     partial(
        #         validation.check_forecast_current_data,
        #         max_latest_init_time_age=timedelta(hours=10),
        #     ),
        #     validation.CheckVirtualManifestCompleteness(),
        #     validation.CheckVirtualDecodeHealth(),
        # )
        raise NotImplementedError(
            f"Implement `validators` on {self.__class__.__name__}"
        )
