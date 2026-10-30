import re
import time
from collections.abc import Sequence
from typing import Any, Literal

import fsspec
import icechunk
from icechunk.store import IcechunkStore
from pydantic import computed_field

from reformatters.common import storage
from reformatters.common.config import Config
from reformatters.common.iterating import item
from reformatters.common.kubernetes import CronJob, Job, ReformatCronJob
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day025DegreeDataset,
)

from .template_config import (
    DATASET_ID,
    END,  # noqa: F401
    START,  # noqa: F401
    VERSION,
    ProfilingTemplateConfig,
)

ROOT = "s3://dynamical-ecmwf-ifs-ens/_profiling/ifsens"


def run_root(run_id: str) -> str:
    assert re.fullmatch(r"[0-9a-f]{32}", run_id), "Invalid profiling run id"
    return f"{ROOT}/{run_id}"


class ProfilingStorage(StorageConfig):
    k8s_secret_name: str = "aws-open-data-icechunk-storage-options-key"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ICECHUNK

    def load_storage_options(self) -> dict[str, Any]:
        options = super().load_storage_options()
        assert not {"endpoint_url", "bucket", "prefix"}.intersection(options), (
            "Profiling requires the fixed AWS destination"
        )
        assert options.get("region", "us-west-2") == "us-west-2"
        return options


class ProfilingStoreFactory(StoreFactory):
    run_id: str

    @property
    def ownership_tag(self) -> str:
        return f"profiling-{self.run_id}"

    @property
    def evidence_url(self) -> str:
        self.assert_destination()
        return f"{run_root(self.run_id)}/profiles"

    def assert_destination(self) -> None:
        assert Config.is_prod, "Profiling requires explicit DYNAMICAL_ENV=prod"
        expected_base = f"{run_root(self.run_id)}/data"
        assert self.primary_storage_config.base_path == expected_base
        assert isinstance(self.primary_storage_config, ProfilingStorage)
        assert self.primary_storage_config.format == DatasetFormat.ICECHUNK
        assert (
            self.primary_storage_config.k8s_secret_name
            == ProfilingStorage.model_fields["k8s_secret_name"].default
        )
        assert self.dataset_id == DATASET_ID
        assert self.version == VERSION
        assert not self.replica_storage_configs
        assert self.icechunk_virtual_config is None
        assert self.primary_url() == f"{expected_base}/{DATASET_ID}/v{VERSION}.icechunk"
        assert (
            self._coordination_base_path() == f"{expected_base}/{DATASET_ID}/_internal"
        )

    def _coordination_fs(self) -> fsspec.AbstractFileSystem:
        self.assert_destination()
        return super()._coordination_fs()

    def _storage(self) -> icechunk.Storage:
        self.assert_destination()
        role, _store_path, primary = item(self._icechunk_storages())
        assert role == "primary"
        return primary

    def _verify_owner(self, repo: icechunk.Repository) -> None:
        assert self.ownership_tag in repo.list_tags(), (
            "Profiling repository is not owned by this run"
        )
        root_snapshot = repo.lookup_tag(self.ownership_tag)
        assert len(list(repo.ancestry(snapshot_id=root_snapshot))) == 1, (
            "Profiling ownership tag must identify the repository root"
        )
        assert list(repo.ancestry(branch="main"))[-1].id == root_snapshot, (
            "Profiling ownership tag must identify the root of main"
        )

    def open_owned_repository(self) -> icechunk.Repository:
        config, credentials = storage._repository_config_and_credentials(None)  # noqa: SLF001
        repo = icechunk.Repository.open(
            self._storage(), config=config, authorize_virtual_chunk_access=credentials
        )
        self._verify_owner(repo)
        return repo

    def bootstrap(self, worker_index: int, timeout_seconds: float = 1800) -> None:
        """Worker 0 creates the repository and tags its root; other workers wait for it."""
        target = self._storage()
        config, credentials = storage._repository_config_and_credentials(None)  # noqa: SLF001
        if worker_index == 0:
            repo = icechunk.Repository.create(
                target,
                config=config,
                authorize_virtual_chunk_access=credentials,
                check_clean_root=True,
            )
            repo.create_tag(self.ownership_tag, repo.lookup_branch("main"))
            self._verify_owner(repo)
            return

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if icechunk.Repository.exists(target):
                repo = icechunk.Repository.open(target, config=config)
                if self.ownership_tag in repo.list_tags():
                    self._verify_owner(repo)
                    return
                assert not repo.list_tags(), "Unexpected owner in profiling repository"
            time.sleep(1)
        raise TimeoutError("Worker 0 did not initialize the profiling repository")

    def primary_store(
        self, writable: bool = False, branch: str = "main"
    ) -> IcechunkStore:
        repo = self.open_owned_repository()
        session = (
            repo.writable_session(branch) if writable else repo.readonly_session(branch)
        )
        return session.store

    def icechunk_repos(
        self, *, sort: Literal["primary-first", "primary-last"]
    ) -> list[tuple[str, icechunk.Repository]]:
        assert sort in {"primary-first", "primary-last"}
        return [("primary", self.open_owned_repository())]


class ProfilingDataset(EcmwfIfsEnsForecast15Day025DegreeDataset):
    run_id: str
    template_config: ProfilingTemplateConfig = ProfilingTemplateConfig()

    @computed_field
    @property
    def store_factory(self) -> ProfilingStoreFactory:
        return ProfilingStoreFactory(
            primary_storage_config=self.primary_storage_config,
            replica_storage_configs=self.replica_storage_configs,
            dataset_id=self.dataset_id,
            template_config_version=self.template_config.version,
            run_id=self.run_id,
        )

    def operational_kubernetes_resources(
        self,
        image_tag: str,  # noqa: ARG002
    ) -> Sequence[CronJob]:
        return ()

    def _backfill_resource(self) -> ReformatCronJob:
        resource = item(
            EcmwfIfsEnsForecast15Day025DegreeDataset.operational_kubernetes_resources(
                self, image_tag="placeholder"
            )
        )
        assert isinstance(resource, ReformatCronJob)
        return resource

    def _backfill_job(
        self,
        command: Sequence[str],
        image: str,
        workers_total: int,
        parallelism: int,
        resource: ReformatCronJob,
    ) -> Job:
        job = super()._backfill_job(
            command, image, workers_total, parallelism, resource
        )
        from .worker import ProfilingJob  # noqa: PLC0415

        profiling_job = ProfilingJob(**job.model_dump(), run_id=self.run_id)
        profiling_job.as_kubernetes_object()
        profiling_job.report_destination()
        return profiling_job


def dataset(run_id: str) -> ProfilingDataset:
    result = ProfilingDataset(
        run_id=run_id,
        primary_storage_config=ProfilingStorage(base_path=f"{run_root(run_id)}/data"),
    )
    result.store_factory.assert_destination()
    return result
