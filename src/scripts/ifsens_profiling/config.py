import re
import time
from typing import Any, Literal

import fsspec
import icechunk
import pandas as pd
from icechunk.store import IcechunkStore
from pydantic import computed_field

from reformatters.common import storage
from reformatters.common.config import Config
from reformatters.common.config_models import DatasetAttributes
from reformatters.common.pydantic import replace
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.common.types import Timestamp
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day025DegreeDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
)

BASELINE = "44a8fede0b31e337093d616fcac379b08195afb6"
DATASET_ID = "ecmwf-ifs-ens-forecast-15-day-0-25-degree"
VERSION = "0.0.0-profiling-20261029"
ROOT = "s3://dynamical-ecmwf-ifs-ens/_profiling/ifsens-20261029"
START = pd.Timestamp("2024-03-30T00:00Z")
END = pd.Timestamp("2024-04-01T00:00Z")
type Leg = Literal["jpp1", "jpp10"]


def run_root(run_token: str) -> str:
    assert re.fullmatch(r"p\d{8}-[a-f0-9]{12}", run_token), "Invalid pilot run token"
    return f"{ROOT}/{run_token}"


class ProfilingTemplate(EcmwfIfsEnsForecast15Day025DegreeTemplateConfig):
    # Zarr datetime coordinates store UTC without a timezone extension dtype.
    append_dim_start: Timestamp = START.tz_localize(None)

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return replace(super().dataset_attributes, dataset_version=VERSION)


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
    run_token: str
    leg: Leg

    @property
    def ownership_tag(self) -> str:
        return f"profiling-{self.run_token}-{self.leg}"

    @property
    def evidence_url(self) -> str:
        self.assert_destination()
        return f"{run_root(self.run_token)}/profiles/{self.leg}"

    def assert_destination(self) -> None:
        assert Config.is_prod, "Profiling requires explicit DYNAMICAL_ENV=prod"
        assert self.leg in {"jpp1", "jpp10"}
        expected_base = f"{run_root(self.run_token)}/data/{self.leg}"
        assert self.primary_storage_config.base_path == expected_base
        assert isinstance(self.primary_storage_config, ProfilingStorage)
        assert self.primary_storage_config.format == DatasetFormat.ICECHUNK
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
        entries = self._icechunk_storages()
        assert len(entries) == 1
        assert entries[0][0] == "primary"
        return entries[0][2]

    def _verify_owner(self, repo: icechunk.Repository) -> None:
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
    run_token: str
    leg: Leg
    template_config: ProfilingTemplate = ProfilingTemplate()

    @computed_field
    @property
    def store_factory(self) -> ProfilingStoreFactory:
        return ProfilingStoreFactory(
            primary_storage_config=self.primary_storage_config,
            replica_storage_configs=self.replica_storage_configs,
            dataset_id=self.dataset_id,
            template_config_version=self.template_config.version,
            run_token=self.run_token,
            leg=self.leg,
        )


def dataset(run_token: str, leg: Leg) -> ProfilingDataset:
    result = ProfilingDataset(
        run_token=run_token,
        leg=leg,
        primary_storage_config=ProfilingStorage(
            base_path=f"{run_root(run_token)}/data/{leg}"
        ),
    )
    result.store_factory.assert_destination()
    return result
