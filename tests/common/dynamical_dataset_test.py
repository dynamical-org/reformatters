import subprocess
from collections.abc import Iterable
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import ClassVar
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import sentry_sdk
import xarray as xr
import zarr
from pydantic import computed_field

from reformatters.common import docker, storage, template_utils, validation
from reformatters.common.config import Config, Env
from reformatters.common.config_models import (
    BaseInternalAttrs,
    DataVar,
    DataVarAttrs,
    Encoding,
)
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.region_job import RegionJob, SourceFileCoord
from reformatters.common.storage import _NO_SECRET_NAME, DatasetFormat, StorageConfig
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp

NOOP_STORAGE_CONFIG = StorageConfig(
    base_path="noop",
    k8s_secret_name="noop-secret",  # noqa: S106
    format=DatasetFormat.ZARR3,
)


class ExampleDatasetStorageConfig(StorageConfig):
    base_path: str = "s3://some-bucket/path"
    k8s_secret_name: str = "k8s-secret-name"  # noqa: S105
    format: DatasetFormat = DatasetFormat.ZARR3


class ExampleDataVar(DataVar[BaseInternalAttrs]):
    name: str = "var"
    encoding: Encoding = Encoding(
        dtype="float32",
        fill_value=np.nan,
        chunks=(1,),
        shards=(1,),
        compressors=[],
    )
    attrs: DataVarAttrs = DataVarAttrs(
        short_name="var",
        long_name="Variable",
        units="C",
        step_type="instant",
    )
    internal_attrs: BaseInternalAttrs = BaseInternalAttrs(keep_mantissa_bits=10)


class ExampleSourceFileCoord(SourceFileCoord):
    pass


class ExampleRegionJob(RegionJob[ExampleDataVar, ExampleSourceFileCoord]):
    max_vars_per_backfill_job: ClassVar[int] = 2


class ExampleConfig(TemplateConfig[ExampleDataVar]):
    dims: tuple[Dim, ...] = ("time",)
    append_dim: AppendDim = "time"
    append_dim_start: Timestamp = pd.Timestamp("2000-01-01")
    append_dim_frequency: Timedelta = pd.Timedelta("1D")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_id(self) -> str:
        return "example-dataset"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def version(self) -> str:
        return "1.2.3"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> list[ExampleDataVar]:
        return [ExampleDataVar()]


class ExampleDataset(DynamicalDataset[ExampleDataVar, ExampleSourceFileCoord]):
    template_config: ExampleConfig = ExampleConfig()
    region_job_class: type[ExampleRegionJob] = ExampleRegionJob
    primary_storage_config: ExampleDatasetStorageConfig = ExampleDatasetStorageConfig()

    def operational_kubernetes_resources(self, image_tag: str) -> Iterable[CronJob]:
        return [
            ReformatCronJob(
                name=f"{self.dataset_id}-update",
                schedule="0 0 * * *",
                pod_active_deadline=timedelta(minutes=30),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1",
                memory="1G",
                shared_memory="1G",
                ephemeral_storage="1G",
                secret_names=self.store_factory.k8s_secret_names(),
            ),
            ValidationCronJob(
                name=f"{self.dataset_id}-validate",
                schedule="0 0 * * *",
                pod_active_deadline=timedelta(minutes=30),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1",
                memory="1G",
                shared_memory="1G",
                ephemeral_storage="1G",
                secret_names=self.store_factory.k8s_secret_names(),
            ),
        ]


def test_dynamical_dataset_methods_exist() -> None:
    methods = [
        "update_template",
        "backfill_kubernetes",
        "backfill_local",
        "process_backfill_region_jobs",
        "update",
        "validate_dataset",
    ]
    for method in methods:
        assert hasattr(DynamicalDataset, method), f"{method} missing"


def test_dynamical_dataset_init() -> None:
    ExampleDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
    )


def test_update_template(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_update_template = Mock()
    monkeypatch.setattr(ExampleConfig, "update_template", mock_update_template)
    template_config = ExampleConfig()

    dataset = ExampleDataset(
        template_config=template_config,
        region_job_class=ExampleRegionJob,
    )

    dataset.update_template()
    mock_update_template.assert_called_once()


def test_process_backfill_region_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_job0 = Mock()
    mock_job0.summary = lambda: "job0-summary"
    mock_job1 = Mock()
    mock_job1.summary = lambda: "job1-summary"
    monkeypatch.setattr(
        ExampleRegionJob,
        "get_jobs",
        classmethod(lambda cls, *args, **kwargs: [mock_job0, mock_job1]),
    )
    monkeypatch.setattr(template_utils, "write_metadata", Mock())

    monkeypatch.setattr(ExampleConfig, "get_template", lambda self, end: xr.Dataset())
    dataset = ExampleDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
    )
    mock_job0.process = Mock(
        return_value=(
            {},
            dataset.store_factory.primary_store(),
            dataset.store_factory.replica_stores(),
        )
    )
    mock_job1.process = Mock(
        return_value=(
            {},
            dataset.store_factory.primary_store(),
            dataset.store_factory.replica_stores(),
        )
    )

    dataset.process_backfill_region_jobs(
        pd.Timestamp("2000-01-02"),
        "test-job-name",
        worker_index=0,
        workers_total=1,
    )

    mock_job0.process.assert_called_once()
    mock_job1.process.assert_called_once()


def test_backfill_local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mock_job0 = Mock()
    mock_job0.summary = lambda: "job0-summary"
    monkeypatch.setattr(
        ExampleRegionJob,
        "get_jobs",
        classmethod(lambda cls, *args, **kwargs: [mock_job0]),
    )
    monkeypatch.setattr(
        ExampleConfig,
        "get_template",
        lambda self, end: xr.Dataset(attrs={"cool": "weather"}),
    )
    process_backfill_region_jobs_mock = Mock()
    monkeypatch.setattr(
        ExampleDataset,
        "process_backfill_region_jobs",
        process_backfill_region_jobs_mock,
    )

    dataset = ExampleDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
        replica_storage_configs=[
            ExampleDatasetStorageConfig(
                base_path="s3://replica-bucket-a/path",
                format=DatasetFormat.ZARR3,
            ),
            ExampleDatasetStorageConfig(
                base_path="s3://replica-bucket-b/path",
                format=DatasetFormat.ICECHUNK,
            ),
        ],
    )
    _original_get_store_path = storage._get_store_path

    def _get_store_path(
        dataset_id: str, version: str, storage_config: StorageConfig
    ) -> str:
        if storage_config.base_path == "s3://replica-bucket-a/path":
            return str(tmp_path / "replica-bucket-a" / f"{dataset_id}/v{version}.zarr")
        else:
            return _original_get_store_path(dataset_id, version, storage_config)

    monkeypatch.setattr(storage, "_get_store_path", _get_store_path)

    mock_job0.process = Mock(
        side_effect=lambda *args, **kwargs: (
            {},
            dataset.store_factory.primary_store(),
            dataset.store_factory.replica_stores(),
        )
    )
    dataset.backfill_local(pd.Timestamp("2000-01-02"))

    assert (
        xr.open_zarr(dataset.store_factory.primary_store()).attrs["cool"] == "weather"
    )

    for replica_store in dataset.store_factory.replica_stores():
        assert xr.open_zarr(replica_store).attrs["cool"] == "weather", (
            "replica store should have the same metadata as the primary store"
        )

    process_backfill_region_jobs_mock.assert_called_once_with(
        pd.Timestamp("2000-01-02"),
        worker_index=0,
        workers_total=1,
        reformat_job_name="local",
        filter_start=None,
        filter_end=None,
        filter_contains=None,
        filter_variable_names=None,
    )


def test_backfill_kubernetes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Generally monkeypatching _can_run_in_kuberneretes is dangerous.
    # However, we need to test the internals of backfill_kubernetes so we override this here.
    monkeypatch.setattr(
        DynamicalDataset, "_can_run_in_kubernetes", Mock(return_value=True)
    )
    mock_run = Mock()
    monkeypatch.setattr(subprocess, "run", mock_run)

    # Simulate 5 backfill jobs
    monkeypatch.setattr(
        ExampleRegionJob,
        "get_jobs",
        classmethod(lambda cls, *args, **kwargs: [object()] * 5),
    )

    # Mock template retrieval and metadata writing
    monkeypatch.setattr(ExampleConfig, "get_template", lambda self, end: xr.Dataset())
    monkeypatch.setattr(template_utils, "write_metadata", lambda *args, **kwargs: None)

    dataset = ExampleDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
    )
    store_factory = Mock()
    monkeypatch.setattr(ExampleDataset, "store_factory", store_factory)
    monkeypatch.setattr(store_factory, "primary_store", lambda: tmp_path)
    monkeypatch.setattr(store_factory, "k8s_secret_names", lambda: [_NO_SECRET_NAME])

    dataset.backfill_kubernetes(
        append_dim_end=datetime(2025, 1, 1),
        jobs_per_pod=2,
        max_parallelism=10,
        filter_start=datetime(2000, 1, 1),
        filter_end=datetime(2020, 1, 1),
        filter_variable_names=["a", "b"],
        docker_image="my-docker-image",
    )

    assert mock_run.call_count == 1
    _, kwargs = mock_run.call_args
    input_str = kwargs["input"]

    # workers_total = ceil(5/2) == 3
    assert '"completions": 3' in input_str
    # Command and filters
    assert (
        '"example-dataset", "process-backfill-region-jobs", "2025-01-01T00:00:00'
        in input_str
    )
    assert "--filter-start=2000-01-01T00:00:00" in input_str
    assert "--filter-end=2020-01-01T00:00:00" in input_str
    assert "--filter-variable-names=a" in input_str
    assert "--filter-variable-names=b" in input_str
    # Docker image
    assert '"my-docker-image"' in input_str


def test_validate_dataset_calls_validators(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configured_validators = [Mock(), Mock()]
    monkeypatch.setattr(
        ExampleDataset, "validators", lambda self: configured_validators
    )

    mock_validate = Mock()
    monkeypatch.setattr(validation, "validate_dataset", mock_validate)

    dataset = ExampleDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
        replica_storage_configs=[
            ExampleDatasetStorageConfig(
                base_path="s3://replica-bucket-a/path",
                format=DatasetFormat.ZARR3,
            ),
        ],
    )

    store_factory = Mock()
    monkeypatch.setattr(ExampleDataset, "store_factory", store_factory)

    mock_store = Mock()
    monkeypatch.setattr(store_factory, "primary_store", lambda: mock_store)

    mock_replica_store = Mock()
    monkeypatch.setattr(store_factory, "replica_stores", lambda: [mock_replica_store])

    mock_replica_store_ds = Mock()
    monkeypatch.setattr(
        xr, "open_zarr", lambda store, chunks=None: mock_replica_store_ds
    )

    dataset.validate_dataset("example-job-name")

    # Check that we have exactly 2 calls
    assert mock_validate.call_count == 2

    # Check the first call (primary store)
    positional_args, keyword_args = mock_validate.call_args_list[0]
    assert positional_args == (mock_store,)
    primary_store_validators = keyword_args["validators"]

    assert primary_store_validators[:-1] == configured_validators
    assert isinstance(primary_store_validators[-1], partial)
    assert primary_store_validators[-1].func == validation.check_for_expected_shards
    assert primary_store_validators[-1].args == (mock_store,)

    # Check the second call (replica store)
    positional_args, keyword_args = mock_validate.call_args_list[1]
    assert positional_args == (mock_replica_store,)
    replica_validators = keyword_args["validators"]

    # Verify replica validators = base validators + compare_replica_and_primary partial
    assert len(replica_validators) == len(configured_validators) + 2
    assert replica_validators[:-2] == configured_validators

    assert replica_validators[-2].func == validation.check_for_expected_shards
    assert replica_validators[-2].args == (mock_replica_store,)

    assert replica_validators[-1].func == validation.compare_replica_and_primary
    assert replica_validators[-1].args == (
        dataset.template_config.append_dim,
        mock_replica_store_ds,
    )


def test_monitor_context_success_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(type(Config), "is_sentry_enabled", True)

    dataset = ExampleDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
    )

    # Mock capture_checkin to record statuses
    mock_capture = Mock()
    monkeypatch.setattr(sentry_sdk.crons, "capture_checkin", mock_capture)

    # Success case: should record "in_progress" then "ok"
    with dataset._monitor(ReformatCronJob, "job-name"):
        pass
    statuses = [c.kwargs["status"] for c in mock_capture.call_args_list]
    assert statuses == ["in_progress", "ok"]

    # Error case: should record "in_progress" then "error"
    mock_capture.reset_mock()
    with pytest.raises(ValueError, match="failure"):  # noqa: SIM117
        with dataset._monitor(ReformatCronJob, "job-name"):
            raise ValueError("failure")
    statuses = [c.kwargs["status"] for c in mock_capture.call_args_list]
    assert statuses == ["in_progress", "error"]


def test_monitor_without_sentry(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test that it's ok to not define operational_kubernetes_resources if sentry reporting is disabled

    # disable sentry reporting
    monkeypatch.setattr(type(Config), "is_sentry_enabled", False)
    dataset = ExampleDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
    )

    # make operational_kubernetes_resources raise if called
    def fail_resources(image_tag: str) -> None:
        raise RuntimeError("operational_kubernetes_resources should not be called")

    monkeypatch.setattr(
        ExampleDataset, "operational_kubernetes_resources", fail_resources
    )

    # this should not raise
    with dataset._monitor(ReformatCronJob, "job"):
        pass


def test_backfill_kubernetes_overwrite_existing_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Generally monkeypatching _can_run_in_kuberneretes is dangerous.
    # However, we need to test the internals of backfill_kubernetes so we override this here.
    monkeypatch.setattr(
        DynamicalDataset, "_can_run_in_kubernetes", Mock(return_value=True)
    )

    dataset = ExampleDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
        primary_storage_config=ExampleDatasetStorageConfig(
            base_path="s3://bucket/data",
            format=DatasetFormat.ZARR3,
        ),
        replica_storage_configs=[
            ExampleDatasetStorageConfig(
                base_path="s3://bucket/data",
                format=DatasetFormat.ICECHUNK,
            ),
        ],
    )
    # Open stores as writable so that they are created
    # (this is only necessary for icechunk stores)
    dataset.store_factory.primary_store(writable=True)
    dataset.store_factory.replica_stores(writable=True)

    monkeypatch.setattr(xr, "open_zarr", Mock())

    monkeypatch.setattr(
        docker, "build_and_push_image", Mock(return_value="test-image-tag")
    )
    monkeypatch.setattr(subprocess, "run", Mock())
    monkeypatch.setattr(ExampleConfig, "get_template", lambda self, end: xr.Dataset())
    monkeypatch.setattr(
        ExampleRegionJob, "get_jobs", Mock(return_value=[Mock(spec=ExampleRegionJob)])
    )
    monkeypatch.setattr(
        ExampleDataset,
        "process_backfill_region_jobs",
        Mock(),
    )
    mock_write_metadata = Mock()
    monkeypatch.setattr(template_utils, "write_metadata", mock_write_metadata)

    dataset.backfill_kubernetes(
        append_dim_end=pd.Timestamp("2000-01-02"),
        jobs_per_pod=1,
        max_parallelism=1,
        overwrite_existing=True,
    )
    mock_write_metadata.assert_not_called()
    mock_write_metadata.reset_mock()

    dataset.backfill_kubernetes(
        append_dim_end=pd.Timestamp("2000-01-02"),
        jobs_per_pod=1,
        max_parallelism=1,
        overwrite_existing=False,
    )
    mock_write_metadata.assert_called_once()


def test_backfill_kubernetes_overwrite_existing_flag_fails_if_not_all_stores_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        DynamicalDataset, "_can_run_in_kubernetes", Mock(return_value=True)
    )
    monkeypatch.setattr(
        xr,
        "open_zarr",
        Mock(side_effect=zarr.errors.GroupNotFoundError("Group not found")),
    )
    monkeypatch.setattr(ExampleConfig, "get_template", lambda self, end: xr.Dataset())

    monkeypatch.setattr(
        docker, "build_and_push_image", Mock(return_value="test-image-tag")
    )
    monkeypatch.setattr(subprocess, "run", Mock())

    dataset = ExampleDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
        primary_storage_config=ExampleDatasetStorageConfig(
            base_path="s3://bucket/data",
            format=DatasetFormat.ZARR3,
        ),
    )

    with pytest.raises(
        AssertionError,
        match="Not all stores exist, cannot run with overwrite_existing=True",
    ):
        dataset.backfill_kubernetes(
            append_dim_end=pd.Timestamp("2000-01-02"),
            jobs_per_pod=1,
            max_parallelism=1,
            overwrite_existing=True,
        )


@pytest.mark.parametrize("env", [Env.dev, Env.test])
def test_backfill_kubernetes_overwrite_existing_flag_fails_in_wrong_environment(
    monkeypatch: pytest.MonkeyPatch,
    env: Env,
) -> None:
    monkeypatch.setattr(Config, "env", env)
    dataset = ExampleDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
    )
    with pytest.raises(
        AssertionError,
        match="backfill_kubernetes is only supported in prod environment",
    ):
        dataset.backfill_kubernetes(
            append_dim_end=pd.Timestamp("2000-01-02"),
            jobs_per_pod=1,
            max_parallelism=1,
            overwrite_existing=True,
        )


@pytest.mark.parametrize("env", [Env.dev, Env.test, "not-dev-or-test"])
def test_backfill_local_fails_in_wrong_environment(
    monkeypatch: pytest.MonkeyPatch, env: Env | str
) -> None:
    monkeypatch.setattr(Config, "env", env)
    monkeypatch.setattr(template_utils, "write_metadata", Mock())
    monkeypatch.setattr(DynamicalDataset, "process_backfill_region_jobs", Mock())
    monkeypatch.setattr(
        DynamicalDataset, "_get_template", Mock(return_value=xr.Dataset())
    )

    dataset = ExampleDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
    )
    if env == "not-dev-or-test":
        with pytest.raises(
            AssertionError,
            match="backfill_local is only supported in dev or test environments",
        ):
            dataset.backfill_local(append_dim_end=pd.Timestamp("2000-01-02"))
    else:
        dataset.backfill_local(append_dim_end=pd.Timestamp("2000-01-02"))
