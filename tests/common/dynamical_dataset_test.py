import subprocess
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import computed_field

from reformatters.common import template_utils, validation
from reformatters.common.config_models import (
    BaseInternalAttrs,
    DataVar,
    DataVarAttrs,
    Encoding,
)
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import Job, ReformatCronJob
from reformatters.common.region_job import RegionJob, SourceFileCoord
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp


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

    def operational_kubernetes_resources(self, image_tag: str) -> Iterable[Job]:
        return [
            ReformatCronJob(
                name=f"{self.dataset_id}-operational-update",
                schedule="0 0 * * *",
                pod_active_deadline=timedelta(minutes=30),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1",
                memory="1G",
                shared_memory="1G",
                ephemeral_storage="1G",
            )
        ]


def test_dynamical_dataset_methods_exist() -> None:
    methods = [
        "update_template",
        "reformat_kubernetes",
        "reformat_local",
        "process_region_jobs",
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


def test_process_region_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_job0 = Mock()
    mock_job0.summary = lambda: "job0-summary"
    mock_job1 = Mock()
    mock_job1.summary = lambda: "job1-summary"
    monkeypatch.setattr(
        ExampleRegionJob,
        "get_jobs",
        classmethod(lambda cls, *args, **kwargs: [mock_job0, mock_job1]),
    )
    monkeypatch.setattr(ExampleConfig, "get_template", lambda self, end: xr.Dataset())
    dataset = ExampleDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
    )

    dataset.process_region_jobs(
        pd.Timestamp("2000-01-02"),
        "test-job-name",
        worker_index=0,
        workers_total=1,
    )

    mock_job0.process.assert_called_once()
    mock_job1.process.assert_called_once()


def test_reformat_local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    monkeypatch.setattr(ExampleDataset, "_final_store", lambda _self: tmp_path)
    process_region_jobs_mock = Mock()
    monkeypatch.setattr(ExampleDataset, "process_region_jobs", process_region_jobs_mock)

    dataset = ExampleDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
    )

    dataset.reformat_local(pd.Timestamp("2000-01-02"))

    assert xr.open_zarr(tmp_path).attrs["cool"] == "weather"
    process_region_jobs_mock.assert_called_once_with(
        pd.Timestamp("2000-01-02"),
        worker_index=0,
        workers_total=1,
        reformat_job_name="local",
        filter_start=None,
        filter_end=None,
        filter_variable_names=None,
    )


def test_reformat_kubernetes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    monkeypatch.setattr(ExampleDataset, "_final_store", lambda self: tmp_path)
    monkeypatch.setattr(template_utils, "write_metadata", lambda *args, **kwargs: None)

    dataset = ExampleDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
    )

    dataset.reformat_kubernetes(
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
    assert '"example-dataset", "process-region-jobs", "2025-01-01T00:00:00' in input_str
    assert "--filter-start=2000-01-01T00:00:00" in input_str
    assert "--filter-end=2020-01-01T00:00:00" in input_str
    assert "--filter-variable-names=a" in input_str
    assert "--filter-variable-names=b" in input_str
    # Docker image
    assert '"my-docker-image"' in input_str


def test_validate_zarr_calls_validators_and_uses_final_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mock_validators = [Mock(), Mock()]
    monkeypatch.setattr(ExampleDataset, "validators", lambda self: mock_validators)

    mock_store = Mock()
    monkeypatch.setattr(ExampleDataset, "_final_store", lambda self: mock_store)

    mock_validate = Mock()
    monkeypatch.setattr(validation, "validate_zarr", mock_validate)

    dataset = ExampleDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
    )

    dataset.validate_zarr("example-job-name")

    # Ensure validate_zarr was called with correct arguments
    # this implies
    # - self._final_store() was called and returned our mock_store
    # - self.validators() was called and returned our mock_validators
    mock_validate.assert_called_once_with(mock_store, validators=mock_validators)
