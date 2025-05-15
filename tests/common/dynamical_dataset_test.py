from typing import ClassVar
from unittest.mock import Mock

import pandas as pd
import pytest

from reformatters.common.config_models import (
    BaseInternalAttrs,
    DatasetAttributes,
    DataVar,
)
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.region_job import RegionJob, SourceFileCoord
from reformatters.common.template_config import TemplateConfig
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp


class ExampleDataVar(DataVar[BaseInternalAttrs]):
    name: str = "var"
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

    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="example",
            dataset_version="v1",
            name="example",
            description="example",
            attribution="example",
            spatial_domain="",
            spatial_resolution="",
            time_domain="",
            time_resolution="",
            forecast_domain=None,
            forecast_resolution=None,
        )


class ExampleDataset(DynamicalDataset[ExampleDataVar, ExampleSourceFileCoord]):
    template_config: ExampleConfig = ExampleConfig()
    region_job_class: type[ExampleRegionJob] = ExampleRegionJob


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
    mock_job1 = Mock()
    monkeypatch.setattr(
        ExampleRegionJob,
        "get_backfill_jobs",
        classmethod(lambda cls, *args, **kwargs: [mock_job0, mock_job1]),
    )
    dataset = ExampleDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
    )
    dataset.process_region_jobs(
        pd.Timestamp("2000-01-02"),
        worker_index=0,
        workers_total=1,
    )
    mock_job0.process.assert_called_once()
    mock_job1.process.assert_called_once()
