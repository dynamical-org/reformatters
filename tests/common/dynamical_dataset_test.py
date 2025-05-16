from pathlib import Path
from typing import ClassVar
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import computed_field

from reformatters.common.config_models import (
    BaseInternalAttrs,
    DataVar,
    DataVarAttrs,
    Encoding,
)
from reformatters.common.dynamical_dataset import DynamicalDataset
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
        "get_backfill_jobs",
        classmethod(lambda cls, *args, **kwargs: [mock_job0, mock_job1]),
    )
    monkeypatch.setattr(ExampleConfig, "get_template", lambda self, end: xr.Dataset())
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


def test_reformat_local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mock_job0 = Mock()
    mock_job0.summary = lambda: "job0-summary"
    monkeypatch.setattr(
        ExampleRegionJob,
        "get_backfill_jobs",
        classmethod(lambda cls, *args, **kwargs: [mock_job0]),
    )
    monkeypatch.setattr(
        ExampleConfig,
        "get_template",
        lambda self, end: xr.Dataset(attrs={"cool": "weather"}),
    )
    monkeypatch.setattr(ExampleDataset, "_store", lambda _self: tmp_path)
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
        filter_start=None,
        filter_end=None,
        filter_variable_names=None,
    )
