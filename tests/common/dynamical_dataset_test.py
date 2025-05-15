from typing import ClassVar, Type

import pandas as pd

from reformatters.common.config_models import AppendDim, BaseInternalAttrs, DataVar, Dim
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.region_job import RegionJob, SourceFileCoord
from reformatters.common.template_config import TemplateConfig


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
    append_dim_start: pd.Timestamp = pd.Timestamp("2000-01-01")
    append_dim_frequency: pd.Timedelta = pd.Timedelta("1D")


class ExampleDataset(DynamicalDataset[ExampleDataVar, ExampleSourceFileCoord]):
    template_config: ExampleConfig
    region_job_class: Type[RegionJob[ExampleDataVar, ExampleSourceFileCoord]]


def test_dynamical_dataset_methods_exist() -> None:
    methods = [
        "update_template",
        "reformat_kubernetes",
        "reformat_local",
        "process_region_jobs",
    ]
    for method in methods:
        assert hasattr(DynamicalDataset, method), f"{method} not implemented"


def test_dynamical_dataset_init() -> None:
    dataset = ExampleDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
    )
