from typing import Any, Generic, TypeVar

import pydantic

from reformatters.common.config_models import DataVar
from reformatters.common.region_job import RegionJob, SourceFileCoord
from reformatters.common.template_config import TemplateConfig

DV = TypeVar("DV", bound=DataVar[Any])
SFC = TypeVar("SFC", bound=SourceFileCoord)


class DynamicalDataset(pydantic.BaseModel, Generic[DV, SFC]):
    """
    Top level class managing a dataset configuration and processing.
    Enforces that template_config and region_job_class share the same DataVar/SFC types.
    """

    template_config: TemplateConfig[DV]
    region_job_class: type[RegionJob[DV, SFC]]

    def update_template(self) -> None:
        """Generate and persist the dataset template using the template_config."""
        pass

    def reformat_kubernetes(self) -> None:
        """Run dataset reformatting using Kubernetes index jobs."""
        pass

    def reformat_local(self) -> None:
        """Run dataset reformatting locally in this process."""
        pass

    def process_region_jobs(self) -> None:
        """Orchestrate running RegionJob instances."""
        pass
