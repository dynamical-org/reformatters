from typing import Generic, TypeVar, Any

import pydantic

from reformatters.common.config_models import DataVar
from reformatters.common.region_job import RegionJob, SourceFileCoord
from reformatters.common.template_config import TemplateConfig

DATA_VAR = TypeVar("DATA_VAR", bound=DataVar[Any])
SOURCE_FILE_COORD = TypeVar("SOURCE_FILE_COORD", bound=SourceFileCoord)


class DynamicalDataset(pydantic.BaseModel, Generic[DATA_VAR, SOURCE_FILE_COORD]):
    """
    Coordinates dataset template updates and RegionJob processing.
    """

    template_config: TemplateConfig[DATA_VAR]
    region_job_class: type[RegionJob[DATA_VAR, SOURCE_FILE_COORD]]

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
        """
        Orchestrate running RegionJob instances over all regions and data variables.
        """
        pass
