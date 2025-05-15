import pydantic
from typing import Generic, Type, TypeVar

from reformatters.common.region_job import RegionJob
from reformatters.common.template_config import TemplateConfig

DATA_VAR = TypeVar("DATA_VAR")
SOURCE_FILE_COORD = TypeVar("SOURCE_FILE_COORD")


class DynamicalDataset(pydantic.BaseModel, Generic[DATA_VAR, SOURCE_FILE_COORD]):
    """
    Coordinates dataset template updates and RegionJob processing.
    """

    template_config: TemplateConfig[DATA_VAR]
    region_job_class: Type[RegionJob[DATA_VAR, SOURCE_FILE_COORD]]

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
