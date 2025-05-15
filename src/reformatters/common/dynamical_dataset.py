import pydantic

from reformatters.common.region_job import RegionJob
from reformatters.common.template_config import TemplateConfig


class DynamicalDataset(pydantic.BaseModel):
    """
    Coordinates dataset template updates and RegionJob processing.
    """

    template_config: TemplateConfig
    region_job_class: type[RegionJob]

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
