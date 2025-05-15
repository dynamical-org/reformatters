from reformatters.common.dynamical_dataset import DynamicalDataset

# AI, make an ExampleDataset, ExampleRegionJob, and ExampleConfig which use each other.
# you might need to make an ExampleDataVar subclass too AI!


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
    dataset = DynamicalDataset(
        template_config=ExampleConfig(),
        region_job_class=ExampleRegionJob,
    )
