from reformatters.common.dynamical_dataset import DynamicalDataset


def test_dynamical_dataset_methods_exist():
    methods = [
        "update_template",
        "reformat_kubernetes",
        "reformat_local",
        "process_region_jobs",
    ]
    for method in methods:
        assert hasattr(DynamicalDataset, method), f"{method} not implemented"
