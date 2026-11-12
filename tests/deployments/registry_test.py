import importlib.util

import pytest

from reformatters import __main__ as cli
from reformatters.deployments import aws_us_east_1, aws_us_west_2


def test_cluster_registry_merge_preserves_order_and_instances() -> None:
    expected_datasets = [
        *aws_us_west_2.CLUSTER_DYNAMICAL_DATASETS,
        *aws_us_east_1.CLUSTER_DYNAMICAL_DATASETS,
    ]
    expected_archivers = [
        *aws_us_west_2.CLUSTER_OPERATIONAL_ARCHIVERS,
        *aws_us_east_1.CLUSTER_OPERATIONAL_ARCHIVERS,
    ]
    assert all(
        actual is expected
        for actual, expected in zip(
            cli.DYNAMICAL_DATASETS, expected_datasets, strict=True
        )
    )
    assert all(
        actual is expected
        for actual, expected in zip(
            cli.OPERATIONAL_ARCHIVERS, expected_archivers, strict=True
        )
    )


def test_east_owns_only_ecmwf_fast_datasets() -> None:
    assert {d.dataset_id for d in aws_us_east_1.CLUSTER_DYNAMICAL_DATASETS} == {
        "ecmwf-ifs-ens-forecast-0-25-degree-fast",
        "ecmwf-aifs-ens-forecast-fast",
    }
    assert aws_us_east_1.CLUSTER_OPERATIONAL_ARCHIVERS == []
    assert aws_us_west_2.CLUSTER_DYNAMICAL_DATASETS
    assert aws_us_west_2.CLUSTER_OPERATIONAL_ARCHIVERS


def test_cli_registers_merged_resources_in_order() -> None:
    assert [group.name for group in cli.app.registered_groups] == [
        resource.dataset_id
        for resource in (*cli.DYNAMICAL_DATASETS, *cli.OPERATIONAL_ARCHIVERS)
    ]


@pytest.mark.parametrize("collision", ["datasets", "archivers", "dataset-archiver"])
def test_merged_registry_rejects_duplicate_ids(
    monkeypatch: pytest.MonkeyPatch, collision: str
) -> None:
    if collision == "datasets":
        monkeypatch.setattr(
            aws_us_east_1,
            "CLUSTER_DYNAMICAL_DATASETS",
            [aws_us_west_2.CLUSTER_DYNAMICAL_DATASETS[0]],
        )
    else:
        resource = (
            aws_us_west_2.CLUSTER_OPERATIONAL_ARCHIVERS[0]
            if collision == "archivers"
            else aws_us_west_2.CLUSTER_DYNAMICAL_DATASETS[0]
        )
        monkeypatch.setattr(aws_us_east_1, "CLUSTER_OPERATIONAL_ARCHIVERS", [resource])

    spec = importlib.util.spec_from_file_location(
        "registry_duplicate_test", cli.__file__
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(AssertionError, match="Dataset and archiver IDs must be unique"):
        spec.loader.exec_module(module)
