from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from kubernetes.utils import parse_quantity

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import validation
from reformatters.common.iterating import get_worker_jobs, item, walk_data_arrays
from reformatters.common.kubernetes import ReformatCronJob
from reformatters.common.types import DatetimeLike
from reformatters.ecmwf.archive_gribs.forecast_46_day_archiver import (
    MATERIALIZED_PRODUCT_ECDS_VARIABLES,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_1_5_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast46Day15DegreeDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_6_hourly_1_5_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset,
)
from tests.chunk_utils import shrink_chunks_and_shards
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> EcmwfIfsEnsForecast46Day15DegreeDataset:
    return EcmwfIfsEnsForecast46Day15DegreeDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG
    )


def test_daily_identity_matches_registry_template_and_storage() -> None:
    dataset = next(
        dataset
        for dataset in DYNAMICAL_DATASETS
        if isinstance(dataset, EcmwfIfsEnsForecast46Day15DegreeDataset)
    )
    dataset_id = "ecmwf-ifs-ens-forecast-46-day-daily-1-5-degree"
    assert dataset.dataset_id == dataset_id
    assert "ecmwf-ifs-ens-forecast-46-day-1-5-degree" not in {
        dataset.dataset_id for dataset in DYNAMICAL_DATASETS
    }
    assert dataset.store_factory.primary_url() == (
        f"s3://dynamical-ecmwf-ifs-ens/{dataset_id}/v0.2.0.icechunk"
    )
    config = dataset.template_config
    template = config.get_template(
        config.append_dim_start + config.append_dim_frequency
    )
    for node in template.subtree:
        assert node.attrs["dataset_id"] == dataset_id
        assert node.attrs["name"] == "ECMWF IFS ENS forecast, 46 day, daily, 1.5 degree"
        np.testing.assert_array_equal(
            node["lead_time"], pd.timedelta_range("0h", "1104h", freq="24h")
        )


def test_validators_check_masked_variables_are_not_all_nan(
    dataset: EcmwfIfsEnsForecast46Day15DegreeDataset,
) -> None:
    validators = tuple(dataset.validators())

    assert len(validators) == 3
    assert isinstance(validators[1], validation.CheckRecentNans)
    assert isinstance(validators[2], validation.CheckRecentNans)
    assert validators[2].include_vars == validators[1].exclude_vars
    assert validators[2].max_nan_fraction == 0.9999
    assert validators[2].spatial_sampling == "quarter"


def test_operational_cron_jobs_are_not_suspended(
    dataset: EcmwfIfsEnsForecast46Day15DegreeDataset,
) -> None:
    update_cron_job, validation_cron_job = dataset.operational_kubernetes_resources(
        "test-image-tag"
    )

    assert update_cron_job.name == "ecmwf-ifs-ens-46-day-daily-update"
    assert update_cron_job.suspend is False
    assert validation_cron_job.name == "ecmwf-ifs-ens-46-day-daily-validate"
    assert validation_cron_job.suspend is False
    for cron_job in (update_cron_job, validation_cron_job):
        manifest = cron_job.as_kubernetes_object()
        assert manifest["metadata"]["name"] == cron_job.name
        assert manifest["spec"]["suspend"] is False
        command = manifest["spec"]["jobTemplate"]["spec"]["template"]["spec"][
            "containers"
        ][0]["command"]
        assert command[2] == "ecmwf-ifs-ens-forecast-46-day-daily-1-5-degree"


def test_archive_contains_every_dataset_source_variable(
    dataset: EcmwfIfsEnsForecast46Day15DegreeDataset,
) -> None:
    assert {
        data_var.internal_attrs.ecds_variable
        for data_var in dataset.template_config.data_vars
    } == set(MATERIALIZED_PRODUCT_ECDS_VARIABLES["daily"])


@pytest.mark.parametrize(
    "dataset",
    [
        EcmwfIfsEnsForecast46Day15DegreeDataset(
            primary_storage_config=NOOP_STORAGE_CONFIG
        ),
        EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset(
            primary_storage_config=NOOP_STORAGE_CONFIG
        ),
    ],
    ids=["daily", "6-hourly"],
)
@pytest.mark.parametrize("init_count", [1, 2, 5])
def test_operational_workers_cover_jobs_and_fit_shared_memory(
    dataset: EcmwfIfsEnsForecast46Day15DegreeDataset
    | EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset,
    init_count: int,
    tmp_path: Path,
) -> None:
    update = item(
        resource
        for resource in dataset.operational_kubernetes_resources("test-image")
        if isinstance(resource, ReformatCronJob)
    )
    config = dataset.template_config
    template = config.get_template(
        config.append_dim_start + init_count * config.append_dim_frequency
    )
    jobs = dataset.region_job_class.get_jobs(
        tmp_store=tmp_path / "tmp.zarr",
        template_ds=template,
        append_dim=config.append_dim,
        all_data_vars=config.data_vars,
        reformat_job_name="test-update",
    )
    assignments = [
        get_worker_jobs(
            jobs,
            worker,
            update.workers_total,
            worker_assignment=dataset.region_job_class.worker_assignment,
        )
        for worker in range(update.workers_total)
    ]
    assert Counter(repr(job) for jobs in assignments for job in jobs) == Counter(
        repr(job) for job in jobs
    )
    assert all(assignments)
    if init_count <= 2:
        assert max(map(len, assignments)) <= 4

    assert update.shared_memory is not None
    for job in jobs:
        region = template.isel({config.append_dim: job.get_processing_region()})
        largest_buffer = max(array.nbytes for _, array in walk_data_arrays(region))
        assert largest_buffer < parse_quantity(update.shared_memory)


LEVELS = (1000, 925, 850, 700, 500, 300, 200, 100, 50, 10)


@pytest.mark.slow
def test_backfill_local_writes_pressure_levels(
    monkeypatch: pytest.MonkeyPatch, dataset: EcmwfIfsEnsForecast46Day15DegreeDataset
) -> None:
    """A pressure level variable lands on its declared axis, with real values."""
    root_var, pressure_var = "pressure_surface", "temperature"
    monkeypatch.setattr(
        type(dataset.template_config),
        "data_vars",
        [
            var
            for var in dataset.template_config.data_vars
            if var.name in (root_var, pressure_var)
        ],
    )

    orig_get_template = dataset.template_config.get_template

    def small_template(self: object, end_time: DatetimeLike) -> xr.DataTree:
        tree = orig_get_template(end_time).sel(
            lead_time=slice("0h", "24h"), ensemble_member=slice(0, 0)
        )
        # Leave init_time at production geometry: one shard per init, so
        # filter_start scopes the run to a single initialization.
        return shrink_chunks_and_shards(
            xr.DataTree.from_dict(
                {
                    "/": tree.to_dataset()[[root_var]],
                    "pressure_level": tree["pressure_level"].to_dataset()[
                        [pressure_var]
                    ],
                }
            ),
            dims=(
                "lead_time",
                "ensemble_member",
                "pressure_level",
                "latitude",
                "longitude",
            ),
        )

    monkeypatch.setattr(type(dataset.template_config), "get_template", small_template)

    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2026-08-11T00:00"),
        filter_start=pd.Timestamp("2026-08-10T00:00"),
    )

    store = dataset.store_factory.primary_store()
    root = xr.open_zarr(store, chunks=None)
    levels = xr.open_zarr(store, group="pressure_level", chunks=None)

    init = np.datetime64("2026-08-10T00:00:00")
    assert init in root.init_time.values
    assert np.isfinite(root[root_var].sel(init_time=init).values).any()

    temperature = (
        levels[pressure_var].sel(init_time=init).isel(lead_time=0, ensemble_member=0)
    )
    assert temperature.dims == ("pressure_level", "latitude", "longitude")
    assert np.isfinite(temperature.values).all()

    # Global mean per level, coldest at the 100 hPa tropopause and warming above it.
    np.testing.assert_allclose(
        [float(temperature.sel(pressure_level=level).mean()) for level in LEVELS],
        [
            10.936,
            7.364,
            4.457,
            -2.785,
            -17.292,
            -41.806,
            -55.184,
            -65.717,
            -62.399,
            -49.726,
        ],
        atol=1e-3,
        rtol=0,
    )
