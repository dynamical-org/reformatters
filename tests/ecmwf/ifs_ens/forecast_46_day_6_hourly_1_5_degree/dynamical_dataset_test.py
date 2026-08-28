import pytest

from reformatters.common import validation
from reformatters.ecmwf.archive_gribs.forecast_46_day_archiver import (
    MATERIALIZED_PRODUCT_ECDS_VARIABLES,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_6_hourly_1_5_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_region_job import (
    EcmwfIfsEns46DayRegionJob,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset:
    return EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG
    )


def test_dataset_reuses_the_shared_region_job(
    dataset: EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset,
) -> None:
    assert dataset.region_job_class is EcmwfIfsEns46DayRegionJob


def test_dataset_variables_are_archived_at_native_resolution(
    dataset: EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset,
) -> None:
    assert {
        data_var.internal_attrs.ecds_variable
        for data_var in dataset.template_config.data_vars
    } == set(MATERIALIZED_PRODUCT_ECDS_VARIABLES["6-hourly"])


def test_operational_cron_jobs_are_suspended_and_do_not_collide_with_daily(
    dataset: EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset,
) -> None:
    update, validate = dataset.operational_kubernetes_resources("test-image-tag")

    assert update.name == "ecmwf-ifs-ens-46-day-6-hourly-update"
    assert validate.name == "ecmwf-ifs-ens-46-day-6-hourly-validate"
    assert update.schedule == "0 10 * * *"
    assert validate.schedule == "0 13 * * *"
    assert update.suspend is True
    assert validate.suspend is True


def test_validators_cover_current_data_and_recent_nans(
    dataset: EcmwfIfsEnsForecast46Day6Hourly15DegreeDataset,
) -> None:
    current, recent_nans = dataset.validators()

    assert isinstance(current, validation.CheckCurrentData)
    assert isinstance(recent_nans, validation.CheckRecentNans)
