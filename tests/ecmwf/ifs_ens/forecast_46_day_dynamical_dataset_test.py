from unittest.mock import Mock, patch

import pandas as pd
import pytest

from reformatters.common import validation
from reformatters.ecmwf.ifs_ens.forecast_46_day_1_5_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast46Day15DegreeDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG

# The lag of the slowest publication measured on the ECDS catalogue, 2026-06-26 to
# 2026-08-11. An initialization younger than this is not reliably fetchable.
MEASURED_PUBLICATION_LAG = pd.Timedelta("52.1h")


@pytest.fixture
def dataset() -> EcmwfIfsEnsForecast46Day15DegreeDataset:
    return EcmwfIfsEnsForecast46Day15DegreeDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG
    )


def archived_init_times(
    dataset: EcmwfIfsEnsForecast46Day15DegreeDataset, init_times_back: int
) -> list[pd.Timestamp]:
    archive = Mock()
    with (
        patch(
            "reformatters.ecmwf.ifs_ens.forecast_46_day_dynamical_dataset.archive_initialization",
            archive,
        ),
        patch(
            "reformatters.ecmwf.ifs_ens.forecast_46_day_dynamical_dataset.kubernetes.load_secret",
            return_value={},
        ),
    ):
        dataset.archive_grib_files(
            reformat_job_name="test", init_times_back=init_times_back
        )
    return [call.args[0] for call in archive.call_args_list]


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


def test_initializations_are_archived_newest_first(
    dataset: EcmwfIfsEnsForecast46Day15DegreeDataset,
) -> None:
    init_times = archived_init_times(dataset, init_times_back=4)

    assert len(init_times) == 4
    assert init_times == sorted(init_times, reverse=True)
    assert init_times[0] - init_times[1] == dataset.template_config.append_dim_frequency
    assert all(init_time.tz is None for init_time in init_times)


def test_the_newest_initialization_checked_is_published_by_cron_time(
    dataset: EcmwfIfsEnsForecast46Day15DegreeDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (schedule_hour,) = {
        int(job.schedule.split()[1])
        for job in dataset.operational_kubernetes_resources("test-image")
        if job.name.endswith("archive-grib-files")
    }
    cron_time = pd.Timestamp("2026-08-14", tz="UTC") + pd.Timedelta(hours=schedule_hour)
    monkeypatch.setattr("pandas.Timestamp.now", lambda tz=None: cron_time)

    newest_init_time = archived_init_times(dataset, init_times_back=1)[0]

    age = cron_time - newest_init_time.tz_localize("UTC")
    assert MEASURED_PUBLICATION_LAG < age < pd.Timedelta("72h")
