from datetime import timedelta

from reformatters.common import validation
from reformatters.dwd.icon_eu.forecast_5_day_icosahedral import (
    DwdIconEuForecast5DayIcosahedralDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


def test_native_update_stays_suspended_until_backfilled_and_uses_existing_timing() -> (
    None
):
    dataset = DwdIconEuForecast5DayIcosahedralDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG
    )
    jobs = list(dataset.operational_kubernetes_resources("test-image"))
    assert len(jobs) == 1
    update = jobs[0]
    assert update.name == f"{dataset.dataset_id}-update"
    assert update.suspend is True
    assert update.schedule == "52 3,9,15,21 * * *"
    assert update.pod_active_deadline == timedelta(hours=6)
    assert update.workers_total == 2 * dataset.num_variable_groups()
    validators = list(dataset.validators())
    assert isinstance(validators[0], validation.CheckCurrentData)
    assert validators[0].max_delay == timedelta(hours=3, minutes=52)
    assert isinstance(validators[1], validation.CheckRecentNans)
