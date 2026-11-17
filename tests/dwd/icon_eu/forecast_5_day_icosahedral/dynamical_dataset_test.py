from datetime import timedelta

from reformatters.common import validation
from reformatters.dwd.icon_eu.forecast_5_day_icosahedral import (
    DwdIconEuForecast5DayIcosahedralDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


def test_native_update_uses_native_timing_and_stays_suspended_until_backfilled() -> (
    None
):
    dataset = DwdIconEuForecast5DayIcosahedralDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG
    )
    (update,) = dataset.operational_kubernetes_resources("test-image")
    assert update.name == f"{dataset.dataset_id}-update"
    assert update.suspend is True
    assert update.schedule == "17 3,9,15,21 * * *"
    assert update.pod_active_deadline == timedelta(hours=6)
    assert update.active_deadline == timedelta(hours=6)
    assert update.cpu == "3"
    assert update.memory == "14G"
    assert update.shared_memory == "400M"
    assert update.ephemeral_storage == "30G"
    assert update.workers_total == 2 * dataset.num_variable_groups()
    assert update.parallelism == update.workers_total
    assert dataset.validators() == (
        validation.CheckCurrentData(max_delay=timedelta(hours=3, minutes=17)),
        validation.CheckRecentNans(),
    )
