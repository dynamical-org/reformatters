from reformatters.common.kubernetes import ReformatCronJob
from reformatters.dwd.icon_eu.forecast_5_day import DwdIconEuForecast5DayDataset
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
    regular = DwdIconEuForecast5DayDataset(primary_storage_config=NOOP_STORAGE_CONFIG)
    regular_update = next(
        job
        for job in regular.operational_kubernetes_resources("test-image")
        if isinstance(job, ReformatCronJob)
    )
    jobs = list(dataset.operational_kubernetes_resources("test-image"))
    assert len(jobs) == 1
    update = jobs[0]
    assert update.name == f"{dataset.dataset_id}-update"
    assert update.suspend is True
    assert update.schedule == regular_update.schedule
    assert update.pod_active_deadline == regular_update.pod_active_deadline
    assert update.active_deadline == regular_update.active_deadline
    assert update.cpu == regular_update.cpu
    assert update.memory == regular_update.memory
    assert update.shared_memory == regular_update.shared_memory
    assert update.ephemeral_storage == regular_update.ephemeral_storage
    assert update.workers_total == regular_update.workers_total
    assert update.parallelism == regular_update.parallelism
    assert dataset.validators() == regular.validators()
