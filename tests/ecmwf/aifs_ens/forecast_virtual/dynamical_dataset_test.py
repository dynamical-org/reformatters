from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from reformatters.common import validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.ecmwf.aifs_ens.forecast_virtual import (
    EcmwfAifsEnsForecastVirtualDataset,
)
from reformatters.ecmwf.aifs_ens.forecast_virtual.region_job import (
    EcmwfAifsEnsForecastVirtualRegionJob,
)
from reformatters.ecmwf.aifs_ens.forecast_virtual.template_config import (
    EcmwfAifsEnsVirtualDataVar,
)

VARIABLES = (
    "temperature_2m",
    "temperature",
    "geopotential_height_surface",
    "total_precipitation_run_total_surface",
)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("init", "root_values", "pressure_values", "update_value"),
    [
        (
            "2025-07-02",
            [24.583047485351585, 22.943048095703148],
            [
                [-3.8390869140624773, 17.487649536132835],
                [-6.04656066894529, 16.595849609375023],
            ],
            24.390969848632835,
        ),
        (
            "2026-11-11",
            [25.70087585449221, 23.352258300781273],
            [
                [-4.323156738281227, 20.909600830078148],
                [-5.272543334960915, 19.574090576171898],
            ],
            27.951409912109398,
        ),
    ],
)
def test_backfill_and_update_decode_each_era_and_perturbed_statics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    init: str,
    root_values: list[float],
    pressure_values: list[list[float]],
    update_value: float,
) -> None:
    dataset = EcmwfAifsEnsForecastVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    original = dataset.template_config.get_template
    start = pd.Timestamp(init)
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: original(
            min(pd.Timestamp(end_time), start + pd.Timedelta("12h"))
        ).sel(
            init_time=slice(start - pd.Timedelta("6h"), None),
            lead_time=pd.to_timedelta([0, 6], unit="h"),
            ensemble_member=[0, 1],
            pressure_level=[10, 500, 850],
        ),
    )
    dataset.backfill_local(
        append_dim_end=start + pd.Timedelta("1h"),
        filter_start=start,
        filter_variable_names=list(VARIABLES),
    )
    ds = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    cell = ds.sel(init_time=start, latitude=-5, longitude=-60)
    perturbed = cell.sel(ensemble_member=1)
    np.testing.assert_allclose(perturbed.temperature_2m.values, root_values)
    np.testing.assert_allclose(
        perturbed["pressure_level/temperature"].sel(pressure_level=[500, 850]).values,
        pressure_values,
    )
    np.testing.assert_allclose(
        cell.geopotential_height_surface.isel(lead_time=0).values,
        [41.25282353842801, 41.25282353842801],
    )
    assert np.isnan(cell.geopotential_height_surface.isel(lead_time=1).values).all()
    np.testing.assert_array_equal(
        cell.total_precipitation_run_total_surface.isel(lead_time=0).values, [0, 0]
    )
    ten_hpa = perturbed["pressure_level/temperature"].sel(pressure_level=10).values
    if init == "2025-07-02":
        assert np.isnan(ten_hpa).all()
    else:
        np.testing.assert_allclose(ten_hpa, [-46.8884338378906, -43.379049682617165])
    assert np.isfinite(cell.temperature_2m.sel(ensemble_member=0).values).all()

    reference = start + pd.Timedelta("11h5min")

    def now(cls: type[pd.Timestamp], tz: str | None = None) -> pd.Timestamp:
        return reference if tz is None else reference.tz_localize(tz)

    monkeypatch.setattr(pd.Timestamp, "now", classmethod(now))
    original_jobs = (
        EcmwfAifsEnsForecastVirtualRegionJob.operational_update_jobs.__func__
    )

    def update_jobs(
        cls: type[EcmwfAifsEnsForecastVirtualRegionJob],
        *,
        all_data_vars: Sequence[EcmwfAifsEnsVirtualDataVar],
        **kwargs: Any,  # noqa: ANN401 - forwarded to the wrapped factory
    ) -> object:
        return original_jobs(
            cls,
            all_data_vars=[v for v in all_data_vars if v.name in VARIABLES],
            **kwargs,
        )

    monkeypatch.setattr(
        EcmwfAifsEnsForecastVirtualRegionJob,
        "operational_update_jobs",
        classmethod(update_jobs),
    )
    monkeypatch.setattr(
        EcmwfAifsEnsForecastVirtualRegionJob,
        "operational_update_window",
        pd.Timedelta("6h"),
    )
    dataset.update("test-update")
    updated = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    assert updated.init_time.values[-1] == (start + pd.Timedelta("6h")).to_datetime64()
    value = updated.temperature_2m.sel(
        init_time=start + pd.Timedelta("6h"),
        lead_time=pd.Timedelta("6h"),
        ensemble_member=1,
        latitude=-5,
        longitude=-60,
    ).item()
    assert value == pytest.approx(update_value)


def test_deployment_stays_suspended_and_polling_overlaps_next_fire(
    tmp_path: Path,
) -> None:
    dataset = EcmwfAifsEnsForecastVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    (cron,) = dataset.operational_kubernetes_resources("test-image")
    spec = cron.as_kubernetes_object()["spec"]
    assert spec["suspend"] is True
    assert spec["schedule"] == "5 5,11,17,23 * * *"
    assert spec["concurrencyPolicy"] == "Allow"
    job = spec["jobTemplate"]["spec"]
    assert job["activeDeadlineSeconds"] == 21960
    assert job["parallelism"] == 1
    pod = job["template"]["spec"]
    assert pod["activeDeadlineSeconds"] == 21960
    container = pod["containers"][0]
    assert container["resources"]["requests"]["cpu"] == "1.7"
    assert container["resources"]["requests"]["memory"] == "7G"
    for hour in (0, 6, 12, 18):
        fire = pd.Timestamp("2026-11-11") + pd.Timedelta(hours=hour, minutes=305)
        cutoff = dataset._virtual_poll_deadline(fire + pd.Timedelta("1min"))
        assert cron.previous_fire_time(fire) == fire
        assert cutoff == cron.next_fire_time(fire) + pd.Timedelta("1min")
    current, completeness, decode = dataset.validators()
    assert isinstance(current, validation.CheckCurrentData)
    assert current.max_delay == timedelta(hours=8, minutes=13)
    assert isinstance(completeness, validation.CheckVirtualManifestCompleteness)
    assert completeness.min_present_fraction == (1.0,)
    assert isinstance(decode, validation.CheckVirtualDecodeHealth)
    assert (decode.sampled_leads, decode.sampled_levels) == (3, 2)
    assert [
        container.url_prefix for container in dataset.icechunk_virtual_config.containers
    ] == ["gs://ecmwf-open-data/"]
