from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.ecmwf.aifs_ens.forecast.dynamical_dataset import (
    EcmwfAifsEnsForecastDataset,
)
from tests.chunk_utils import shrink_chunks_and_shards
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> EcmwfAifsEnsForecastDataset:
    return EcmwfAifsEnsForecastDataset(primary_storage_config=NOOP_STORAGE_CONFIG)


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    monkeypatch: pytest.MonkeyPatch, dataset: EcmwfAifsEnsForecastDataset
) -> None:
    variables_to_check = ["temperature_2m", "precipitation_surface"]
    monkeypatch.setattr(
        type(dataset.template_config),
        "data_vars",
        [
            var
            for var in dataset.template_config.data_vars
            if var.name in variables_to_check
        ],
    )

    orig_get_template = dataset.template_config.get_template

    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: shrink_chunks_and_shards(
            orig_get_template(end_time),
            {
                "lead_time": (4, 4),
                "ensemble_member": (2, 2),
                "latitude": (361, 361),
                "longitude": (720, 1440),
            },
        ).sel(
            lead_time=slice("0h", "6h"),
            ensemble_member=slice(0, 1),  # control (0) + first perturbed (1)
        )[variables_to_check],
    )
    dataset.backfill_local(append_dim_end=pd.Timestamp("2025-07-02T06:00:00"))

    ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)
    np.testing.assert_array_equal(
        ds.init_time.values, [np.datetime64("2025-07-02T00:00:00")]
    )

    point_ds = ds.sel(init_time="2025-07-02T00:00:00", latitude=0, longitude=0)

    # Snapshot values at (init_time=2025-07-02T00, lat=0, lon=0).
    # Shape is (lead_time=[0h, 6h], ensemble_member=[0 (cf), 1 (pf)]).
    t2m_backfill_expected = np.array(
        [
            [23.375, 23.375],  # lead=0h, members=[0, 1]
            [23.0, 23.125],  # lead=6h, members=[0, 1]
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(point_ds.temperature_2m.values, t2m_backfill_expected)

    precip_backfill_expected = np.array(
        [
            [np.nan, np.nan],  # lead=0h: deaccumulated, no previous step
            [4.9546361e-07, 2.8461218e-06],  # lead=6h, members=[0, 1]
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        point_ds.precipitation_surface.values,
        precip_backfill_expected,
        rtol=1e-6,
    )

    # Operational update
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        Mock(return_value=pd.Timestamp("2025-07-02T12:00:00")),
    )
    dataset.update("test-update")

    updated_ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)
    np.testing.assert_array_equal(
        updated_ds.init_time,
        np.array(
            [
                np.datetime64("2025-07-02T00:00:00"),
                np.datetime64("2025-07-02T06:00:00"),
            ]
        ),
    )

    updated_point = updated_ds.sel(latitude=0, longitude=0)
    # Shape is (init_time, lead_time, ensemble_member). Backfill values for
    # init=00z must be unchanged after the update.
    t2m_expected = np.array(
        [
            # init=2025-07-02T00 (unchanged from backfill)
            [
                [23.375, 23.375],  # lead=0h, members=[0, 1]
                [23.0, 23.125],  # lead=6h, members=[0, 1]
            ],
            # init=2025-07-02T06 (newly added by update)
            [
                [23.25, 23.25],  # lead=0h, members=[0, 1]
                [23.875, 23.75],  # lead=6h, members=[0, 1]
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(updated_point.temperature_2m.values, t2m_expected)

    precip_expected = np.array(
        [
            [
                [np.nan, np.nan],
                [4.9546361e-07, 2.8461218e-06],
            ],
            [
                [np.nan, np.nan],
                [0.0, 1.5795231e-06],
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        updated_point.precipitation_surface.values,
        precip_expected,
        rtol=1e-6,
    )


def test_operational_kubernetes_resources(
    dataset: EcmwfAifsEnsForecastDataset,
) -> None:
    cron_jobs = dataset.operational_kubernetes_resources("test-image-tag")

    assert len(cron_jobs) == 2
    update_cron_job, validation_cron_job = cron_jobs
    assert update_cron_job.name == f"{dataset.dataset_id}-update"
    assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
    assert update_cron_job.secret_names == [
        dataset.primary_storage_config.k8s_secret_name
    ]
    assert validation_cron_job.secret_names == [
        dataset.primary_storage_config.k8s_secret_name
    ]


def test_validators(dataset: EcmwfAifsEnsForecastDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.DataValidator) for v in validators)
