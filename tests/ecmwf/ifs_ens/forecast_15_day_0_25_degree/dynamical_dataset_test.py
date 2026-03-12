from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# from reformatters.common import validation
from reformatters.common import validation
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day025DegreeDataset,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> EcmwfIfsEnsForecast15Day025DegreeDataset:
    return EcmwfIfsEnsForecast15Day025DegreeDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG
    )


@pytest.mark.slow
def test_backfill_local_and_operational_update(
    monkeypatch: pytest.MonkeyPatch, dataset: EcmwfIfsEnsForecast15Day025DegreeDataset
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
        lambda self, end_time: orig_get_template(end_time).sel(
            lead_time=slice("0h", "6h"), ensemble_member=slice(0, 1)
        )[variables_to_check],
    )
    dataset.backfill_local(append_dim_end=pd.Timestamp("2024-04-02T00:00:00"))

    ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)
    # existing_ds_append_dim_end is exclusive, so should have only processed the first forecast
    np.testing.assert_array_equal(
        ds.init_time.values, [np.datetime64("2024-04-01T00:00:00")]
    )

    t2m_actual_values = ds.sel(
        init_time="2024-04-01T00:00:00", latitude=0, longitude=0
    ).temperature_2m

    t2m_expected_values = np.array(
        [
            [29.5, 29.125],  # lead time 0h, ensemble members 0 and 1
            [28.125, 28.125],  # lead time 3h, ensemble members 0 and 1
            [28.375, 27.75],  # lead time 6h, ensemble members 0 and 1
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(t2m_actual_values, t2m_expected_values)

    precip_surface_actual_values = ds.sel(
        init_time="2024-04-01T00:00:00", latitude=0, longitude=0
    ).precipitation_surface

    precip_surface_expected_values = np.array(
        [
            [np.nan, np.nan],  # lead time 0h, ensemble members 0 and 1
            [
                1.198053e-05,
                1.236796e-06,
            ],  # lead time 3h, ensemble members 0 and 1
            [
                1.415610e-06,
                1.537800e-05,
            ],  # lead time 6h, ensemble members 0 and 1
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        precip_surface_actual_values,
        precip_surface_expected_values,
        atol=1e-6,
    )

    # Operational update
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        Mock(return_value=pd.Timestamp("2024-04-03T00:00:00")),
    )
    dataset.update("test-update")

    updated_ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)

    np.testing.assert_array_equal(
        updated_ds.init_time,
        np.array(
            [np.datetime64("2024-04-01T00:00:00"), np.datetime64("2024-04-02T00:00:00")]
        ),
    )
    t2m_actual_values = updated_ds.sel(latitude=0, longitude=0).temperature_2m
    t2m_expected_values = np.array(
        [  # init time 2024-04-01T00:00:00
            [
                [29.5, 29.125],  # lead time 0h, ensemble members 0 and 1
                [28.125, 28.125],  # lead time 3h, ensemble members 0 and 1
                [28.375, 27.75],  # lead time 6h, ensemble members 0 and 1
            ],
            [  # init time 2024-04-02T00:00:00
                [29.75, 29.125],  # lead time 0h, ensemble members 0 and 1
                [28.5, 27.0],  # lead time 3h, ensemble members 0 and 1
                [27.0, 27.25],  # lead time 6h, ensemble members 0 and 1
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(t2m_actual_values, t2m_expected_values)

    precip_surface_actual_values = updated_ds.sel(
        latitude=0, longitude=0
    ).precipitation_surface

    precip_surface_expected_values = np.array(
        [
            [
                [np.nan, np.nan],
                [1.1980534e-05, 1.2367964e-06],
                [1.4156103e-06, 1.5377998e-05],
            ],
            [
                [np.nan, np.nan],
                [8.7261200e-05, 3.0517578e-04],
                [7.3623657e-04, 2.2029877e-04],
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        precip_surface_actual_values,
        precip_surface_expected_values,
        rtol=1e-6,
    )


def test_operational_kubernetes_resources(
    dataset: EcmwfIfsEnsForecast15Day025DegreeDataset,
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


def test_validators(dataset: EcmwfIfsEnsForecast15Day025DegreeDataset) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.DataValidator) for v in validators)


@pytest.mark.slow
def test_backfill_local_mars_source(
    monkeypatch: pytest.MonkeyPatch, dataset: EcmwfIfsEnsForecast15Day025DegreeDataset
) -> None:
    """Run the full backfill pipeline on MARS-archived data from source.coop.

    This test mirrors test_backfill_local_and_operational_update but uses a pre-2024-04-01
    init time so data is fetched from the MARS staging bucket instead of ECMWF open data.
    """
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

    # Move append_dim_start before the MARS test date (frozen model, so bypass via __dict__)
    object.__setattr__(
        dataset.template_config, "append_dim_start", pd.Timestamp("2016-03-08T00:00")
    )

    orig_get_template = dataset.template_config.get_template

    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: orig_get_template(end_time).sel(
            lead_time=slice("0h", "6h"), ensemble_member=slice(0, 1)
        )[variables_to_check],
    )
    dataset.backfill_local(append_dim_end=pd.Timestamp("2016-03-09T00:00:00"))

    ds = xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)
    np.testing.assert_array_equal(
        ds.init_time.values, [np.datetime64("2016-03-08T00:00:00")]
    )

    # temperature_2m: check shape and finite values at a sample point
    t2m = ds.sel(
        init_time="2016-03-08T00:00:00", latitude=0, longitude=0
    ).temperature_2m
    assert t2m.shape == (3, 2)  # 3 lead times x 2 ensemble members
    assert np.all(np.isfinite(t2m.values))

    # precipitation_surface: NaN at lead_time=0, finite and non-negative at later times
    precip = ds.sel(
        init_time="2016-03-08T00:00:00", latitude=0, longitude=0
    ).precipitation_surface
    assert precip.shape == (3, 2)
    assert np.all(np.isnan(precip.sel(lead_time="0h").values))
    later_precip = precip.sel(lead_time=slice("3h", "6h")).values
    assert np.all(np.isfinite(later_precip))
    assert np.all(later_precip >= 0)
