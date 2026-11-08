import numpy as np
import pandas as pd
import pytest

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common.config_models import ROOT
from reformatters.ecmwf.ifs_ens.virtual_dynamical_dataset import (
    EcmwfIfsEnsForecastVirtualDataset,
)

DATASETS = [
    d for d in DYNAMICAL_DATASETS if isinstance(d, EcmwfIfsEnsForecastVirtualDataset)
]


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda d: d.dataset_id)
def test_product_geometry(dataset: EcmwfIfsEnsForecastVirtualDataset) -> None:
    config = dataset.template_config
    coords = config.dimension_coordinates()
    fine = config.resolution == "0p1"
    assert len(coords["lead_time"]) == (49 if config.forecast_days == 6 else 85)
    assert config.append_dim_frequency == pd.Timedelta(
        hours=6 if config.forecast_days == 6 else 12
    )
    np.testing.assert_array_equal(coords["ensemble_member"], np.arange(51))
    assert coords["longitude"][1200 if fine else 480] == -60
    assert coords["latitude"][950 if fine else 380] == -5
    assert coords["latitude"][0] == 90
    assert coords["latitude"][-1] == -90
    assert len(coords["latitude"]) == (1801 if fine else 721)
    np.testing.assert_allclose(coords["longitude"][-1], 179.9 if fine else 179.75)
    assert config.dims[ROOT] == (
        "init_time",
        "lead_time",
        "ensemble_member",
        "latitude",
        "longitude",
    )
    assert config.dims["pressure_level"] == (*config.dims[ROOT], "pressure_level")
    for var in config.data_vars:
        assert var.encoding.chunks == (
            (1, 1, 1, len(coords["latitude"]), len(coords["longitude"]))
            + (() if var.group is ROOT else (1,))
        )
        assert np.isnan(var.encoding.fill_value)
        if var.internal_attrs.control_only:
            assert "control member 0 only" in (var.attrs.comment or "")
        if var.internal_attrs.lead_0_only:
            assert "lead time 0 only" in (var.attrs.comment or "")
        if var.attrs.step_type in {"min", "max"}:
            assert "Not available at lead time 0" in (var.attrs.comment or "")
        if var.attrs.step_type == "accum":
            assert "since forecast initialization" in (var.attrs.comment or "")
    cape = next(
        (v for v in config.data_vars if v.internal_attrs.grib_index_param == "cape"),
        None,
    )
    assert (cape is None) == fine


def test_native_water_accumulation_units() -> None:
    for var in DATASETS[0].template_config.data_vars:
        if var.internal_attrs.grib_index_param in {"tp", "cp", "sf", "ro"}:
            assert var.attrs.units == "m"
            assert not var.encoding.filters


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda d: d.dataset_id)
def test_operational_configuration(dataset: EcmwfIfsEnsForecastVirtualDataset) -> None:
    (cron,) = dataset.operational_kubernetes_resources("test-image")
    assert cron.suspend
    assert cron.concurrency_policy == "Allow"
    assert cron.workers_total == cron.parallelism == 1
    fine = dataset.template_config.resolution == "0p1"
    prefixes = [c.url_prefix for c in dataset.icechunk_virtual_config.containers]
    assert prefixes == (
        ["gs://ecmwf-open-data/"]
        if fine
        else ["gs://ecmwf-open-data/", "s3://ecmwf-forecasts/"]
    )
    assert len(dataset.validators()) == 3
    for hour in (
        (0, 6, 12, 18) if dataset.template_config.forecast_days == 6 else (0, 12)
    ):
        init = pd.Timestamp("2026-11-06") + pd.Timedelta(hours=hour)
        offset = (
            (pd.Timedelta("8h12min") if fine else pd.Timedelta("6h12min"))
            if dataset.template_config.forecast_days == 6
            else (pd.Timedelta("9h19min") if fine else pd.Timedelta("7h19min"))
        )
        fire = init + offset
        assert cron.previous_fire_time(fire) == fire
        cutoff = dataset._virtual_poll_deadline(fire)
        assert cutoff == cron.next_fire_time(fire) + pd.Timedelta("1min")
        earliest = pd.Timedelta("8h27min") if fine else pd.Timedelta("6h27min")
        if dataset.template_config.forecast_days == 15:
            earliest = pd.Timedelta("9h34min") if fine else pd.Timedelta("7h34min")
        assert fire < init + earliest
        assert cutoff < init + earliest + dataset.template_config.append_dim_frequency
