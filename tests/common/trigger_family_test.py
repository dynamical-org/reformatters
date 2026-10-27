from pathlib import Path
from unittest.mock import Mock
from urllib.parse import urlparse

import pandas as pd
import pytest
import xarray as xr

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import kubernetes
from reformatters.noaa.gefs.forecast_35_day.region_job import GefsForecast35DayRegionJob
from reformatters.noaa.gefs.forecast_35_day.template_config import (
    GefsForecast35DayTemplateConfig,
)
from reformatters.noaa.gefs.gefs_config_models import GEFS_PRE_EXTENSION_MAX


def test_trigger_family_mappings_use_only_the_designated_producers() -> None:
    expected = {
        "noaa-hrrr-forecast-48-hour-virtual": "noaa-hrrr-forecast-48-hour-update",
        "noaa-gfs-forecast-virtual": "noaa-gfs-forecast-update",
        "noaa-gefs-forecast-16-day-0-5-degree-virtual": "noaa-gefs-forecast-35-day-update",
        "ecmwf-aifs-single-forecast-virtual": "ecmwf-aifs-single-forecast-update",
    }
    mappings = {
        dataset.dataset_id: tuple(
            trigger.target_cronjob for trigger in dataset.materialized_update_triggers
        )
        for dataset in DYNAMICAL_DATASETS
        if dataset.materialized_update_triggers
    }
    assert mappings == {
        producer: (consumer,) for producer, consumer in expected.items()
    }
    crons = {
        cron.name
        for dataset in DYNAMICAL_DATASETS
        for cron in dataset.operational_kubernetes_resources("test")
    }
    assert set(expected.values()) <= crons


def test_gefs_tier_one_triggers_only_the_daily_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = next(
        dataset
        for dataset in DYNAMICAL_DATASETS
        if dataset.dataset_id == "noaa-gefs-forecast-16-day-0-5-degree-virtual"
    )
    (trigger,) = producer.materialized_update_triggers
    submit = Mock()
    monkeypatch.setattr(kubernetes, "create_job_from_cronjob", submit)
    for hour in (0, 6, 12, 18):
        trigger.dispatch(pd.Timestamp("2026-10-21") + pd.Timedelta(hours=hour))
    submit.assert_called_once_with(
        "noaa-gefs-forecast-35-day-update",
        "noaa-gefs-forecast-35-day-t202610210000z",
        kind="trigger",
    )
    extension = next(
        dataset
        for dataset in DYNAMICAL_DATASETS
        if dataset.dataset_id == "noaa-gefs-forecast-35-day-0-5-degree-virtual"
    )
    assert not extension.materialized_update_triggers


def test_gefs_trigger_covers_the_consumers_actual_tier_one_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init = pd.Timestamp("2026-10-21")
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: init + pd.Timedelta("7h")),
    )
    config = GefsForecast35DayTemplateConfig()
    dimensions = config.dimension_coordinates()
    region = xr.Dataset(
        coords={
            "init_time": [init],
            "lead_time": dimensions["lead_time"],
            "ensemble_member": dimensions["ensemble_member"],
        }
    )
    consumer = GefsForecast35DayRegionJob(
        tmp_store=Path("unused.zarr"),
        template_ds=xr.DataTree(region),
        data_vars=config.data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
    )
    required = [
        coord
        for group in consumer.source_file_var_groups(config.data_vars)
        for coord in consumer.generate_source_file_coords(region, group)
    ]
    assert required
    assert max(coord.lead_time for coord in required) == GEFS_PRE_EXTENSION_MAX
    assert {coord.ensemble_member for coord in required} == set(range(31))
    sources: dict[str, set[str]] = {}
    for dataset_id in (
        "noaa-gefs-forecast-10-day-0-25-degree-virtual",
        "noaa-gefs-forecast-16-day-0-5-degree-virtual",
    ):
        dataset = next(
            dataset
            for dataset in DYNAMICAL_DATASETS
            if dataset.dataset_id == dataset_id
        )
        dimensions = dataset.template_config.dimension_coordinates()
        source_region = xr.Dataset(
            coords={
                "init_time": [init],
                "lead_time": dimensions["lead_time"],
                "ensemble_member": dimensions["ensemble_member"],
            }
        )
        job = dataset.region_job_class(
            tmp_store=Path("unused.zarr"),
            template_ds=xr.DataTree(source_region),
            data_vars=dataset.template_config.data_vars,
            append_dim="init_time",
            region=slice(0, 1),
            reformat_job_name="test",
        )
        sources[dataset_id] = {
            urlparse(coord.get_url()).path
            for coord in job.generate_source_file_coords(source_region, job.data_vars)
        }
    early = {
        urlparse(coord.get_url()).path
        for coord in required
        if coord.gefs_file_type == "s"
    }
    completing = [coord for coord in required if coord.gefs_file_type != "s"]
    assert early <= sources["noaa-gefs-forecast-10-day-0-25-degree-virtual"]
    assert early
    assert completing
    assert {urlparse(coord.get_url()).path for coord in completing} <= sources[
        "noaa-gefs-forecast-16-day-0-5-degree-virtual"
    ]
    assert {
        coord.lead_time
        for coord in completing
        if coord.lead_time > pd.Timedelta("240h")
    } == set(pd.timedelta_range("246h", "384h", freq="6h"))
    assert any(coord.lead_time <= pd.Timedelta("240h") for coord in completing)
    for lead in pd.timedelta_range("246h", "384h", freq="6h"):
        assert {
            coord.ensemble_member for coord in completing if coord.lead_time == lead
        } == set(range(31))
