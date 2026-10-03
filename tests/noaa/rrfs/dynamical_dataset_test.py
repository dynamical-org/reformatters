import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.common.logging import get_logger
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.noaa.refs.forecast_virtual.dynamical_dataset import (
    NoaaRefsForecastVirtualDataset,
)
from reformatters.noaa.rrfs.forecast_18_hour_virtual.dynamical_dataset import (
    NoaaRrfsForecast18HourVirtualDataset,
)
from reformatters.noaa.rrfs.forecast_84_hour_virtual.dynamical_dataset import (
    NoaaRrfsForecast84HourVirtualDataset,
)
from reformatters.noaa.rrfs.forecast_sub_hourly_virtual.dynamical_dataset import (
    NoaaRrfsForecastSubHourlyVirtualDataset,
)
from reformatters.noaa.rrfs.region_job import NoaaRrfsRegionJob, NoaaRrfsSourceFileCoord

type DatasetClass = type[
    NoaaRrfsForecast84HourVirtualDataset
    | NoaaRrfsForecast18HourVirtualDataset
    | NoaaRrfsForecastSubHourlyVirtualDataset
    | NoaaRefsForecastVirtualDataset
]

SNAPSHOTS: dict[str, dict[str, dict[str, object]]] = json.loads(
    (Path(__file__).parent / "fixtures/value_snapshots.json").read_text()
)

CLASSES = (
    NoaaRrfsForecast84HourVirtualDataset,
    NoaaRrfsForecast18HourVirtualDataset,
    NoaaRrfsForecastSubHourlyVirtualDataset,
    NoaaRefsForecastVirtualDataset,
)
log = get_logger(__name__)
POINT = {"y": 635, "x": 1062}


@pytest.mark.parametrize("cls", CLASSES)
def test_operational_jobs_are_suspended_and_use_the_source_container(
    cls: DatasetClass, tmp_path: Path
) -> None:
    dataset = cls(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    jobs = dataset.operational_kubernetes_resources("test")
    assert len(jobs) == 2
    assert all(job.suspend for job in jobs)
    assert all(job.dataset_id == dataset.dataset_id for job in jobs)
    assert all(job.as_kubernetes_object()["spec"]["suspend"] is True for job in jobs)
    assert dataset.icechunk_virtual_config is not None
    assert (
        dataset.icechunk_virtual_config.containers[0].url_prefix
        == "s3://noaa-rrfs-ops-pds/"
    )


@pytest.mark.slow
@pytest.mark.parametrize("cls", CLASSES)
def test_real_source_backfill_and_update_in_isolated_store(
    cls: DatasetClass, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = cls(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path / "store"), format=DatasetFormat.ICECHUNK
        )
    )
    config = dataset.template_config
    init = pd.Timestamp(
        "2026-09-15T12:00"
        if cls in (NoaaRrfsForecast84HourVirtualDataset, NoaaRefsForecastVirtualDataset)
        else "2026-09-15T00:00"
    )
    subhourly = getattr(config, "sub_hourly", False)
    members = getattr(config, "members", False)
    variables = [
        "temperature_2m",
        "wind_u_10m",
        "total_precipitation_run_total_surface",
    ]
    if not subhourly:
        variables += (
            ["temperature", "geopotential_height_cloud_ceiling"]
            if not members
            else ["temperature", "cloud_ceiling_height"]
        )
    if not subhourly and not members:
        variables += [
            "soil_temperature",
            "mass_density_8m_particulate_organic_matter_dry_below_2p5um",
            "categorical_precipitation_exceeding_2_year_average_recurrence_interval_run_total_surface",
        ]
    original_template = config.get_template
    selected_variables = [v for v in config.data_vars if v.name in variables]
    monkeypatch.setattr(
        type(config), "data_vars", property(lambda self: selected_variables)
    )
    selected_leads = [0, 1, 2, 3, 4, 5, 6, 7] if subhourly else [0, 1, 2, 6]
    monkeypatch.setattr(
        type(config),
        "get_template",
        lambda self, end_time: (
            original_template(end_time)
            .isel(lead_time=selected_leads)
            .map_over_datasets(
                lambda ds: ds.drop_vars(
                    [name for name in ds.data_vars if name not in variables]
                )
            )
        ),
    )
    dataset.backfill_local(
        append_dim_end=init + pd.Timedelta("1min"),
        filter_start=init,
        filter_variable_names=variables,
    )
    initial = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    assert pd.Timestamp(initial.init_time.values[-1]) == init
    assert_snapshot(initial, dataset.dataset_id, init)
    initial.close()

    next_init = init + config.append_dim_frequency
    offset, duration = dataset._operational_timing
    fire = next_init + pd.Timedelta(minutes=offset)
    now = fire + pd.Timedelta(minutes=duration) - dataset.virtual_poll_deadline_grace
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *a, **k: now),
    )
    monkeypatch.setattr(
        NoaaRrfsRegionJob,
        "operational_update_window",
        fire - init + pd.Timedelta("1min"),
    )
    original_discover = dataset.region_job_class.discover_available

    def published_interval(
        job: NoaaRrfsRegionJob, pending: list[NoaaRrfsSourceFileCoord]
    ) -> list[tuple[NoaaRrfsSourceFileCoord, int]]:
        return original_discover(
            job, [coord for coord in pending if init <= coord.init_time <= next_init]
        )

    monkeypatch.setattr(
        dataset.region_job_class, "discover_available", published_interval
    )
    assert dataset._virtual_poll_deadline(now) == now
    dataset.update("isolated-test-update")
    updated = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    assert pd.Timestamp(updated.init_time.values[-1]) == next_init
    assert_snapshot(updated, dataset.dataset_id, next_init)
    assert_snapshot(updated, dataset.dataset_id, init)
    updated.close()


def assert_snapshot(ds: xr.Dataset, dataset_id: str, init: pd.Timestamp) -> None:
    expected = SNAPSHOTS[dataset_id][init.isoformat()]
    assert set(ds.data_vars) == set(expected)
    for name, value in expected.items():
        soil = name.startswith("depth_below_ground/")
        da = (
            ds[name]
            .sel(init_time=init)
            .isel(y=500 if soil else 635, x=900 if soil else 1062)
        )
        if "pressure_level" in da.dims:
            da = da.sel(pressure_level=500)
        if "height_above_mean_sea_level" in da.dims:
            da = da.sel(height_above_mean_sea_level=305)
        if "categorical_precipitation_exceeding" in name:
            assert np.isnan(da.sel(lead_time=pd.Timedelta("2h")).values).all()
            da = da.sel(lead_time=pd.Timedelta("6h"))
        elif "sub-hourly" not in dataset_id:
            da = da.sel(lead_time=pd.Timedelta("2h"))
        observed = np.asarray(da.values, dtype=float)
        log.info(
            f"RAW SNAPSHOT {dataset_id} {init.isoformat()} {name} {observed.tolist()!r}"
        )
        np.testing.assert_allclose(
            observed,
            np.asarray(value, dtype=float),
            rtol=1e-12,
            atol=1e-20 if "mass_density" in name else 1e-12,
            equal_nan=True,
            err_msg=f"{dataset_id} {init} {name}",
        )
        if "sub-hourly" in dataset_id and "precipitation" in name:
            assert np.all(np.diff(observed[:4]) >= 0)
        if "ensemble_member" in da.dims:
            assert da.ensemble_member.values.tolist() == list(range(6))
            if name == "temperature_2m":
                assert np.unique(observed).size > 1


@pytest.mark.parametrize(
    ("cls", "offset", "duration"),
    [
        (NoaaRrfsForecast84HourVirtualDataset, 100, 135),
        (NoaaRrfsForecast18HourVirtualDataset, 100, 60),
        (NoaaRrfsForecastSubHourlyVirtualDataset, 75, 55),
        (NoaaRefsForecastVirtualDataset, 75, 160),
    ],
)
def test_poll_deadline_validation_schedule_and_decode_health(
    cls: DatasetClass, offset: int, duration: int, tmp_path: Path
) -> None:
    dataset = cls(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    update, validate = dataset.operational_kubernetes_resources("test")
    init = pd.Timestamp("2026-10-02T00")
    fire = init + pd.Timedelta(minutes=offset)
    end = fire + pd.Timedelta(minutes=duration)
    assert update.previous_fire_time(fire) == fire
    assert update.pod_active_deadline == pd.Timedelta(minutes=duration)
    assert dataset._virtual_poll_deadline(fire) == end - pd.Timedelta("5min")
    assert dataset._virtual_poll_deadline(
        fire + pd.Timedelta("10min")
    ) == end - pd.Timedelta("5min")
    assert validate.previous_fire_time(
        end + pd.Timedelta("5min")
    ) == end + pd.Timedelta("5min")
    current, completeness, decode = dataset.validators()
    assert isinstance(current, validation.CheckCurrentData)
    assert current.max_delay == pd.Timedelta(minutes=offset + duration + 5)
    assert isinstance(completeness, validation.CheckVirtualManifestCompleteness)
    assert completeness.min_present_fraction == (
        (0.05, 1.0) if dataset.template_config.sub_hourly else (1.0,)
    )
    assert isinstance(decode, validation.CheckVirtualDecodeHealth)
    assert decode.max_workers == 2
    assert decode.allow_all_nan_vars == ()
