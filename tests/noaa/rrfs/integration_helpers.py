import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import template_utils, validation
from reformatters.common.logging import get_logger
from reformatters.common.pydantic import replace
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.common.types import DatetimeLike
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
from reformatters.noaa.rrfs_ens.forecast_virtual.dynamical_dataset import (
    NoaaRrfsEnsForecastVirtualDataset,
)

type DatasetClass = type[
    NoaaRrfsForecast84HourVirtualDataset
    | NoaaRrfsForecast18HourVirtualDataset
    | NoaaRrfsForecastSubHourlyVirtualDataset
    | NoaaRrfsEnsForecastVirtualDataset
]

SNAPSHOTS: dict[str, dict[str, dict[str, object]]] = json.loads(
    (Path(__file__).parent / "fixtures/value_snapshots.json").read_text()
)

log = get_logger(__name__)


def backfill_and_update_in_isolated_store(
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
        if cls
        in (NoaaRrfsForecast84HourVirtualDataset, NoaaRrfsEnsForecastVirtualDataset)
        else "2026-09-15T00:00"
    )
    dataset = replace(dataset, template_config=replace(config, append_dim_start=init))
    config = dataset.template_config
    subhourly = config.sub_hourly
    members = config.members
    variables = ["temperature_2m", "total_precipitation_run_total_surface"]
    if members:
        variables += ["temperature", "cloud_ceiling_height"]
    elif not subhourly:
        variables += [
            "soil_temperature",
            "geopotential_height_cloud_ceiling",
            "mass_density_8m_particulate_organic_matter_dry_below_2p5um",
        ]
    selected_variables = [v for v in config.data_vars if v.name in variables]
    monkeypatch.setattr(
        type(config), "data_vars", property(lambda self: selected_variables)
    )
    selected_leads = list(range(8)) if subhourly else [2]

    def select_dataset(ds: xr.Dataset) -> xr.Dataset:
        ds = ds.drop_vars([name for name in ds.data_vars if name not in variables])
        levels = {
            "pressure_level": [500],
            "depth_below_ground": [0.01],
            "ensemble_member": [0, 5],
        }
        return ds.sel({dim: values for dim, values in levels.items() if dim in ds.dims})

    template = (
        config.get_template(init + pd.Timedelta("1min"))
        .isel(lead_time=selected_leads)
        .map_over_datasets(select_dataset)
    )

    template = xr.DataTree.from_dict(
        {
            node.path: node.to_dataset()
            for node in template.subtree
            if node.path == "/" or node.data_vars
        }
    )

    def get_template(self: object, end_time: DatetimeLike) -> xr.DataTree:
        return template.map_over_datasets(
            lambda ds: template_utils.empty_copy_with_reindex(
                ds,
                config.append_dim,
                config.append_dim_coordinates(end_time),
                derive_coordinates_fn=config.derive_coordinates,
            )
        )

    monkeypatch.setattr(type(config), "get_template", get_template)
    dataset.backfill_local(
        append_dim_end=init + pd.Timedelta("1min"),
        filter_variable_names=variables,
    )
    initial = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    assert pd.Timestamp(initial.init_time.values[-1]) == init
    assert_snapshot(initial, dataset.dataset_id, init)
    initial.close()

    next_init = init + config.append_dim_frequency
    fire = next_init + pd.Timedelta(minutes=dataset.update_offset_minutes)
    now = (
        fire
        + pd.Timedelta(minutes=dataset.update_deadline_minutes)
        - dataset.virtual_poll_deadline_grace
    )
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
    expected_paths = (
        {"temperature_2m", "total_precipitation_run_total_surface"}
        if "sub-hourly" in dataset_id
        else {
            "temperature_2m",
            "total_precipitation_run_total_surface",
            "cloud_ceiling_height",
            "pressure_level/temperature",
        }
        if "rrfs-ens" in dataset_id
        else {
            "temperature_2m",
            "total_precipitation_run_total_surface",
            "geopotential_height_cloud_ceiling",
            "depth_below_ground/soil_temperature",
            "mass_density_8m_particulate_organic_matter_dry_below_2p5um",
        }
    )
    assert set(ds.data_vars) == expected_paths, (set(ds.data_vars), expected_paths)
    for name in sorted(expected_paths):
        value = expected[name]
        soil = name.startswith("depth_below_ground/")
        da = (
            ds[name]
            .sel(init_time=init)
            .isel(y=500 if soil else 635, x=900 if soil else 1062)
        )
        if "pressure_level" in da.dims:
            assert da.pressure_level.values.tolist() == [500]
            da = da.sel(pressure_level=500)
        if soil:
            assert da.depth_below_ground.values.tolist() == [0.01]
            da = da.sel(depth_below_ground=0.01)
            value = np.asarray(value)[1]
        if "sub-hourly" in dataset_id:
            da = da.isel(lead_time=[0, 3, 4])
            assert (
                da.lead_time.values.tolist()
                == pd.to_timedelta([15, 60, 75], unit="min").values.tolist()
            )
            value = np.asarray(value)[[0, 3, 4]]
        else:
            da = da.sel(lead_time=pd.Timedelta("2h"))
        if "ensemble_member" in da.dims:
            assert da.ensemble_member.values.tolist() == [0, 5]
            value = np.asarray(value)[[0, 5]]
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
            assert observed[1] >= observed[0]
        if "ensemble_member" in da.dims and name == "temperature_2m":
            assert np.unique(observed).size > 1
