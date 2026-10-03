import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from icechunk import IcechunkStore

from reformatters.common import validation
from reformatters.common.region_job import CoordinateValue
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.common.types import Dim
from reformatters.noaa.refs.forecast_products_virtual.dynamical_dataset import (
    NoaaRefsForecastProductsVirtualDataset,
)
from reformatters.noaa.refs.forecast_products_virtual.region_job import (
    NoaaRefsProductsRegionJob,
    NoaaRefsProductsSourceFileCoord,
)
from reformatters.noaa.refs.forecast_products_virtual.template_config import (
    NoaaRefsForecastProductsVirtualTemplateConfig,
)

SNAPSHOTS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures/value_snapshots.json").read_text()
)
GDAL_RTOL = float(np.finfo(np.float32).eps)


def test_operational_configuration_is_suspended_and_requires_complete_manifests(
    tmp_path: Path,
) -> None:
    dataset = NoaaRefsForecastProductsVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    update, validate = dataset.operational_kubernetes_resources("test")
    assert update.suspend
    assert validate.suspend
    assert update.schedule == "55 1,7,13,19 * * *"
    assert validate.schedule == "5 4,10,16,22 * * *"
    assert dataset.icechunk_virtual_config is not None
    assert (
        dataset.icechunk_virtual_config.containers[0].url_prefix
        == "s3://noaa-rrfs-ops-pds/"
    )
    _, completeness, decode = dataset.validators()
    assert isinstance(completeness, validation.CheckVirtualManifestCompleteness)
    assert completeness.min_present_fraction == (1.0,)
    assert isinstance(decode, validation.CheckVirtualDecodeHealth)
    assert decode.allow_all_nan_vars == ()
    assert decode.max_workers == 2


def assert_numeric_snapshots(ds: xr.Dataset, init: pd.Timestamp) -> None:
    samples = [s for s in SNAPSHOTS["samples"] if s["init_time"] == init.isoformat()]
    assert samples
    assert {s["family"] for s in samples} == {
        "mean",
        "sprd",
        "pmmn",
        "lpmm",
        "avrg",
        "prob",
        "eas",
        "ffri",
    }
    assert set(ds.data_vars) == set(SNAPSHOTS["variables"])
    for sample in samples:
        da = ds[sample["name"]].sel(
            init_time=init, lead_time=pd.Timedelta(hours=sample["lead_hours"])
        )
        if "statistic" in da.dims:
            da = da.sel(statistic="mean" if sample["family"] == "mean" else "spread")
        np.testing.assert_allclose(
            da.isel(y=sample["peak_y"], x=sample["peak_x"]).values,
            sample["peak"],
            rtol=GDAL_RTOL,
            atol=1e-10,
            err_msg=f"{init} {sample['name']} {sample['family']} f{sample['lead_hours']}",
        )
        np.testing.assert_allclose(
            da.isel(y=635, x=1062).values,
            sample["point"],
            rtol=GDAL_RTOL,
            atol=1e-10,
            equal_nan=True,
        )
    assert np.isnan(
        ds.composite_reflectivity.sel(init_time=init, statistic="mean").values
    ).all()
    assert "statistic" not in ds.probability_matched_mean_composite_reflectivity.dims
    assert "pressure_level" not in ds.dims
    assert ds.temperature_500hpa.dims == ds.temperature_2m.dims


@pytest.mark.slow
@pytest.mark.parametrize(
    "with_update", [False, True], ids=["early-backfill", "backfill-update"]
)
def test_real_source_backfill_and_update_numeric_snapshots_and_structural_mean(
    with_update: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init = pd.Timestamp("2026-09-15T12" if with_update else "2026-08-13T00")
    dataset = NoaaRefsForecastProductsVirtualDataset(
        template_config=NoaaRefsForecastProductsVirtualTemplateConfig(
            append_dim_start=init - pd.Timedelta("6h")
        ),
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path / "store"), format=DatasetFormat.ICECHUNK
        ),
    )
    config = dataset.template_config
    variables = SNAPSHOTS["variables"]
    selected = [v for v in config.data_vars if v.name in variables]
    original_template = config.get_template
    monkeypatch.setattr(type(config), "data_vars", property(lambda self: selected))
    monkeypatch.setattr(
        type(config),
        "get_template",
        lambda self, end_time: (
            original_template(end_time)
            .sel(lead_time=pd.to_timedelta(SNAPSHOTS["leads"], unit="h"))
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
    store = dataset.store_factory.primary_store()
    initial = validation.open_flattened_dataset(store, consolidated=False)
    assert pd.Timestamp(initial.init_time.values[-1]) == init
    assert_numeric_snapshots(initial, init)
    initial.close()
    if with_update:
        next_init = init + pd.Timedelta("6h")
        offset, duration = dataset._operational_timing
        fire = next_init + pd.Timedelta(minutes=offset)
        now = (
            fire + pd.Timedelta(minutes=duration) - dataset.virtual_poll_deadline_grace
        )
        monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *a, **k: now))
        monkeypatch.setattr(
            NoaaRefsProductsRegionJob,
            "operational_update_window",
            fire - init + pd.Timedelta("1min"),
        )
        original_discover = NoaaRefsProductsRegionJob.discover_available

        def published_interval(
            job: NoaaRefsProductsRegionJob,
            pending: list[NoaaRefsProductsSourceFileCoord],
        ) -> list[tuple[NoaaRefsProductsSourceFileCoord, int]]:
            return original_discover(
                job,
                [coord for coord in pending if init <= coord.init_time <= next_init],
            )

        monkeypatch.setattr(
            NoaaRefsProductsRegionJob, "discover_available", published_interval
        )
        dataset.update("isolated-products-update")
        store = dataset.store_factory.primary_store()
        updated = validation.open_flattened_dataset(store, consolidated=False)
        assert pd.Timestamp(updated.init_time.values[-1]) == next_init
        assert_numeric_snapshots(updated, next_init)
        assert_numeric_snapshots(updated, init)
    else:
        updated = validation.open_flattened_dataset(store, consolidated=False)
    window = updated.sel(init_time=slice(init, None))
    job = NoaaRefsProductsRegionJob(
        tmp_store=tmp_path / "validation",
        template_ds=xr.DataTree(updated),
        data_vars=selected,
        append_dim="init_time",
        region=slice(1, updated.sizes["init_time"]),
        reformat_job_name="isolated-products-validate",
        processing_mode="backfill",
    )
    assert isinstance(store, IcechunkStore)
    group = zarr.open_group(store, mode="r")
    spread_only = next(v for v in selected if v.name == "composite_reflectivity")
    for checked_init in pd.to_datetime(window.init_time.values):
        for statistic, present in (("mean", False), ("spread", True)):
            loc: dict[Dim, CoordinateValue] = {
                "init_time": checked_init,
                "lead_time": pd.Timedelta("6h"),
                "statistic": statistic,
            }
            assert (
                validation.CheckVirtualDecodeHealth._sampled_refs_present(
                    window[spread_only.name].sel(loc),
                    loc,
                    spread_only,
                    job,
                    store,
                    group,
                )
                == present
            )
    context = validation.ValidationContext(
        store=store,
        ds=window,
        append_dim="init_time",
        data_vars=selected,
        region_job=job,
        append_dim_frequency=pd.Timedelta("6h"),
    )
    complete = validation.CheckVirtualManifestCompleteness().check(context)
    assert complete.passed, complete.message
    healthy = validation.CheckVirtualDecodeHealth(sampled_leads=6, max_workers=1).check(
        context
    )
    assert healthy.passed, healthy.message
    updated.close()
