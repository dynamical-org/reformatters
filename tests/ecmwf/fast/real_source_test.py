import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import download, kubernetes, storage
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.ecmwf.aifs_ens.forecast_fast import EcmwfAifsEnsForecastFastDataset
from reformatters.ecmwf.fast import source
from reformatters.ecmwf.fast.region_job import EcmwfFastRegionJob
from reformatters.ecmwf.fast.template_config import FastModel
from reformatters.ecmwf.ifs_ens.forecast_0_25_degree_fast import (
    EcmwfIfsEnsForecast025DegreeFastDataset,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("ECMWF_FAST_REAL_SOURCE_TEST") != "1",
        reason="Set ECMWF_FAST_REAL_SOURCE_TEST=1 with explicit read-only source credentials",
    ),
]

_SNAPSHOTS = {
    "ifs": [
        26.840173339843773,
        0.779876708984375,
        4.6253814697265625,
        0.85186767578125,
        4.96087646484375,
        0.0457763671875,
        11776.0,
    ],
    "aifs": [
        26.739312744140648,
        0.020704269409179688,
        5.718044281005859,
        0.001262664794921875,
        6.246171951293945,
        0.0,
        4608.0,
    ],
}


@pytest.fixture
def authorized_source(monkeypatch: pytest.MonkeyPatch) -> None:
    assert os.environ.get("AWS_ACCESS_KEY_ID"), "Explicit source access key required"
    assert os.environ.get("AWS_SECRET_ACCESS_KEY"), (
        "Explicit source secret key required"
    )
    if os.environ.get("AWS_ENDPOINT_URL") != source.ENDPOINT:
        raise AssertionError(
            "Explicit source endpoint must match the registered container"
        )

    def credentials(secret_name: str) -> dict[str, str]:
        assert secret_name == source.SOURCE_SECRET
        options = {
            "access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
            "secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        }
        if token := os.environ.get("AWS_SESSION_TOKEN"):
            options["session_token"] = token
        return download.validate_s3_credentials(options)

    def forbidden_secret(secret_name: str) -> dict[str, Any]:
        raise AssertionError("Real-source tests must never read Kubernetes Secrets")

    monkeypatch.setattr(download, "load_s3_credentials", credentials)
    monkeypatch.setattr(storage, "load_s3_credentials", credentials)
    monkeypatch.setattr(kubernetes, "load_secret", forbidden_secret)
    monkeypatch.setattr(EcmwfFastRegionJob, "download_concurrency", 2)
    source.listing_client.cache_clear()
    download.s3_store_from_secret.cache_clear()


@pytest.mark.parametrize("model", ["ifs", "aifs"])
def test_real_backfill_update_and_snapshot_isolation(
    model: FastModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authorized_source: None,
) -> None:
    cls = (
        EcmwfIfsEnsForecast025DegreeFastDataset
        if model == "ifs"
        else EcmwfAifsEnsForecastFastDataset
    )
    dataset = cls(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path / "output"), format=DatasetFormat.ICECHUNK
        )
    )
    get_template = dataset.template_config.get_template

    def trimmed_template(self: Any, end_time: Any) -> xr.DataTree:  # noqa: ANN401
        return get_template(end_time).sel(
            init_time=slice("2026-11-04", None),
            lead_time=pd.to_timedelta([0, 6], unit="h"),
            ensemble_member=[0, 1, 50],
        )

    monkeypatch.setattr(type(dataset.template_config), "get_template", trimmed_template)
    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2026-11-04T01:00"),
    )
    repo, _ = dataset.store_factory.icechunk_primary_and_replica_repos()
    backfill_snapshot = repo.readonly_session("main").snapshot_id
    before = xr.open_zarr(
        repo.readonly_session(snapshot_id=backfill_snapshot).store,
        chunks=None,
        decode_timedelta=True,
        consolidated=False,
    )
    assert before.sizes["init_time"] == 1
    cell = before.sel(
        init_time=pd.Timestamp("2026-11-04"),
        lead_time=pd.Timedelta("6h"),
        latitude=0,
        longitude=0,
    )
    names = [v.name for v in dataset.template_config.data_vars[:7]]
    for var in dataset.template_config.data_vars[7:]:
        assert before[var.name].isnull().all()
    np.testing.assert_allclose(
        [float(cell[name].sel(ensemble_member=0).values) for name in names],
        _SNAPSHOTS[model],
        rtol=1e-12,
        atol=1e-12,
    )
    for name in names:
        values = cell[name].values
        assert np.isfinite(values).all()
        assert before[name].encoding["chunks"] == (1, 1, 1, 721, 1440)
    for name in names[-2:]:
        np.testing.assert_array_equal(
            before[name].sel(lead_time=pd.Timedelta(0), latitude=0, longitude=0).values,
            0,
        )
    assert len(np.unique(cell.temperature_2m.values)) == 3
    reference = np.array([cell[name].values for name in names])

    fire = pd.Timestamp("2026-11-04T11:08" if model == "ifs" else "2026-11-04T11:34")
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda cls, *args, **kwargs: fire)
    )
    dataset.update("test-fast-update")
    after_session = repo.readonly_session("main")
    assert after_session.snapshot_id != backfill_snapshot
    after = xr.open_zarr(
        after_session.store, chunks=None, decode_timedelta=True, consolidated=False
    )
    assert after.sizes["init_time"] == 2
    assert after.init_time.values[-1] == np.datetime64("2026-11-04T06:00")
    new_cell = after.sel(
        init_time=pd.Timestamp("2026-11-04T06:00"),
        lead_time=pd.Timedelta("6h"),
        latitude=0,
        longitude=0,
    )
    assert all(np.isfinite(new_cell[name].values).all() for name in names)
    np.testing.assert_array_equal(
        np.array(
            [
                after[name]
                .sel(
                    init_time=pd.Timestamp("2026-11-04"),
                    lead_time=pd.Timedelta("6h"),
                    latitude=0,
                    longitude=0,
                )
                .values
                for name in names
            ]
        ),
        reference,
    )
    historical = xr.open_zarr(
        repo.readonly_session(snapshot_id=backfill_snapshot).store,
        chunks=None,
        decode_timedelta=True,
        consolidated=False,
    )
    assert historical.sizes["init_time"] == 1
    np.testing.assert_array_equal(
        historical.temperature_2m.values, before.temperature_2m.values
    )
