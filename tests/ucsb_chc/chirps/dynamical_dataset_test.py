from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr.storage
from numpy.testing import assert_allclose, assert_array_equal

from reformatters.common import validation
from reformatters.common.pydantic import replace
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.ucsb_chc.chirps.analysis_final import (
    UcsbChcChirpsAnalysisFinalDataset,
)
from reformatters.ucsb_chc.chirps.analysis_preliminary import (
    UcsbChcChirpsAnalysisPreliminaryDataset,
)
from reformatters.ucsb_chc.chirps.dynamical_dataset import (
    UcsbChcChirpsAnalysisMaterializedDataset,
)
from reformatters.ucsb_chc.chirps.template_config import (
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    MM_PER_DAY_TO_KG_M2_S,
    SOURCE_FILL_VALUE,
)
from scripts.trim_chirps import trim_store
from tests.chunk_utils import shrink_chunks_and_shards
from tests.common.dynamical_dataset_test import (
    NOOP_STORAGE_CONFIG,
    assert_configured_validators,
)

# A land point in the western Amazon and an open ocean point in the Pacific.
_LAND_POINT = {"latitude": -1.975, "longitude": -60.025}
_OCEAN_POINT = {"latitude": 0.025, "longitude": -140.025}


def _final_dataset() -> UcsbChcChirpsAnalysisFinalDataset:
    return UcsbChcChirpsAnalysisFinalDataset(primary_storage_config=NOOP_STORAGE_CONFIG)


def _preliminary_dataset() -> UcsbChcChirpsAnalysisPreliminaryDataset:
    return UcsbChcChirpsAnalysisPreliminaryDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG
    )


def _shrink_template(
    monkeypatch: pytest.MonkeyPatch, dataset: UcsbChcChirpsAnalysisMaterializedDataset
) -> None:
    original_get_template = dataset._get_template
    monkeypatch.setattr(
        dataset,
        "_get_template",
        lambda end: shrink_chunks_and_shards(original_get_template(end)),
    )


def _set_time_chunks(template: xr.DataTree, size: int) -> xr.DataTree:
    """Shrink the append dim chunk and shard so a small test spans several shards."""
    for node in template.subtree:
        for var in node.to_dataset().data_vars.values():
            encoding = node.dataset[str(var.name)].encoding
            for key in ("chunks", "shards"):
                encoding[key] = (size, *encoding[key][1:])
    return template


def _open_store(dataset: UcsbChcChirpsAnalysisMaterializedDataset) -> xr.Dataset:
    return xr.open_zarr(dataset.store_factory.primary_store(), chunks=None)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("make_dataset", "first_day", "source_mm_per_day"),
    [
        pytest.param(
            _final_dataset,
            pd.Timestamp("1981-01-01"),
            [0.31350774, 0.6122222, 3.8714871],
            id="final",
        ),
        pytest.param(
            _preliminary_dataset,
            pd.Timestamp("2025-01-01"),
            [5.660802, 22.610807, 0.004789341],
            id="preliminary",
        ),
    ],
)
def test_backfill_local_and_operational_update(
    monkeypatch: pytest.MonkeyPatch,
    make_dataset: Callable[[], UcsbChcChirpsAnalysisMaterializedDataset],
    first_day: pd.Timestamp,
    source_mm_per_day: list[float],
) -> None:
    dataset = make_dataset()
    _shrink_template(monkeypatch, dataset)

    dataset.backfill_local(append_dim_end=first_day + pd.Timedelta(days=2))
    backfilled_ds = _open_store(dataset)
    assert_array_equal(
        backfilled_ds["time"], pd.date_range(first_day, periods=2, freq="1D")
    )

    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: first_day + pd.Timedelta(days=3)),
    )
    dataset.update("test-update")

    updated_ds = _open_store(dataset)
    assert_array_equal(
        updated_ds["time"], pd.date_range(first_day, periods=3, freq="1D")
    )

    land = updated_ds.sel(_LAND_POINT, method="nearest")
    # Source mm/day converted to kg m-2 s-1; rtol covers keep_mantissa_bits=8.
    assert_allclose(
        land["precipitation_surface"].values,
        np.array(source_mm_per_day, dtype=np.float32) * MM_PER_DAY_TO_KG_M2_S,
        rtol=1e-2,
    )
    ocean = updated_ds.sel(_OCEAN_POINT, method="nearest")
    assert np.isnan(ocean["precipitation_surface"].values).all()

    assert_configured_validators(dataset)


def _patch_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    available: Callable[[pd.Timestamp], bool],
    requested_urls: list[str],
) -> None:
    """Serve a fresh constant grid for available days and report all others missing."""

    def fake_download(url: str, dataset_id: str) -> Path:
        requested_urls.append(url)
        day = pd.Timestamp(url.removesuffix(".tif")[-10:].replace(".", "-"))
        if not available(day):
            raise FileNotFoundError(url)
        path = tmp_path / f"{day:%Y%m%d}.tif"
        path.touch()
        return path

    values = np.full((GRID_LAT_SIZE, GRID_LON_SIZE), 24.0, dtype=np.float32)
    values[0, 0] = np.float32(SOURCE_FILL_VALUE)
    reader = MagicMock()
    reader.read.side_effect = lambda *args, **kwargs: values.copy()
    reader.__enter__ = lambda self: self
    reader.__exit__ = lambda self, *args: None

    monkeypatch.setattr(
        "reformatters.ucsb_chc.chirps.region_job.http_download_to_disk", fake_download
    )
    monkeypatch.setattr("rasterio.open", lambda _path: reader)


def _assert_constant_land_values(ds: xr.Dataset) -> None:
    land = ds.sel(_LAND_POINT, method="nearest")["precipitation_surface"]
    assert_allclose(
        land.values,
        np.full(land.shape, 24.0 * MM_PER_DAY_TO_KG_M2_S, dtype=np.float32),
        rtol=1e-2,
    )


def _icechunk_dataset_with_three_days(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[UcsbChcChirpsAnalysisPreliminaryDataset, set[pd.Timestamp]]:
    dataset = UcsbChcChirpsAnalysisPreliminaryDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    _shrink_template(monkeypatch, dataset)
    failed_days: set[pd.Timestamp] = set()

    def available(day: pd.Timestamp) -> bool:
        if day in failed_days:
            raise RuntimeError(f"failed to download {day:%Y-%m-%d}")
        return day <= pd.Timestamp("2025-01-03")

    _patch_source(monkeypatch, tmp_path, available, [])
    dataset.backfill_local(append_dim_end=pd.Timestamp("2025-01-04"))
    return dataset, failed_days


@pytest.mark.slow
def test_update_trims_to_last_day_with_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = _preliminary_dataset()
    _shrink_template(monkeypatch, dataset)
    requested_urls: list[str] = []
    _patch_source(
        monkeypatch,
        tmp_path,
        lambda day: day <= pd.Timestamp("2025-01-04"),
        requested_urls,
    )

    dataset.backfill_local(append_dim_end=pd.Timestamp("2025-01-03"))
    assert_array_equal(
        _open_store(dataset)["time"], pd.date_range("2025-01-01", "2025-01-02")
    )

    # The update window runs from the store's newest day through now, so it asks for
    # 2025-01-06 and 2025-01-07 which the source has not published.
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2025-01-07T12:00")),
    )
    requested_urls.clear()
    dataset.update("test-update")

    updated_ds = _open_store(dataset)
    assert_array_equal(updated_ds["time"], pd.date_range("2025-01-01", "2025-01-04"))
    _assert_constant_land_values(updated_ds)

    requested_days = sorted(url.removesuffix(".tif")[-10:] for url in requested_urls)
    # Every day from the store's newest through now is requested outright; nothing
    # lists what the source holds.
    assert set(requested_days) >= {
        "2025.01.02",
        "2025.01.03",
        "2025.01.04",
        "2025.01.05",
        "2025.01.06",
        "2025.01.07",
    }
    assert requested_days[-1] == "2025.01.07"
    assert all("/prelim/sat/" in url for url in requested_urls)


@pytest.mark.slow
def test_update_advances_past_an_unread_day_and_fills_it_in_later(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = _preliminary_dataset()
    _shrink_template(monkeypatch, dataset)
    requested_urls: list[str] = []
    gap = pd.Timestamp("2025-01-03")
    missing_gap = {gap}
    _patch_source(
        monkeypatch,
        tmp_path,
        lambda day: day <= pd.Timestamp("2025-01-04") and day not in missing_gap,
        requested_urls,
    )

    dataset.backfill_local(append_dim_end=pd.Timestamp("2025-01-03"))
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2025-01-05T12:00")),
    )
    dataset.update("test-update")

    gap_ds = _open_store(dataset)
    assert_array_equal(gap_ds["time"], pd.date_range("2025-01-01", "2025-01-04"))
    gap_land = gap_ds.sel(_LAND_POINT, method="nearest")["precipitation_surface"]
    assert np.isnan(gap_land.sel(time=gap).item())
    assert_allclose(
        gap_land.sel(time="2025-01-04").item(),
        24.0 * MM_PER_DAY_TO_KG_M2_S,
        rtol=1e-2,
    )

    missing_gap.clear()
    dataset.update("test-update-2")

    updated_ds = _open_store(dataset)
    assert_array_equal(updated_ds["time"], pd.date_range("2025-01-01", "2025-01-04"))
    _assert_constant_land_values(updated_ds)


@pytest.mark.slow
def test_failed_reread_before_a_later_success_writes_nan_then_refills(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset, failed_days = _icechunk_dataset_with_three_days(monkeypatch, tmp_path)
    failed_day = pd.Timestamp("2025-01-02")
    failed_days.add(failed_day)
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2025-01-05T12:00")),
    )

    dataset.update("test-failed-reread")
    failed_reread_ds = _open_store(dataset)
    assert_array_equal(
        failed_reread_ds["time"], pd.date_range("2025-01-01", "2025-01-03")
    )
    failed_land = failed_reread_ds.sel(_LAND_POINT, method="nearest")[
        "precipitation_surface"
    ]
    assert np.isnan(failed_land.sel(time=failed_day).item())
    assert_allclose(
        failed_land.sel(time="2025-01-03").item(),
        24.0 * MM_PER_DAY_TO_KG_M2_S,
        rtol=1e-2,
    )

    failed_days.clear()
    dataset.update("test-reread-restored")
    restored_ds = _open_store(dataset)
    assert_allclose(
        restored_ds.sel({**_LAND_POINT, "time": failed_day}, method="nearest")[
            "precipitation_surface"
        ].item(),
        24.0 * MM_PER_DAY_TO_KG_M2_S,
        rtol=1e-2,
    )


@pytest.mark.slow
def test_update_rejects_retraction_after_a_failed_reread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset, failed_days = _icechunk_dataset_with_three_days(monkeypatch, tmp_path)
    before = _open_store(dataset)
    before_time = before["time"].values.copy()
    before_land = before.sel(_LAND_POINT, method="nearest")[
        "precipitation_surface"
    ].values.copy()
    primary_repo, _ = dataset.store_factory.icechunk_primary_and_replica_repos()
    before_snapshot = primary_repo.lookup_branch("main")

    failed_days.add(pd.Timestamp("2025-01-03"))
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2025-01-05T12:00")),
    )
    with pytest.raises(ValueError, match="retract"):
        dataset.update("test-failed-reread")

    after = _open_store(dataset)
    assert_array_equal(after["time"].values, before_time)
    assert_allclose(
        after.sel(_LAND_POINT, method="nearest")["precipitation_surface"].values,
        before_land,
    )
    assert primary_repo.lookup_branch("main") == before_snapshot


@pytest.mark.slow
def test_update_spanning_two_time_shards_extends_through_the_second(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = _preliminary_dataset()
    original_get_template = dataset._get_template
    monkeypatch.setattr(
        dataset,
        "_get_template",
        lambda end: _set_time_chunks(
            shrink_chunks_and_shards(original_get_template(end)), 2
        ),
    )
    requested_urls: list[str] = []
    _patch_source(monkeypatch, tmp_path, lambda _day: True, requested_urls)

    dataset.backfill_local(append_dim_end=pd.Timestamp("2025-01-03"))
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2025-01-06T12:00")),
    )
    dataset.update("test-update")

    updated_ds = _open_store(dataset)
    assert_array_equal(updated_ds["time"], pd.date_range("2025-01-01", "2025-01-06"))
    _assert_constant_land_values(updated_ds)


def test_operational_kubernetes_resources() -> None:
    for dataset in (_final_dataset(), _preliminary_dataset()):
        update_cron_job, validation_cron_job = dataset.operational_kubernetes_resources(
            "test-image-tag"
        )
        assert update_cron_job.name == f"{dataset.dataset_id}-update"
        assert validation_cron_job.name == f"{dataset.dataset_id}-validate"
        assert not update_cron_job.suspend
        assert not validation_cron_job.suspend
        assert update_cron_job.secret_names == [
            dataset.primary_storage_config.k8s_secret_name
        ]


@pytest.mark.slow
@pytest.mark.parametrize("make_dataset", [_final_dataset, _preliminary_dataset])
def test_update_after_trimming_backfill_tail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_dataset: Callable[[], UcsbChcChirpsAnalysisMaterializedDataset],
) -> None:
    dataset = replace(
        make_dataset(),
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        ),
    )
    _shrink_template(monkeypatch, dataset)
    first_day = dataset.template_config.append_dim_start
    last_available = first_day + pd.Timedelta(days=2)
    _patch_source(monkeypatch, tmp_path, lambda day: day <= last_available, [])
    dataset.backfill_local(append_dim_end=first_day + pd.Timedelta(days=5))
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: first_day + pd.Timedelta(days=4, hours=12)),
    )
    with pytest.raises(ValueError, match="retract"):
        dataset.update("padded-tail")
    repo, _ = dataset.store_factory.icechunk_primary_and_replica_repos()
    assert trim_store(repo, dataset.dataset_id, commit=True).after_size == 3
    dataset.update("after-trim")
    assert _open_store(dataset).sizes["time"] == 3
    last_available = first_day + pd.Timedelta(days=3)
    dataset.update("new-source-day")
    updated = _open_store(dataset)
    assert updated.sizes["time"] == 4
    _assert_constant_land_values(updated)


@pytest.mark.parametrize(
    "make_dataset", [_final_dataset, _preliminary_dataset], ids=["final", "preliminary"]
)
@pytest.mark.parametrize(
    "now",
    [
        pd.Timestamp("2026-01-31T23:30"),
        pd.Timestamp("2026-02-28T23:30"),
        pd.Timestamp("2024-02-29T23:30"),
        pd.Timestamp("2026-04-30T23:30"),
    ],
)
def test_validation_follows_update_after_active_deadline(
    make_dataset: Callable[[], UcsbChcChirpsAnalysisMaterializedDataset],
    now: pd.Timestamp,
) -> None:
    update, validate = make_dataset().operational_kubernetes_resources("test-image-tag")
    deadline = update.pod_active_deadline

    assert validate.previous_fire_time(now + deadline) == (
        update.previous_fire_time(now) + deadline
    )


@pytest.mark.parametrize(
    ("dataset", "expected_delay"),
    [
        pytest.param(
            UcsbChcChirpsAnalysisFinalDataset(
                primary_storage_config=NOOP_STORAGE_CONFIG
            ),
            timedelta(days=60),
            id="final",
        ),
        pytest.param(
            UcsbChcChirpsAnalysisPreliminaryDataset(
                primary_storage_config=NOOP_STORAGE_CONFIG
            ),
            timedelta(days=10),
            id="preliminary",
        ),
    ],
)
def test_validators(
    dataset: UcsbChcChirpsAnalysisMaterializedDataset,
    expected_delay: timedelta,
) -> None:
    validators = tuple(dataset.validators())
    assert len(validators) == 2
    assert all(isinstance(v, validation.Validator) for v in validators)
    current_data = next(
        validator
        for validator in validators
        if isinstance(validator, validation.CheckCurrentData)
    )
    assert (
        current_data.max_delay
        == dataset.region_job_class.expected_unavailable_window
        == expected_delay
    )


def _nan_check_dataset(missing_newest: bool) -> xr.Dataset:
    times = pd.date_range("2025-01-01", periods=3, freq="1D")
    lat = np.linspace(59.975, -59.975, 60)
    lon = np.linspace(-179.975, 179.975, 180)
    values = np.zeros((len(times), len(lat), len(lon)), dtype=np.float32)
    rng = np.random.default_rng(0)
    ocean = rng.random((len(lat), len(lon))) < 0.719
    values[:, ocean] = np.nan
    if missing_newest:
        values[-1] = np.nan
    return xr.Dataset(
        {"precipitation_surface": (("time", "latitude", "longitude"), values)},
        coords={"time": times, "latitude": lat, "longitude": lon},
    )


@pytest.mark.parametrize("missing_newest", [False, True], ids=["complete", "missing"])
def test_nan_check_ignores_ocean_and_catches_a_missing_day(
    missing_newest: bool,
) -> None:
    check = next(
        v
        for v in _preliminary_dataset().validators()
        if isinstance(v, validation.CheckRecentNans)
    )
    context = validation.ValidationContext(
        store=zarr.storage.MemoryStore(),
        ds=_nan_check_dataset(missing_newest),
        append_dim="time",
    )

    assert check.check(context).passed is not missing_newest
