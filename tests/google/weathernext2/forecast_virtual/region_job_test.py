from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr

from reformatters.common.config_models import ROOT
from reformatters.google.weathernext2.forecast_virtual import (
    region_job as region_job_module,
)
from reformatters.google.weathernext2.forecast_virtual.region_job import (
    SOURCE_LOCATION_PREFIX,
    GoogleWeathernext2ForecastVirtualRegionJob,
    GoogleWeathernext2ForecastVirtualSourceFileCoord,
)
from reformatters.google.weathernext2.forecast_virtual.template_config import (
    GoogleWeathernext2DataVar,
    GoogleWeathernext2ForecastVirtualTemplateConfig,
)

TEMPLATE_CONFIG = GoogleWeathernext2ForecastVirtualTemplateConfig()

_YEARLY_INIT = pd.Timestamp("2022-01-02T12:00")  # store index 6 of 2022_to_2023
_PER_INIT_INIT = pd.Timestamp("2025-03-01T06:00")

_YEARLY_STORE = (
    f"{SOURCE_LOCATION_PREFIX}weathernext_2_0_0_mean/zarr/2022_to_2023/predictions.zarr"
)
_PER_INIT_STORE = (
    f"{SOURCE_LOCATION_PREFIX}weathernext_2_0_0_mean/zarr/2025_to_present/"
    "20250301_06hr_01_preds/predictions.zarr"
)


def get_var(path: str) -> GoogleWeathernext2DataVar:
    return next(v for v in TEMPLATE_CONFIG.data_vars if v.path == path)


@pytest.fixture(scope="module")
def template_ds() -> xr.DataTree:
    return TEMPLATE_CONFIG.get_template(pd.Timestamp("2022-01-01T06:00"))


def make_job(
    template_ds: xr.DataTree,
    data_vars: Sequence[GoogleWeathernext2DataVar] | None = None,
    processing_mode: Literal["backfill", "update"] = "backfill",
) -> GoogleWeathernext2ForecastVirtualRegionJob:
    return GoogleWeathernext2ForecastVirtualRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=data_vars or TEMPLATE_CONFIG.data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
        processing_mode=processing_mode,
    )


def _coord(
    data_vars: Sequence[GoogleWeathernext2DataVar],
    init_time: pd.Timestamp = _PER_INIT_INIT,
) -> GoogleWeathernext2ForecastVirtualSourceFileCoord:
    return GoogleWeathernext2ForecastVirtualSourceFileCoord(
        init_time=init_time, data_vars=data_vars
    )


# --- URLs, markers and chunk keys ---


def test_get_url_yearly_store() -> None:
    assert _coord([get_var("temperature_2m")], _YEARLY_INIT).get_url() == _YEARLY_STORE


def test_get_url_per_init_store() -> None:
    assert _coord([get_var("temperature_2m")]).get_url() == _PER_INIT_STORE


def test_success_marker_sits_beside_the_store() -> None:
    yearly = _coord([get_var("temperature_2m")], _YEARLY_INIT)
    assert yearly.get_success_marker_url() == (
        f"{SOURCE_LOCATION_PREFIX}weathernext_2_0_0_mean/zarr/2022_to_2023/success"
    )
    per_init = _coord([get_var("temperature_2m")])
    assert per_init.get_success_marker_url() == (
        f"{SOURCE_LOCATION_PREFIX}weathernext_2_0_0_mean/zarr/2025_to_present/"
        "20250301_06hr_01_preds/success"
    )


def test_chunk_key_yearly_store_leads_with_the_init_index() -> None:
    coord = _coord([get_var("temperature_2m")], _YEARLY_INIT)
    # 2022-01-02T12 is the 7th 6-hourly init of 2022, at index 6.
    assert coord.chunk_key(get_var("temperature_2m"), 3, None) == (
        "2m_temperature/6.3.0.0"
    )


def test_chunk_key_per_init_store_has_no_init_index() -> None:
    coord = _coord([get_var("temperature_2m")])
    assert coord.chunk_key(get_var("temperature_2m"), 3, None) == (
        "2m_temperature/3.0.0"
    )
    assert coord.chunk_key(get_var("pressure_level/temperature"), 3, 12) == (
        "temperature/3.12.0.0"
    )


def test_out_loc_names_one_representative_lead() -> None:
    assert dict(_coord([get_var("temperature_2m")]).out_loc()) == {
        "init_time": _PER_INIT_INIT,
        "lead_time": pd.Timedelta("6h"),
    }


# --- generate_source_file_coords ---


def _region_ds(template_ds: xr.DataTree, init_time: pd.Timestamp) -> xr.Dataset:
    return (
        template_ds.to_dataset()
        .isel(init_time=slice(0, 1))
        .assign_coords(init_time=[init_time])
    )


def test_generate_source_file_coords_yearly_era_omits_pressure_level(
    template_ds: xr.DataTree,
) -> None:
    job = make_job(template_ds)
    (coord,) = job.generate_source_file_coords(
        _region_ds(template_ds, _YEARLY_INIT), TEMPLATE_CONFIG.data_vars
    )
    assert coord.init_time == _YEARLY_INIT
    # Before the per-init stores all 13 levels share one chunk, which no reference can
    # address, so only root variables are available.
    assert {var.group for var in coord.data_vars} == {ROOT}
    assert len(coord.data_vars) == 10


def test_generate_source_file_coords_per_init_era_includes_every_var(
    template_ds: xr.DataTree,
) -> None:
    job = make_job(template_ds)
    (coord,) = job.generate_source_file_coords(
        _region_ds(template_ds, _PER_INIT_INIT), TEMPLATE_CONFIG.data_vars
    )
    assert {var.path for var in coord.data_vars} == {
        var.path for var in TEMPLATE_CONFIG.data_vars
    }


def test_generate_source_file_coords_one_coord_per_init(
    template_ds: xr.DataTree,
) -> None:
    job = make_job(template_ds)
    region_ds = template_ds.to_dataset().assign_coords(
        init_time=pd.date_range(_PER_INIT_INIT, periods=1, freq="6h")
    )
    # A coord is a whole store (every lead time of one init), not one file per lead.
    assert (
        len(job.generate_source_file_coords(region_ds, TEMPLATE_CONFIG.data_vars)) == 1
    )


# --- file_refs ---


def _fake_listing(
    monkeypatch: pytest.MonkeyPatch, sizes_by_key: dict[str, int]
) -> None:
    def fake(
        store: Any,  # noqa: ANN401 - stands in for an obstore store
        store_key_prefix: str,
        chunk_key_prefix: str,
    ) -> dict[str, int]:
        return {
            key: size
            for key, size in sizes_by_key.items()
            if key.startswith(chunk_key_prefix)
        }

    monkeypatch.setattr(region_job_module, "gcs_store", lambda url: Mock())
    monkeypatch.setattr(region_job_module, "_list_chunk_sizes", fake)


_N_LEADS = 60
_N_SOURCE_LEVELS = 13


def _full_listing(
    source_name: str,
    *,
    store_init_index: int | None = None,
    n_levels: int | None = None,
) -> dict[str, int]:
    """Every chunk object of one init's slice of `source_name`, each with a distinct
    size so a reference's length identifies the chunk it came from."""
    init_prefix = "" if store_init_index is None else f"{store_init_index}."
    level_indices: Sequence[int | None] = (
        [None] if n_levels is None else list(range(n_levels))
    )
    listing: dict[str, int] = {}
    for lead_index in range(_N_LEADS):
        for level_index in level_indices:
            indices = (
                f"{lead_index}"
                if level_index is None
                else f"{lead_index}.{level_index}"
            )
            listing[f"{source_name}/{init_prefix}{indices}.0.0"] = 1000 + len(listing)
    return listing


def test_file_refs_root_var_points_at_whole_chunk_objects(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    listing = _full_listing("2m_temperature")
    _fake_listing(monkeypatch, listing)
    var = get_var("temperature_2m")
    job = make_job(template_ds, data_vars=[var])

    refs = job.file_refs(_coord([var]), file_size=0)

    assert len(refs) == _N_LEADS
    by_lead = {ref.out_loc["lead_time"]: ref for ref in refs}
    first = by_lead[pd.Timedelta("6h")]
    assert first.location == f"{_PER_INIT_STORE}/2m_temperature/0.0.0"
    # One source chunk is one whole object, so every reference starts at byte 0.
    assert (first.offset, first.length) == (0, listing["2m_temperature/0.0.0"])
    assert by_lead[pd.Timedelta("12h")].length == listing["2m_temperature/1.0.0"]
    assert all(ref.out_loc["init_time"] == _PER_INIT_INIT for ref in refs)


def test_file_refs_maps_descending_levels_onto_ascending_source_indices(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Source levels ascend [50 ... 1000], so 1000 hPa is source index 12 and 50 hPa is 0.
    listing = _full_listing("temperature", n_levels=_N_SOURCE_LEVELS)
    _fake_listing(monkeypatch, listing)
    var = get_var("pressure_level/temperature")
    job = make_job(template_ds, data_vars=[var])

    refs = job.file_refs(_coord([var]), file_size=0)

    assert len(refs) == _N_LEADS * _N_SOURCE_LEVELS
    by_level = {
        ref.out_loc["pressure_level"]: ref
        for ref in refs
        if ref.out_loc["lead_time"] == pd.Timedelta("6h")
    }
    assert by_level[1000].location == f"{_PER_INIT_STORE}/temperature/0.12.0.0"
    assert by_level[1000].length == listing["temperature/0.12.0.0"]
    assert by_level[50].location == f"{_PER_INIT_STORE}/temperature/0.0.0.0"
    assert by_level[50].length == listing["temperature/0.0.0.0"]


def test_file_refs_yearly_store_keys_carry_the_init_index(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_listing(monkeypatch, _full_listing("2m_temperature", store_init_index=6))
    var = get_var("temperature_2m")
    job = make_job(template_ds, data_vars=[var])

    refs = job.file_refs(_coord([var], _YEARLY_INIT), file_size=0)

    assert len(refs) == _N_LEADS
    first = next(r for r in refs if r.out_loc["lead_time"] == pd.Timedelta("6h"))
    assert first.location == f"{_YEARLY_STORE}/2m_temperature/6.0.0.0"


@pytest.mark.parametrize("dropped_keys", [1, _N_LEADS])
def test_file_refs_rejects_a_listing_missing_chunks(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, dropped_keys: int
) -> None:
    # Silently dropping refs would commit the init with those positions reading as fill,
    # which no validator can distinguish from a legitimately absent forecast.
    listing = _full_listing("2m_temperature")
    for key in list(listing)[-dropped_keys:]:
        del listing[key]
    _fake_listing(monkeypatch, listing)
    var = get_var("temperature_2m")
    job = make_job(template_ds, data_vars=[var])

    with pytest.raises(
        AssertionError,
        match=f"listed {_N_LEADS - dropped_keys} of {_N_LEADS} expected source chunks",
    ):
        job.file_refs(_coord([var]), file_size=0)


# --- discover_available ---


def test_discover_available_gates_on_the_success_marker(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = _coord([get_var("temperature_2m")])
    unpublished = _coord([get_var("temperature_2m")], pd.Timestamp("2025-03-01T12:00"))
    probed: list[str] = []

    def fake_exists(store: Any, key: str) -> bool:  # noqa: ANN401 - stands in for a store
        probed.append(key)
        return "20250301_06hr" in key

    monkeypatch.setattr(region_job_module, "gcs_store", lambda url: Mock())
    monkeypatch.setattr(region_job_module, "_object_exists", fake_exists)
    job = make_job(template_ds)

    result = job.discover_available([published, unpublished])

    assert [coord for coord, _ in result] == [published]
    assert len(probed) == 2


def test_discover_available_probes_a_shared_yearly_marker_once(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    coords = [
        _coord([get_var("temperature_2m")], pd.Timestamp("2022-01-01T00:00")),
        _coord([get_var("temperature_2m")], pd.Timestamp("2022-06-01T00:00")),
    ]
    probed: list[str] = []

    def fake_exists(store: Any, key: str) -> bool:  # noqa: ANN401 - stands in for a store
        probed.append(key)
        return True

    monkeypatch.setattr(region_job_module, "gcs_store", lambda url: Mock())
    monkeypatch.setattr(region_job_module, "_object_exists", fake_exists)
    job = make_job(template_ds)

    result = job.discover_available(coords)

    assert [coord for coord, _ in result] == coords
    assert len(probed) == 1


# --- operational_update_jobs ---


def test_operational_update_jobs_single_polling_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A production fire time: init+6h55m for the 2025-03-01T00 init.
    now = pd.Timestamp("2025-03-01T06:55")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *a, **kw: now))

    jobs, template_ds = (
        GoogleWeathernext2ForecastVirtualRegionJob.operational_update_jobs(
            primary_store=Mock(),
            tmp_store=Path("unused-tmp.zarr"),
            get_template_fn=TEMPLATE_CONFIG.get_template,
            append_dim="init_time",
            all_data_vars=TEMPLATE_CONFIG.data_vars,
            reformat_job_name="test",
        )
    )

    (job,) = jobs
    assert isinstance(job, GoogleWeathernext2ForecastVirtualRegionJob)
    assert job.processing_mode == "update"
    init_times = template_ds.to_dataset().get_index("init_time")
    assert now - init_times[-1] < TEMPLATE_CONFIG.append_dim_frequency
    # 30h window at a fire time = the just-published init, its 3 prior cycles, and the
    # next init's slot, which the source will not publish for another ~6 hours.
    assert job.region == slice(len(init_times) - 5, len(init_times))
