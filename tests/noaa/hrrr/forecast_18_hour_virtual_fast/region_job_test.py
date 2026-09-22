from collections.abc import Sequence
from pathlib import Path

import icechunk
import numpy as np
import obstore.store
import pandas as pd
import pytest
import xarray as xr
from zarr.storage import MemoryStore

from reformatters.common import template_utils
from reformatters.common.storage import (
    DatasetFormat,
    IcechunkVirtualConfig,
    StorageConfig,
    manifest_append_dim_split,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast import (
    region_job as region_job_module,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast.dynamical_dataset import (
    NoaaHrrrForecast18HourVirtualFastDataset,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast.region_job import (
    NoaaHrrrForecast18HourVirtualFastRegionJob,
    NoaaHrrrForecast18HourVirtualFastSourceFileCoord,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast.template_config import (
    RETENTION,
    NoaaHrrrForecast18HourVirtualFastTemplateConfig,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from reformatters.noaa.hrrr.nomads_mirror import MIRROR_LOCATION_PREFIX
from reformatters.noaa.hrrr.virtual_region_job import (
    NoaaHrrrForecastVirtualSourceFileCoord,
)

TEMPLATE_CONFIG = NoaaHrrrForecast18HourVirtualFastTemplateConfig()
FIXTURES = Path(__file__).parents[2] / "fixtures"
FIXTURE_GRIB = FIXTURES / "hrrr.t19z.wrfsfcf00.first2.grib2"
FIXTURE_INDEX = FIXTURES / "hrrr.t19z.wrfsfcf00.first2.grib2.idx"
MIRRORED_INIT = pd.Timestamp("2026-09-07T19:00")
LEAD_0 = pd.Timedelta(0)


def get_var(name: str) -> NoaaHrrrDataVar:
    return next(v for v in TEMPLATE_CONFIG.data_vars if v.name == name)


def coord(
    data_vars: Sequence[NoaaHrrrDataVar] | None = None,
    *,
    init_time: pd.Timestamp = MIRRORED_INIT,
    lead_time: pd.Timedelta = LEAD_0,
) -> NoaaHrrrForecast18HourVirtualFastSourceFileCoord:
    return NoaaHrrrForecast18HourVirtualFastSourceFileCoord(
        init_time=init_time,
        lead_time=lead_time,
        domain="conus",
        file_type="sfc",
        data_vars=data_vars or [get_var("composite_reflectivity")],
    )


def local_mirror_job_class(
    tmp_path: Path,
) -> type[NoaaHrrrForecast18HourVirtualFastRegionJob]:
    mirror_dir = tmp_path / "mirror"
    mirror_dir.mkdir(exist_ok=True)

    class LocalMirrorJob(NoaaHrrrForecast18HourVirtualFastRegionJob):
        def mirror_store(self) -> obstore.store.ObjectStore:
            return obstore.store.LocalStore(mirror_dir)

    return LocalMirrorJob


def make_job(
    tmp_path: Path, data_vars: Sequence[NoaaHrrrDataVar]
) -> NoaaHrrrForecast18HourVirtualFastRegionJob:
    return local_mirror_job_class(tmp_path)(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=TEMPLATE_CONFIG.get_template(MIRRORED_INIT + pd.Timedelta("1h")),
        data_vars=data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
        processing_mode="update",
    )


def mirror_file(
    tmp_path: Path,
    source_coord: NoaaHrrrForecast18HourVirtualFastSourceFileCoord,
) -> None:
    path = tmp_path / "mirror" / source_coord.relative_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(FIXTURE_GRIB.read_bytes())
    path.with_name(path.name + ".idx").write_bytes(FIXTURE_INDEX.read_bytes())


def test_source_file_coord_uses_public_mirror_url_and_nodd_key() -> None:
    source_coord = coord(
        init_time=pd.Timestamp("2024-06-01T01:00"),
        lead_time=pd.Timedelta("18h"),
    )
    expected = f"{MIRROR_LOCATION_PREFIX}hrrr.20240601/conus/hrrr.t01z.wrfsfcf18.grib2"

    assert source_coord.get_url() == expected
    assert source_coord.get_index_url() == expected + ".idx"


def test_discover_available_lists_only_mirror_and_requires_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    pending: list[NoaaHrrrForecastVirtualSourceFileCoord] = [coord()]
    captured: list[dict[str, object]] = []

    def fake(pending: list[object], **kwargs: object) -> list[tuple[object, int]]:
        captured.append(kwargs)
        return [(candidate, 100) for candidate in pending]

    monkeypatch.setattr(
        region_job_module, "discover_available_by_obstore_listing", fake
    )

    assert job.discover_available(pending) == [(pending[0], 100)]
    assert captured == [
        {
            "store": job.mirror_store(),
            "location_prefix": MIRROR_LOCATION_PREFIX,
            "require_index": True,
        }
    ]


def test_update_sweeps_whole_window_once_then_polls_only_recent_inits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(tmp_path, [get_var("composite_reflectivity")]).model_copy(
        update={"poll_deadline": pd.Timestamp.now() + pd.Timedelta("10s")}
    )
    job_class = type(job)
    monkeypatch.setattr(job_class, "tick_interval", pd.Timedelta(0))
    newest = job.template_ds.to_dataset().get_index("init_time")[-1]
    old_never_arrives = coord(init_time=newest - job.poll_window)
    recent_arrives_late = coord(init_time=newest)
    offered: list[list[object]] = []

    def fake_discover(
        pending: list[NoaaHrrrForecast18HourVirtualFastSourceFileCoord],
    ) -> list[tuple[NoaaHrrrForecast18HourVirtualFastSourceFileCoord, int]]:
        offered.append(list(pending))
        # Nothing is there on the first two sweeps; the recent file then arrives.
        return [(c, 100) for c in pending if len(offered) > 2]

    monkeypatch.setattr(
        job_class, "discover_available", lambda self, p: fake_discover(p)
    )
    monkeypatch.setattr(job_class, "file_refs", lambda self, c, size: [object()])

    batches = list(job.process_virtual_refs([old_never_arrives, recent_arrives_late]))

    assert offered == [
        [old_never_arrives, recent_arrives_late],
        [recent_arrives_late],
        [recent_arrives_late],
    ]
    assert [[c for c, _ in batch] for batch in batches] == [[recent_arrives_late]]


def test_index_and_data_reads_use_signed_mirror_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(tmp_path, [get_var("composite_reflectivity")])
    source_coord = coord()
    mirror_file(tmp_path, source_coord)
    local_index = tmp_path / "download.idx"
    monkeypatch.setattr(
        region_job_module, "get_local_path", lambda dataset_id, key: local_index
    )

    assert job.download_index(source_coord).read_bytes() == FIXTURE_INDEX.read_bytes()
    assert job.read_data_bytes(source_coord, 0, 16) == FIXTURE_GRIB.read_bytes()[:16]


def test_partial_source_file_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_vars = [get_var("composite_reflectivity"), get_var("temperature_2m")]
    job = make_job(tmp_path, data_vars)
    source_coord = coord(data_vars)
    mirror_file(tmp_path, source_coord)
    monkeypatch.setattr(
        region_job_module,
        "get_local_path",
        lambda dataset_id, key: tmp_path / "download.idx",
    )
    size = (tmp_path / "mirror" / source_coord.relative_path()).stat().st_size

    with pytest.raises(ValueError, match="temperature_2m"):
        job.file_refs(source_coord, size)
    assert job._file_refs_or_skip(source_coord, size) == []

    complete_coord = coord([get_var("composite_reflectivity")])
    assert len(job.file_refs(complete_coord, size)) == 1


def test_region_job_maintains_the_template_retention_window() -> None:
    assert NoaaHrrrForecast18HourVirtualFastRegionJob.drops_before_template_start
    assert (
        NoaaHrrrForecast18HourVirtualFastRegionJob.operational_update_window
        == RETENTION + pd.Timedelta("1h")
    )


def test_operational_update_region_covers_every_template_position(
    tmp_path: Path,
) -> None:
    fire_time = pd.Timestamp("2026-09-10T11:50")
    jobs, template_ds = (
        NoaaHrrrForecast18HourVirtualFastRegionJob.operational_update_jobs(
            primary_store=MemoryStore(),
            tmp_store=tmp_path / "tmp.zarr",
            get_template_fn=TEMPLATE_CONFIG.get_template,
            append_dim="init_time",
            all_data_vars=TEMPLATE_CONFIG.data_vars,
            reformat_job_name="test",
            job_fire_time=fire_time,
        )
    )
    assert template_ds.sizes["init_time"] == 73
    assert [job.region for job in jobs] == [slice(0, 73)]


def test_operational_update_decodes_fixture_after_window_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mirror_dir = tmp_path / "mirror"
    mirror_prefix = f"file://{mirror_dir}/"
    job_class = local_mirror_job_class(tmp_path)
    monkeypatch.setattr(job_class, "tick_interval", pd.Timedelta(0))
    monkeypatch.setattr(
        NoaaHrrrForecast18HourVirtualFastDataset,
        "_virtual_poll_deadline",
        lambda self, now: now,
    )
    monkeypatch.setattr(
        region_job_module,
        "get_local_path",
        lambda dataset_id, key: tmp_path / "download.idx",
    )
    monkeypatch.setattr(region_job_module, "MIRROR_LOCATION_PREFIX", mirror_prefix)
    container = icechunk.VirtualChunkContainer(
        mirror_prefix,
        icechunk.local_filesystem_store(str(mirror_dir)),
    )
    dataset = NoaaHrrrForecast18HourVirtualFastDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path / "store"), format=DatasetFormat.ICECHUNK
        ),
        region_job_class=job_class,
        icechunk_virtual_config=IcechunkVirtualConfig(
            containers=(container,),
            manifest_split=manifest_append_dim_split(split_size=100, dim="init_time"),
        ),
    )
    data_vars = [get_var("composite_reflectivity")]
    source_coord = coord(data_vars)
    mirror_file(tmp_path, source_coord)

    first_template = TEMPLATE_CONFIG.get_template(
        MIRRORED_INIT + pd.Timedelta("1h")
    ).isel(init_time=slice(-2, None), lead_time=[0])
    template_utils.write_metadata(
        first_template.isel(init_time=slice(0, 0)), dataset.store_factory
    )
    first_job = job_class(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=first_template,
        data_vars=data_vars,
        append_dim="init_time",
        region=slice(0, 2),
        reformat_job_name="test-first",
        processing_mode="update",
    )

    dataset._run_virtual_operational_update(
        [first_job], worker_index=0, workers_total=1
    )

    first = xr.open_zarr(
        dataset.store_factory.primary_store(), consolidated=False, chunks=None
    )
    first_values = (
        first["composite_reflectivity"]
        .sel(init_time=MIRRORED_INIT, lead_time=LEAD_0)
        .values.copy()
    )
    assert np.isfinite(first_values).any()
    primary_repo, _ = dataset.store_factory.icechunk_primary_and_replica_repos()
    assert mirror_prefix + source_coord.relative_path() in (
        primary_repo.readonly_session("main").all_virtual_chunk_locations()
    )

    second_template = TEMPLATE_CONFIG.get_template(
        MIRRORED_INIT + pd.Timedelta("2h")
    ).isel(init_time=slice(-2, None), lead_time=[0])
    second_job = job_class(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=second_template,
        data_vars=data_vars,
        append_dim="init_time",
        region=slice(0, 2),
        reformat_job_name="test-second",
        processing_mode="update",
    )

    dataset._run_virtual_operational_update(
        [second_job], worker_index=0, workers_total=1
    )

    second = xr.open_zarr(
        dataset.store_factory.primary_store(), consolidated=False, chunks=None
    )
    assert second.get_index("init_time").equals(pd.DatetimeIndex([MIRRORED_INIT]))
    np.testing.assert_array_equal(
        second["composite_reflectivity"]
        .sel(init_time=MIRRORED_INIT, lead_time=LEAD_0)
        .values,
        first_values,
    )
    np.testing.assert_array_equal(
        second["valid_time"].values,
        (second["init_time"] + second["lead_time"]).values,
    )
