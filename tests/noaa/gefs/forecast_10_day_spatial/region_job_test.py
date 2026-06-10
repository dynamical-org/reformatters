from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr

from reformatters.noaa.gefs.forecast_10_day_spatial import (
    region_job as region_job_module,
)
from reformatters.noaa.gefs.forecast_10_day_spatial.region_job import (
    GefsForecast10DaySpatialRegionJob,
    GefsForecast10DaySpatialSourceFileCoord,
)
from reformatters.noaa.gefs.forecast_10_day_spatial.template_config import (
    GefsForecast10DaySpatialTemplateConfig,
)
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar

TEMPLATE_CONFIG = GefsForecast10DaySpatialTemplateConfig()


def get_var(name: str) -> GEFSDataVar:
    return next(v for v in TEMPLATE_CONFIG.data_vars if v.name == name)


@pytest.fixture(scope="module")
def template_ds() -> xr.Dataset:
    # Spans the 2022-10-18T12 transition when "s+b-b22" vars entered the s files.
    return TEMPLATE_CONFIG.get_template(pd.Timestamp("2022-10-19T00:00"))


def make_job(
    template_ds: xr.Dataset,
    data_vars: Sequence[GEFSDataVar] | None = None,
    region: slice = slice(0, 1),
) -> GefsForecast10DaySpatialRegionJob:
    return GefsForecast10DaySpatialRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=data_vars or TEMPLATE_CONFIG.data_vars,
        append_dim="init_time",
        region=region,
        reformat_job_name="test",
    )


def test_source_file_coord_urls() -> None:
    coord = GefsForecast10DaySpatialSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01T06:00"),
        ensemble_member=1,
        lead_time=pd.Timedelta("3h"),
        data_vars=[get_var("temperature_2m")],
    )
    assert coord.get_url() == (
        "https://noaa-gefs-pds.s3.amazonaws.com/"
        "gefs.20240101/06/atmos/pgrb2sp25/gep01.t06z.pgrb2s.0p25.f003"
    )
    assert coord.get_s3_location() == (
        "s3://noaa-gefs-pds/gefs.20240101/06/atmos/pgrb2sp25/gep01.t06z.pgrb2s.0p25.f003"
    )


def test_source_file_coord_url_control_member_max_lead() -> None:
    coord = GefsForecast10DaySpatialSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01T00:00"),
        ensemble_member=0,
        lead_time=pd.Timedelta("240h"),
        data_vars=[get_var("temperature_2m")],
    )
    assert coord.get_url() == (
        "https://noaa-gefs-pds.s3.amazonaws.com/"
        "gefs.20240101/00/atmos/pgrb2sp25/gec00.t00z.pgrb2s.0p25.f240"
    )


def test_source_file_coord_rejects_lead_beyond_s_files() -> None:
    coord = GefsForecast10DaySpatialSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01T00:00"),
        ensemble_member=1,
        lead_time=pd.Timedelta("246h"),
        data_vars=[get_var("temperature_2m")],
    )
    with pytest.raises(AssertionError, match="s files end at"):
        _ = coord.gefs_file_type


def test_source_file_coord_rejects_pre_b22_var() -> None:
    coord = GefsForecast10DaySpatialSourceFileCoord(
        init_time=pd.Timestamp("2021-01-01T00:00"),  # before the b22 transition
        ensemble_member=1,
        lead_time=pd.Timedelta("3h"),
        data_vars=[get_var("percent_frozen_precipitation_surface")],  # "s+b-b22"
    )
    with pytest.raises(AssertionError, match="not in the s file"):
        _ = coord.gefs_file_type


def test_generate_source_file_coords_one_file_per_init_member_lead(
    template_ds: xr.Dataset,
) -> None:
    data_vars = [
        get_var("temperature_2m"),  # instant
        get_var("precipitation_surface"),  # no hour-0 values
        get_var("percent_frozen_precipitation_surface"),  # "s+b-b22", instant
    ]
    job = make_job(template_ds, data_vars=data_vars)
    # The final init (2022-10-18T18) is after the b22 transition.
    processing_region_ds = template_ds.isel(
        init_time=slice(-1, None), ensemble_member=slice(0, 2)
    ).sel(lead_time=[pd.Timedelta("0h"), pd.Timedelta("3h")])

    coords = job.generate_source_file_coords(processing_region_ds, data_vars)

    # One s file per (init, member, lead): 1 init x 2 members x 2 leads.
    assert len(coords) == 4
    for coord in coords:
        assert coord.gefs_file_type == "s"
        if coord.lead_time == pd.Timedelta("0h"):
            # precipitation has no hour-0 values
            assert [v.name for v in coord.data_vars] == [
                "temperature_2m",
                "percent_frozen_precipitation_surface",
            ]
        else:
            assert [v.name for v in coord.data_vars] == [v.name for v in data_vars]


def test_generate_source_file_coords_excludes_b22_vars_before_transition(
    template_ds: xr.Dataset,
) -> None:
    data_vars = [
        get_var("temperature_2m"),
        get_var("percent_frozen_precipitation_surface"),  # "s+b-b22"
    ]
    job = make_job(template_ds, data_vars=data_vars)
    # The first init (2020-10-01T00) is before the b22 transition.
    processing_region_ds = template_ds.isel(
        init_time=slice(0, 1), ensemble_member=slice(0, 1)
    ).sel(lead_time=[pd.Timedelta("3h")])

    (coord,) = job.generate_source_file_coords(processing_region_ds, data_vars)
    assert [v.name for v in coord.data_vars] == ["temperature_2m"]
    assert coord.gefs_file_type == "s"


def test_representative_var_uses_coords_own_file_vars(template_ds: xr.Dataset) -> None:
    job = make_job(template_ds)
    avg_then_instant = GefsForecast10DaySpatialSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01T00:00"),
        ensemble_member=1,
        lead_time=pd.Timedelta("3h"),
        data_vars=[get_var("precipitation_surface"), get_var("temperature_2m")],
    )
    assert job.representative_var(avg_then_instant).name == "temperature_2m"

    avg_only = GefsForecast10DaySpatialSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01T00:00"),
        ensemble_member=1,
        lead_time=pd.Timedelta("3h"),
        data_vars=[get_var("precipitation_surface")],
    )
    assert job.representative_var(avg_only).name == "precipitation_surface"


_INDEX_CONTENT = """1:0:d=2020100100:PRES:surface:3 hour fcst:ENS=+1
2:1000:d=2020100100:TMP:2 m above ground:3 hour fcst:ENS=+1
3:2500:d=2020100100:APCP:surface:0-3 hour acc fcst:ENS=+1
"""


def test_process_virtual_refs(
    template_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_vars = [get_var("temperature_2m"), get_var("precipitation_surface")]
    job = make_job(template_ds, data_vars=data_vars)

    def fake_download(url: str, dataset_id: str) -> Path:
        if "gep02" in url:  # member 2's file is not yet published
            raise FileNotFoundError(url)
        index_path = tmp_path / "index.idx"
        index_path.write_text(_INDEX_CONTENT)
        return index_path

    monkeypatch.setattr(region_job_module, "http_download_to_disk", fake_download)
    monkeypatch.setattr(region_job_module, "_content_length", lambda url: 9000)

    def coord(member: int) -> GefsForecast10DaySpatialSourceFileCoord:
        return GefsForecast10DaySpatialSourceFileCoord(
            init_time=pd.Timestamp("2020-10-01T00:00"),
            ensemble_member=member,
            lead_time=pd.Timedelta("3h"),
            data_vars=data_vars,
        )

    batches = list(job.process_virtual_refs([coord(1), coord(2)]))

    # One yield containing only the published file's refs, one ref per var.
    assert len(batches) == 1
    (refs,) = batches
    assert [r.data_var.name for r in refs] == [
        "temperature_2m",
        "precipitation_surface",
    ]
    tmp_ref, apcp_ref = refs
    assert tmp_ref.offset == 1000
    assert tmp_ref.length == 1500
    # APCP is the last message in the index; its end comes from the file size.
    assert apcp_ref.offset == 2500
    assert apcp_ref.length == 9000 - 2500
    for ref in refs:
        assert ref.location == (
            "s3://noaa-gefs-pds/gefs.20201001/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f003"
        )
        assert ref.out_loc == {
            "init_time": pd.Timestamp("2020-10-01T00:00"),
            "lead_time": pd.Timedelta("3h"),
            "ensemble_member": 1,
        }


def test_process_virtual_refs_no_available_files_yields_nothing(
    template_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(template_ds)

    def fake_download(url: str, dataset_id: str) -> Path:
        raise FileNotFoundError(url)

    monkeypatch.setattr(region_job_module, "http_download_to_disk", fake_download)
    coord = GefsForecast10DaySpatialSourceFileCoord(
        init_time=pd.Timestamp("2020-10-01T00:00"),
        ensemble_member=1,
        lead_time=pd.Timedelta("3h"),
        data_vars=[get_var("temperature_2m")],
    )
    assert list(job.process_virtual_refs([coord])) == []


def test_operational_update_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    now = pd.Timestamp("2020-10-03T01:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *a, **kw: now))

    jobs, template_ds = GefsForecast10DaySpatialRegionJob.operational_update_jobs(
        primary_store=Mock(),
        tmp_store=Path("unused-tmp.zarr"),
        get_template_fn=TEMPLATE_CONFIG.get_template,
        append_dim="init_time",
        all_data_vars=TEMPLATE_CONFIG.data_vars,
        reformat_job_name="test",
    )

    (job,) = jobs
    init_times = template_ds.get_index("init_time")
    assert init_times[-1] == pd.Timestamp("2020-10-03T00:00")
    # One job spanning the 24h active window (4 init times at 6h frequency).
    assert job.region == slice(len(init_times) - 4, len(init_times))
    assert [v.name for v in job.data_vars] == [
        v.name for v in TEMPLATE_CONFIG.data_vars
    ]
