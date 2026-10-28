from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import xarray as xr

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common.template_config import TemplateConfig
from reformatters.ecmwf.aifs_single.forecast.region_job import (
    EcmwfAifsSingleForecastSourceFileCoord,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.source_file_coord import (
    OpenDataSourceFileCoord,
)
from reformatters.noaa.gefs.forecast_16_day_0_5_degree_virtual.region_job import (
    NoaaGefsForecast16Day05DegreeVirtualRegionJob,
)
from scripts.timing_profile.datasets import expected_units, source_gap, usable_file
from scripts.timing_profile.report import aggregate_unit, recommend
from scripts.timing_profile.runtime import RuntimeStats
from scripts.timing_profile.sources import Listing, TimestampSource, listing_prefix


@pytest.mark.parametrize(
    "dataset_id",
    [
        "noaa-gfs-analysis",
        "noaa-gfs-analysis-virtual",
        "noaa-gefs-analysis",
        "noaa-gefs-analysis-0-25-degree-virtual",
    ],
)
def test_analysis_readiness_includes_boundary_and_only_target_cycle(
    dataset_id: str, tmp_path: Path
) -> None:
    dataset = next(ds for ds in DYNAMICAL_DATASETS if ds.dataset_id == dataset_id)
    source = TimestampSource(tmp_path, 1)
    units = list(
        expected_units(
            dataset, pd.Timestamp("2026-10-20"), pd.Timestamp("2026-10-20T06"), source
        )
    )
    source.close()
    assert len(units) == 1
    urls = {url for file in units[0].files for url in file}
    assert any("f006" in url for url in urls)
    assert any("f000" in url for url in urls)
    assert not any("20261019" in url or "/06/" in url for url in urls)
    assert all(len(file) == 2 for file in units[0].files)
    if dataset_id.endswith("virtual"):
        assert units[0].visibility_groups
        gated_urls = {
            url
            for group in units[0].visibility_groups
            for file in group
            for url in file
        }
        assert not any("f006" in url for url in gated_urls)
        assert any("f000" in url for url in gated_urls)


def test_hrrr_materialized_readiness_waits_for_the_trim_boundary_index(
    tmp_path: Path,
) -> None:
    dataset = next(
        ds for ds in DYNAMICAL_DATASETS if ds.dataset_id == "noaa-hrrr-analysis"
    )
    anchor = pd.Timestamp("2026-10-20T12:00")
    source = TimestampSource(tmp_path, 1)
    (unit,) = expected_units(dataset, anchor, anchor + pd.Timedelta(hours=1), source)
    source.close()
    prefix = "s3://noaa-hrrr-bdp-pds/hrrr.20261020/conus/"
    arrivals = {
        "hrrr.t11z.wrfsfcf01.grib2": "2026-10-20T11:53",
        "hrrr.t11z.wrfsfcf01.grib2.idx": "2026-10-20T11:54",
        "hrrr.t12z.wrfsfcf00.grib2": "2026-10-20T12:50",
        "hrrr.t12z.wrfsfcf00.grib2.idx": "2026-10-20T12:51",
        "hrrr.t12z.wrfsfcf01.grib2": "2026-10-20T13:01",
        "hrrr.t12z.wrfsfcf01.grib2.idx": "2026-10-20T13:02",
        "hrrr.t13z.wrfsfcf00.grib2": "2026-10-20T13:50",
        "hrrr.t13z.wrfsfcf00.grib2.idx": "2026-10-20T13:51",
    }
    expected = {
        frozenset((prefix + name, prefix + name + ".idx"))
        for name in (
            "hrrr.t11z.wrfsfcf01.grib2",
            "hrrr.t12z.wrfsfcf00.grib2",
            "hrrr.t12z.wrfsfcf01.grib2",
        )
    }
    assert set(unit.files) == expected
    objects = {prefix + name: pd.Timestamp(time) for name, time in arrivals.items()}
    listings = {listing_prefix(url): Listing(objects, "fixed input") for url in objects}
    measured = aggregate_unit(unit, listings)
    assert measured.first == -360
    assert measured.complete == 62 * 60
    before_boundary_index = aggregate_unit(
        unit, listings, as_of=anchor + pd.Timedelta(minutes=61)
    )
    assert before_boundary_index.complete is None
    assert before_boundary_index.missing_files == 1


def test_gefs_35_day_readiness_tiers_include_members_and_indexes(
    tmp_path: Path,
) -> None:
    dataset = next(
        ds
        for ds in DYNAMICAL_DATASETS
        if ds.dataset_id == "noaa-gefs-forecast-35-day-0-5-degree-virtual"
    )
    source = TimestampSource(tmp_path, 1)
    units = list(
        expected_units(
            dataset, pd.Timestamp("2026-10-20"), pd.Timestamp("2026-10-21"), source
        )
    )
    source.close()
    assert {unit.tier for unit in units} == {
        "tier 1 (through f384)",
        "tier 2 (extension)",
    }
    early = next(unit for unit in units if unit.tier.startswith("tier 1"))
    late = next(unit for unit in units if unit.tier.startswith("tier 2"))
    assert any("gep30" in url for file in early.files for url in file)
    assert any(".f840" in url for file in late.files for url in file)
    assert all(len(file) == 2 for unit in units for file in unit.files)


@pytest.mark.parametrize(
    "dataset_id",
    [
        "u-arizona-swann-analysis",
        "nasa-smap-level3-36km-v9",
        "nasa-imerg-analysis-early",
    ],
)
def test_rewritten_and_protected_sources_are_gaps(dataset_id: str) -> None:
    assert source_gap(dataset_id) is not None


def test_weathernext_unit_contains_every_configured_member_and_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = next(
        ds
        for ds in DYNAMICAL_DATASETS
        if ds.dataset_id == "google-weathernext2-forecast-operational-virtual"
    )
    config = dataset.template_config
    coords = config.dimension_coordinates()
    coords["ensemble_member"] = [0, 1]
    coords["lead_time"] = pd.to_timedelta(["6h"])

    def small_coords(self: TemplateConfig[Any]) -> dict[str, Any]:
        return coords

    monkeypatch.setattr(type(config), "dimension_coordinates", small_coords)
    source = TimestampSource(tmp_path, 1)
    units = list(
        expected_units(
            dataset, pd.Timestamp("2026-10-20"), pd.Timestamp("2026-10-20T06"), source
        )
    )
    source.close()
    pressure = next(unit for unit in units if unit.tier.startswith("pressure_level/"))
    assert len(pressure.files) == 1
    assert len(pressure.files[0]) == 2 * len(coords["pressure_level"])
    assert all(url.startswith("gs://weathernext/") for url in pressure.files[0])


def test_ecmwf_materialized_units_measure_the_operational_gcs_primary() -> None:
    for coord in [
        EcmwfAifsSingleForecastSourceFileCoord(
            init_time=pd.Timestamp("2026-10-20"),
            lead_time=pd.Timedelta("6h"),
            data_var_group=[],
        ),
        OpenDataSourceFileCoord(
            init_time=pd.Timestamp("2026-10-20"),
            lead_time=pd.Timedelta("6h"),
            data_var_group=[],
            ensemble_member=0,
        ),
    ]:
        files = usable_file(coord)
        assert len(files) == 2
        assert all(url.startswith("gs://ecmwf-open-data/") for url in files)
        assert any(url.endswith(".index") for url in files)


def test_gefs_timing_representatives_preserve_every_required_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = next(
        ds
        for ds in DYNAMICAL_DATASETS
        if ds.dataset_id == "noaa-gefs-forecast-16-day-0-5-degree-virtual"
    )
    config = dataset.template_config
    coords = config.dimension_coordinates()
    coords["ensemble_member"] = [0, 30]
    coords["lead_time"] = pd.to_timedelta(["0h", "3h", "384h"])
    anchor = pd.Timestamp("2026-10-20")
    coords["init_time"] = pd.DatetimeIndex([anchor])

    def small_coords(self: TemplateConfig[Any]) -> dict[str, Any]:
        return coords

    monkeypatch.setattr(type(config), "dimension_coordinates", small_coords)
    processing = xr.Dataset(coords=coords)
    job = NoaaGefsForecast16Day05DegreeVirtualRegionJob(
        tmp_store=tmp_path,
        template_ds=xr.DataTree(processing),
        data_vars=config.data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
    )
    full_files = {
        usable_file(coord)
        for coord in job.generate_source_file_coords(processing, config.data_vars)
    }
    source = TimestampSource(tmp_path, 1)
    measured = list(
        expected_units(dataset, anchor, anchor + pd.Timedelta("6h"), source)
    )
    source.close()
    assert set(measured[0].files) == full_files


@pytest.mark.parametrize("product", ["final", "preliminary"])
def test_chirps_keeps_unavailable_positions_without_calling_live_discovery(
    product: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = next(
        ds
        for ds in DYNAMICAL_DATASETS
        if ds.dataset_id == f"ucsb-chc-chirps-analysis-{product}"
    )

    def unexpected_discovery(*args: object, **kwargs: object) -> None:
        raise AssertionError("Timing probes must not filter unavailable source dates")

    monkeypatch.setattr(
        "reformatters.ucsb_chc.chirps.region_job._available_file_urls",
        unexpected_discovery,
    )
    source = TimestampSource(tmp_path, 1)
    units = list(
        expected_units(
            dataset, pd.Timestamp("2026-10-20"), pd.Timestamp("2026-10-23"), source
        )
    )
    source.close()
    assert len(units) == 3
    assert all(unit.anchor_minus_label == 86400 for unit in units)
    assert any("2026.10.19.tif" in url for file in units[0].files for url in file)


def test_mrms_pass_one_uses_only_its_own_product(tmp_path: Path) -> None:
    dataset = next(
        ds
        for ds in DYNAMICAL_DATASETS
        if ds.dataset_id == "noaa-mrms-conus-analysis-hourly"
    )
    source = TimestampSource(tmp_path, 1)
    units = list(
        expected_units(
            dataset, pd.Timestamp("2026-10-20"), pd.Timestamp("2026-10-20T01"), source
        )
    )
    source.close()
    assert [unit.tier for unit in units] == ["position", "pass 1 (exception)"]
    assert len(units[1].files) == 1
    assert all(
        "MultiSensor_QPE_01H_Pass1" in url for file in units[1].files for url in file
    )


def test_hrrr_generated_visibility_does_not_require_next_hour_before_its_gate(
    tmp_path: Path,
) -> None:
    dataset = next(
        ds for ds in DYNAMICAL_DATASETS if ds.dataset_id == "noaa-hrrr-analysis-virtual"
    )
    anchor = pd.Timestamp("2026-10-20T12")
    source = TimestampSource(tmp_path, 1)
    (unit,) = expected_units(dataset, anchor, anchor + pd.Timedelta(hours=1), source)
    source.close()
    assert len(unit.visibility_groups) == 1
    own_cycle = set(unit.visibility_groups[0])
    assert len(own_cycle) == 3
    assert len(unit.files) == 4
    objects = {}
    for file in unit.files:
        for url in file:
            objects[url] = anchor + pd.Timedelta(
                minutes=53 if file in own_cycle else -6
            )
    listings = {url.rsplit("/", 1)[0] + "/": Listing(objects, "S3") for url in objects}
    measured = aggregate_unit(unit, listings)
    assert measured.raw_first == -360
    assert measured.first == 3180
    result = recommend(
        [measured],
        RuntimeStats(30, None, 20, 60, 1, 1, 0, 0, None, None),
        virtual=True,
        interval=3600,
        currency_interval=3600,
        spacing=3600,
    )
    assert result.delay == 4320
    # Before 13Z's own-cycle gate, the previous pod must not require 13Z.
    assert (
        pd.Timestamp("2026-10-20T13:48") - pd.Timedelta(seconds=result.delay)
    ).floor("h") == anchor


def test_hrrr_materialized_analysis_includes_current_cycle_f01_boundary(
    tmp_path: Path,
) -> None:
    dataset = next(
        ds for ds in DYNAMICAL_DATASETS if ds.dataset_id == "noaa-hrrr-analysis"
    )
    source = TimestampSource(tmp_path, 1)
    (unit,) = expected_units(
        dataset, pd.Timestamp("2026-10-20T12"), pd.Timestamp("2026-10-20T13"), source
    )
    source.close()
    urls = {url for file in unit.files for url in file}
    assert len(urls) == 6
    assert not any("hrrr.t13z" in url for url in urls)
    assert {url.rsplit("/", 1)[1] for url in urls} == {
        "hrrr.t11z.wrfsfcf01.grib2",
        "hrrr.t11z.wrfsfcf01.grib2.idx",
        "hrrr.t12z.wrfsfcf00.grib2",
        "hrrr.t12z.wrfsfcf00.grib2.idx",
        "hrrr.t12z.wrfsfcf01.grib2",
        "hrrr.t12z.wrfsfcf01.grib2.idx",
    }
