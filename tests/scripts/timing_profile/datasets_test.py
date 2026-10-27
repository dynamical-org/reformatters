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
from scripts.timing_profile.sources import TimestampSource


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
