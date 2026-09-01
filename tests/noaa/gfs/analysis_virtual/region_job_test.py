from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import xarray as xr

from reformatters.noaa.gfs.analysis_virtual.region_job import (
    NoaaGfsAnalysisVirtualRegionJob,
    NoaaGfsAnalysisVirtualSourceFileCoord,
)
from reformatters.noaa.gfs.analysis_virtual.template_config import (
    NoaaGfsAnalysisVirtualTemplateConfig,
)
from reformatters.noaa.gfs.virtual_region_job import NoaaGfsVirtualRegionJob
from reformatters.noaa.models import NoaaDataVar

TEMPLATE_CONFIG = NoaaGfsAnalysisVirtualTemplateConfig()


def get_var(path: str) -> NoaaDataVar:
    return next(v for v in TEMPLATE_CONFIG.data_vars if v.path == path)


@pytest.fixture(scope="module")
def template_ds() -> xr.DataTree:
    return TEMPLATE_CONFIG.get_template(pd.Timestamp("2021-03-24T12:00"))


def make_job(
    template_ds: xr.DataTree,
    data_vars: Sequence[NoaaDataVar],
    region: slice = slice(0, 1),
) -> NoaaGfsAnalysisVirtualRegionJob:
    return NoaaGfsAnalysisVirtualRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=data_vars,
        append_dim="time",
        region=region,
        reformat_job_name="test",
    )


def coords_for_times(
    template_ds: xr.DataTree,
    data_vars: Sequence[NoaaDataVar],
    times: Sequence[pd.Timestamp],
) -> list[NoaaGfsAnalysisVirtualSourceFileCoord]:
    job = make_job(template_ds, data_vars=list(data_vars))
    region_ds = xr.Dataset(coords={"time": pd.DatetimeIndex(list(times))})
    return list(job.generate_source_file_coords(region_ds, list(data_vars)))


def test_every_hour_of_a_day_takes_the_shortest_published_lead(
    template_ds: xr.DataTree,
) -> None:
    """A windowed variable's window must open at the synoptic hour STRICTLY before its
    time, so at 00, 06, 12 and 18 UTC it reads lead 6 of the previous cycle, not lead 0
    of its own. Enumerated over a whole day because that is the only place the
    off-by-one shows.
    """
    instant = get_var("temperature_2m")
    windowed = get_var("total_precipitation_surface")
    day = list(pd.date_range("2021-03-24T00:00", periods=24, freq="1h"))
    assert len(day) == 24

    coords = coords_for_times(template_ds, [instant, windowed], day)
    placement = {
        (var.name, coord.valid_time()): (coord.init_time, coord.lead_time)
        for coord in coords
        if coord.file_type == "pgrb2"
        for var in coord.data_vars
    }

    assert len(placement) == 2 * len(day)
    for time in day:
        cycle = time.floor("6h")
        assert placement[(instant.name, time)] == (cycle, time - cycle), time
        previous_cycle = (time - pd.Timedelta("1h")).floor("6h")
        assert placement[(windowed.name, time)] == (
            previous_cycle,
            time - previous_cycle,
        ), time
        assert pd.Timedelta("1h") <= time - previous_cycle <= pd.Timedelta("6h"), time


def test_both_products_are_read_for_every_time(template_ds: xr.DataTree) -> None:
    """Away from a synoptic hour the two variable sets share a file, so a time costs two
    coords; at a synoptic hour they come from different cycles and it costs four."""
    data_vars = [get_var("temperature_2m"), get_var("total_precipitation_surface")]

    for time, expected in (
        (pd.Timestamp("2021-03-24T07:00"), 2),
        (pd.Timestamp("2021-03-24T12:00"), 4),
    ):
        coords = coords_for_times(template_ds, data_vars, [time])
        assert len(coords) == expected, time
        assert {c.file_type for c in coords} == {"pgrb2", "pgrb2b"}, time


def test_the_five_instant_variables_absent_at_hour_0_are_read_at_a_longer_lead() -> (
    None
):
    """GFS publishes no windowed message at f000, and also drops five instantaneous
    ones. Nine other instantaneous variables share an element with a windowed sibling
    and ARE published there, so an hour-0 rule keyed on the element would wrongly drop
    them.
    """
    absent_at_hour_0 = {
        "potential_evaporation_rate_surface",
        "instantaneous_precipitation_convective_surface",
        "pressure_convective_cloud_bottom",
        "pressure_convective_cloud_top",
        "convective_cloud_cover",
    }
    present_at_hour_0 = {
        "instantaneous_categorical_snow_surface",
        "instantaneous_categorical_rain_surface",
        "instantaneous_total_cloud_cover_atmosphere",
        "precipitation_rate_surface",
        "low_cloud_cover",
        "medium_cloud_cover",
        "high_cloud_cover",
    }
    instant_without_hour_0 = {
        var.name
        for var in TEMPLATE_CONFIG.data_vars
        if var.attrs.step_type == "instant" and not var.has_hour_0_values()
    }
    assert instant_without_hour_0 == absent_at_hour_0
    assert all(get_var(name).has_hour_0_values() for name in present_at_hour_0)


def test_representative_var_is_carried_only_by_its_own_product(
    template_ds: xr.DataTree,
) -> None:
    """A probe on a variable the file does not fill would never be marked ingested."""
    data_vars = TEMPLATE_CONFIG.data_vars
    coords = coords_for_times(
        template_ds, data_vars, [pd.Timestamp("2021-03-24T12:00")]
    )
    job = make_job(template_ds, data_vars=list(data_vars))

    picked = {
        (c.file_type, c.data_vars[0].has_hour_0_values()): job.representative_var(
            c
        ).name
        for c in coords
    }
    assert picked == {
        ("pgrb2", True): "temperature_2m",
        ("pgrb2", False): "total_precipitation_surface",
        ("pgrb2b", True): "temperature_305m_amsl",
        ("pgrb2b", False): "uv_b_downward_solar_flux_surface",
    }
    for coord in coords:
        var = job.representative_var(coord)
        assert var in coord.data_vars
        assert dict(job.representative_probe_loc(coord, var)) == {
            "time": coord.valid_time()
        }


def test_operational_update_jobs_single_polling_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2021-03-24T12:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *a, **kw: now))

    jobs, template_ds = NoaaGfsAnalysisVirtualRegionJob.operational_update_jobs(
        primary_store=Mock(),
        tmp_store=Path("unused-tmp.zarr"),
        get_template_fn=TEMPLATE_CONFIG.get_template,
        append_dim="time",
        all_data_vars=TEMPLATE_CONFIG.data_vars,
        reformat_job_name="test",
    )

    (job,) = jobs
    assert isinstance(job, NoaaGfsAnalysisVirtualRegionJob)
    assert job.processing_mode == "update"
    times = template_ds.to_dataset().get_index("time")
    assert job.region == slice(len(times) - 12, len(times))


def discover(
    job: NoaaGfsAnalysisVirtualRegionJob,
    pending: Sequence[NoaaGfsAnalysisVirtualSourceFileCoord],
    published: Sequence[NoaaGfsAnalysisVirtualSourceFileCoord],
    monkeypatch: pytest.MonkeyPatch,
) -> list[pd.Timestamp]:
    """Run the gate over `pending` with `published` listed by the source."""
    monkeypatch.setattr(
        NoaaGfsVirtualRegionJob,
        "discover_available",
        lambda self, pending: [(c, 100) for c in pending if c in published],
    )
    return sorted(
        coord.valid_time() for coord, _ in job.discover_available(list(pending))
    )


def gate_setup(
    template_ds: xr.DataTree, times: Sequence[pd.Timestamp]
) -> tuple[
    NoaaGfsAnalysisVirtualRegionJob, list[NoaaGfsAnalysisVirtualSourceFileCoord]
]:
    data_vars = [get_var("temperature_2m"), get_var("total_precipitation_surface")]
    return (
        make_job(template_ds, data_vars=data_vars),
        coords_for_times(template_ds, data_vars, times),
    )


def test_a_synoptic_hour_waits_for_the_cycle_that_starts_at_it(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At 12 UTC the windowed files come from the 06 UTC cycle and publish about six
    hours before the 12 UTC cycle's own files."""
    time = pd.Timestamp("2021-03-24T12:00")
    job, coords = gate_setup(template_ds, [time])
    job = job.model_copy(update={"ingested_through": time - pd.Timedelta("1h")})
    from_previous_cycle = [c for c in coords if c.init_time < time]
    assert len(from_previous_cycle) == 2

    assert discover(job, coords, from_previous_cycle, monkeypatch) == []
    assert discover(job, coords, coords, monkeypatch) == [time] * 4


def test_only_the_time_holding_every_file_extends_the_append_dim(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    times = [pd.Timestamp("2021-03-24T11:00"), pd.Timestamp("2021-03-24T12:00")]
    job, coords = gate_setup(template_ds, times)
    job = job.model_copy(update={"ingested_through": times[0] - pd.Timedelta("1h")})
    published = [
        c for c in coords if c.valid_time() == times[0] or c.init_time < times[1]
    ]

    assert discover(job, coords, published, monkeypatch) == [times[0]] * 2


def test_a_time_the_store_already_covers_is_never_withheld(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file the archive never published must not block the files beside it forever."""
    time = pd.Timestamp("2021-03-24T12:00")
    job, coords = gate_setup(template_ds, [time])
    job = job.model_copy(update={"ingested_through": time})
    from_previous_cycle = [c for c in coords if c.init_time < time]

    assert discover(job, coords, from_previous_cycle, monkeypatch) == [time] * 2


def test_an_empty_store_waits_for_a_whole_time(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    time = pd.Timestamp("2021-03-24T12:00")
    job, coords = gate_setup(template_ds, [time])
    assert job.ingested_through is None

    assert (
        discover(job, coords, [c for c in coords if c.init_time < time], monkeypatch)
        == []
    )
