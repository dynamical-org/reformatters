from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import pytest
import zarr

from reformatters.common import template_utils, validation
from reformatters.google.weathernext2.forecast_operational_virtual.template_config import (
    GoogleWeathernext2ForecastOperationalVirtualTemplateConfig,
)
from reformatters.google.weathernext2.forecast_virtual.region_job import (
    GoogleWeathernext2ForecastOperationalVirtualRegionJob,
)
from reformatters.google.weathernext2.forecast_virtual.validation import (
    CheckNoRefsInsideHoldback,
)

OPERATIONAL = GoogleWeathernext2ForecastOperationalVirtualTemplateConfig()


def _job_and_repo() -> tuple[
    GoogleWeathernext2ForecastOperationalVirtualRegionJob, icechunk.Repository
]:
    template = OPERATIONAL.get_template(pd.Timestamp("2025-03-02T18:00"))
    data_vars = [
        var
        for var in OPERATIONAL.data_vars
        if var.path in ("temperature_2m", "pressure_level/temperature")
    ]
    n_inits = template.to_dataset().sizes["init_time"]
    job = GoogleWeathernext2ForecastOperationalVirtualRegionJob(
        tmp_store=Path("unused.zarr"),
        template_ds=template,
        data_vars=data_vars,
        append_dim="init_time",
        region=slice(n_inits - 7, n_inits),
        reformat_job_name="test",
        publication_cutoff=pd.Timestamp("2025-03-02T12:00"),
    )
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    session = repo.writable_session("main")
    template_utils.write_metadata(
        template, session.store, "w-", consolidated=False, skip_icechunk_commit=True
    )
    session.commit("template")
    return job, repo


def _write_chunk(
    job: GoogleWeathernext2ForecastOperationalVirtualRegionJob,
    repo: icechunk.Repository,
    path: str,
    init_time: str,
    lead_time: str,
) -> None:
    template = job.template_ds.to_dataset()
    init_index = template.get_index("init_time").get_loc(pd.Timestamp(init_time))
    lead_index = template.get_index("lead_time").get_loc(pd.Timedelta(lead_time))
    assert isinstance(init_index, int)
    assert isinstance(lead_index, int)
    index: tuple[int | slice, ...] = (
        init_index,
        0,
        lead_index,
        slice(None),
        slice(None),
    )
    if path.startswith("pressure_level/"):
        index = (*index, 0)
    session = repo.writable_session("main")
    array = zarr.open_group(session.store, mode="r+")[path]
    assert isinstance(array, zarr.Array)
    array[index] = np.ones(array.chunks[-2:], dtype=array.dtype)
    session.commit(f"write {path} {init_time} {lead_time}")


def _run_validator(
    job: GoogleWeathernext2ForecastOperationalVirtualRegionJob,
    repo: icechunk.Repository,
) -> None:
    validation.validate_dataset(
        [CheckNoRefsInsideHoldback()],
        store=repo.readonly_session("main").store,
        append_dim="init_time",
        dataset_id="test",
        region_job=job,
    )


def test_held_back_source_file_coords_complement_publishable_steps() -> None:
    job, _ = _job_and_repo()
    published = {
        (coord.init_time, coord.lead_time, coord.data_vars[0].path)
        for coord in job.source_file_coords()
    }
    held_back = {
        (coord.init_time, coord.lead_time, coord.data_vars[0].path)
        for coord in job.held_back_source_file_coords()
    }

    assert published.isdisjoint(held_back)
    assert all(init + lead <= job.publication_cutoff for init, lead, _ in published)
    assert all(init + lead > job.publication_cutoff for init, lead, _ in held_back)
    assert len(published) + len(held_back) == 7 * 60 * 2
    assert len(held_back) == len(job.held_back_source_file_coords())
    boundary = (
        pd.Timestamp("2025-03-01T18:00"),
        pd.Timedelta("18h"),
        "temperature_2m",
    )
    assert boundary in published
    assert (boundary[0], pd.Timedelta("24h"), boundary[2]) in held_back


def test_check_no_refs_inside_holdback_detects_representative_ref() -> None:
    job, repo = _job_and_repo()

    _run_validator(job, repo)

    _write_chunk(job, repo, "pressure_level/temperature", "2025-03-02T12:00", "6h")
    with pytest.raises(
        validation.OperationalValidationError,
        match=r"1 of \d+ steps with a valid time after the publication cutoff 2025-03-02 12:00:00",
    ) as excinfo:
        _run_validator(job, repo)
    assert (
        "init_time=2025-03-02 12:00:00 lead_time=0 days 06:00:00 "
        "pressure_level/temperature"
    ) in str(excinfo.value)

    _write_chunk(job, repo, "temperature_2m", "2025-03-01T18:00", "18h")
    with pytest.raises(validation.OperationalValidationError, match=r"1 of \d+ steps"):
        _run_validator(job, repo)


def test_check_no_refs_inside_holdback_uses_store_extent() -> None:
    job, repo = _job_and_repo()
    template = job.template_ds.to_dataset()
    short_end = template.get_index("init_time")[-3]
    short = job.template_ds.isel(init_time=slice(0, -2))
    short_repo = icechunk.Repository.create(icechunk.in_memory_storage())
    session = short_repo.writable_session("main")
    template_utils.write_metadata(
        short, session.store, "w-", consolidated=False, skip_icechunk_commit=True
    )
    session.commit("template")
    context = validation.ValidationContext(
        store=short_repo.readonly_session("main").store,
        ds=validation.open_flattened_dataset(
            short_repo.readonly_session("main").store, consolidated=False
        ),
        append_dim="init_time",
        region_job=job,
    )

    result = CheckNoRefsInsideHoldback().check(context)

    assert result.passed
    n_candidates = sum(
        coord.init_time <= short_end for coord in job.held_back_source_file_coords()
    )
    assert result.checked_count == n_candidates
    assert f"None of the {n_candidates} steps" in result.message

    long_job = job.model_copy(
        update={"template_ds": job.template_ds.isel(init_time=slice(0, -1))}
    )
    context = validation.ValidationContext(
        store=repo.readonly_session("main").store,
        ds=validation.open_flattened_dataset(
            repo.readonly_session("main").store, consolidated=False
        ),
        append_dim="init_time",
        region_job=long_job,
    )

    result = CheckNoRefsInsideHoldback().check(context)

    assert not result.passed
    assert "past the update job's template" in result.message
