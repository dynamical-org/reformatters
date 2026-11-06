from pathlib import Path

import pytest
import zarr.core.sync

from reformatters.common import validation
from tests.common.virtual_region_job_test import (
    APPEND_DIM_FREQ,
    APPEND_DIM_START,
    _virtual_context,
)
from tests.common.virtual_retention_test import _open, _setup


@pytest.mark.parametrize("oldest_missing", [False, True])
def test_completeness_checks_retained_lower_bound_beyond_hot_region(
    tmp_path: Path, oldest_missing: bool
) -> None:
    repo, job, _ = _setup(tmp_path)
    job.trim_retention(repo, "main")
    if oldest_missing:
        session = repo.writable_session("main")
        zarr.core.sync.sync(session.store.delete("temperature_2m/c/0/0/0/0"))
        session.commit("Remove oldest retained reference")
    job = job.model_copy(update={"region": slice(3, 4)})
    context = _virtual_context(
        job, repo.readonly_session("main").store, _open(repo).to_dataset()
    )
    mapper = context.virtual_region_job()
    assert mapper.rolling_positions is None
    assert mapper.region == slice(0, 4)
    assert mapper.template_ds.init_time.values[0] == APPEND_DIM_START + APPEND_DIM_FREQ
    assert job.region == slice(3, 4)
    result = validation.CheckVirtualManifestCompleteness().check(context)
    assert result.passed is not oldest_missing, result.message
    assert (
        str(APPEND_DIM_START + APPEND_DIM_FREQ) in result.message
        if oldest_missing
        else "All 3 ingested" in result.message
    )
