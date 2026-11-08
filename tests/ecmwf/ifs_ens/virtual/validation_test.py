from pathlib import Path

import icechunk
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import template_utils
from reformatters.common.iterating import flatten_groups
from reformatters.common.storage import (
    IcechunkVirtualConfig,
    _repository_config_and_credentials,
    manifest_append_dim_split,
)
from reformatters.common.validation import ValidationContext
from reformatters.ecmwf.ifs_ens.virtual_region_job import (
    ifs_ens_virtual_chunk_containers,
)
from reformatters.ecmwf.ifs_ens.virtual_validation import (
    EcmwfIfsEnsVirtualManifestCompleteness,
)
from tests.ecmwf.ifs_ens.virtual.conftest import get_var, make_job
from tests.ecmwf.ifs_ens.virtual.region_job_test import coordinates, serve_fixture


def test_missing_control_messages_fail_completeness_while_perturbed_data_ingests(
    template: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init = pd.Timestamp("2024-09-18T18")
    nodes = {
        node.path: node.to_dataset()
        .assign_coords(
            init_time=node.to_dataset().init_time.copy(data=[init.to_datetime64()])
        )
        .isel(lead_time=[0], ensemble_member=[0, 1, 50])
        for node in template.subtree
    }
    template = xr.DataTree.from_dict(nodes)
    var = get_var("pressure_surface")
    job = make_job(template, [var])
    (coord,) = job.generate_source_file_coords(coordinates(str(init), [0]), [var])
    serve_fixture(monkeypatch, tmp_path, "20240918_18_0_enfo_0p25.index")
    refs = job._file_refs_or_skip(coord, 10_000_000_000)
    assert len(refs) == 2
    config, credentials = _repository_config_and_credentials(
        IcechunkVirtualConfig(
            containers=ifs_ens_virtual_chunk_containers(fine_resolution=False),
            manifest_split=manifest_append_dim_split(split_size=1, dim="init_time"),
        )
    )
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "icechunk")),
        config=config,
        authorize_virtual_chunk_access=credentials,
    )
    session = repo.writable_session("main")
    template_utils.write_metadata(
        template,
        session.store,
        mode="w-",
        skip_icechunk_commit=True,
        consolidated=False,
    )
    job._emit_refs([session.store], refs)
    context = ValidationContext(
        store=session.store,
        ds=flatten_groups(template),
        append_dim="init_time",
        data_vars=[var],
        region_job=job,
    )
    result = EcmwfIfsEnsVirtualManifestCompleteness().check(context)
    assert not result.passed
    assert "1/3 message references missing" in result.message
