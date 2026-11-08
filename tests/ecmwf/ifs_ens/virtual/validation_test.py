import json
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

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
from reformatters.common.virtual_region_job import VirtualRef
from reformatters.ecmwf.ifs_ens.virtual_region_job import (
    ifs_ens_virtual_chunk_containers,
)
from reformatters.ecmwf.ifs_ens.virtual_validation import (
    EcmwfIfsEnsVirtualManifestCompleteness,
)
from tests.ecmwf.ifs_ens.virtual.conftest import FIXTURES, get_var, make_job
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

    # The representative perturbed chunk hides a partial-control file from retries.
    assert job.filter_already_present([coord], session.store) == []
    repaired = [
        *refs,
        refs[0]._replace(out_loc={**refs[0].out_loc, "ensemble_member": 0}),
    ]
    assert len(repaired) == 3
    repaired_source = Mock(return_value=repaired)
    monkeypatch.setattr(type(job), "file_refs", repaired_source)
    pending = job.filter_already_present([coord], session.store)
    assert list(job.process_virtual_refs(pending)) == []
    repaired_source.assert_not_called()
    job._emit_refs([session.store], repaired)
    assert EcmwfIfsEnsVirtualManifestCompleteness().check(context).passed


def test_empty_candidate_window_fails(
    template: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(tmp_path)))
    job = make_job(template)
    monkeypatch.setattr(type(job), "source_file_coords", lambda _self: [])
    result = EcmwfIfsEnsVirtualManifestCompleteness().check(
        ValidationContext(
            store=repo.writable_session("main").store,
            ds=flatten_groups(template),
            append_dim="init_time",
            region_job=job,
        )
    )
    assert not result.passed
    assert result.checked_count == 0
    assert "No source files" in result.message


@pytest.mark.slow
def test_real_temperature_alias_decodes_with_configured_celsius_filter(
    template: xr.DataTree, tmp_path: Path
) -> None:
    init = pd.Timestamp("2024-09-17T18")
    var = get_var("temperature_2m")
    template = xr.DataTree.from_dict(
        {
            node.path: node.to_dataset()
            .assign_coords(
                init_time=node.to_dataset().init_time.copy(data=[init.to_datetime64()])
            )
            .isel(lead_time=[1])
            .sel(ensemble_member=[44])
            for node in template.subtree
        }
    )
    job = make_job(template, [var])
    (coord,) = job.source_file_coords()
    rows = [
        json.loads(line)
        for line in (FIXTURES / "20240917_18_3_temperature_aliases.index")
        .read_text()
        .splitlines()
    ]
    row = next(
        row for row in rows if row.get("number") == "44" and row["param"] == "mean2t"
    )
    config, credentials = _repository_config_and_credentials(
        IcechunkVirtualConfig(
            containers=ifs_ens_virtual_chunk_containers(fine_resolution=False),
            manifest_split=manifest_append_dim_split(split_size=1, dim="init_time"),
        )
    )
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "alias.icechunk")),
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
    job._emit_refs(
        [session.store],
        [
            VirtualRef(
                data_var=var,
                out_loc={**coord.out_loc(), "ensemble_member": 44},
                location=coord.get_url(),
                offset=row["_offset"],
                length=row["_length"],
            )
        ],
    )
    ds = xr.open_zarr(
        session.store, chunks=None, consolidated=False, decode_timedelta=True
    )
    value = ds.temperature_2m.sel(latitude=-5, longitude=-60).item()
    assert value == pytest.approx(36.296060180664085)


@pytest.mark.parametrize(
    ("positions", "passed", "count"), [(1, True, 3), ("all", False, 6)]
)
def test_exact_completeness_position_window(
    template: xr.DataTree,
    tmp_path: Path,
    positions: int | Literal["all"],
    passed: bool,
    count: int,
) -> None:
    inits = pd.to_datetime(["2024-09-18T18", "2024-09-19T00"])
    nodes = {}
    for node in template.subtree:
        ds = node.to_dataset().isel(
            init_time=[0, 0], lead_time=[0], ensemble_member=[0, 1, 50]
        )
        nodes[node.path] = ds.assign_coords(
            init_time=ds.init_time.copy(data=inits.values)
        )
    template = xr.DataTree.from_dict(nodes)
    var = get_var("pressure_surface")
    job = make_job(template, [var]).model_copy(update={"region": slice(0, 2)})
    config, credentials = _repository_config_and_credentials(
        IcechunkVirtualConfig(
            containers=ifs_ens_virtual_chunk_containers(fine_resolution=False),
            manifest_split=manifest_append_dim_split(split_size=1, dim="init_time"),
        )
    )
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path)),
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
    refs = [
        VirtualRef(
            data_var=var, out_loc=loc, location=coord.get_url(), offset=0, length=1
        )
        for coord in job.source_file_coords()
        for loc in job.message_locations(coord, var)
        if coord.init_time == inits[-1] or loc["ensemble_member"] != 0
    ]
    job._emit_refs([session.store], refs)
    result = EcmwfIfsEnsVirtualManifestCompleteness(positions=positions).check(
        ValidationContext(
            store=session.store,
            ds=flatten_groups(template),
            append_dim="init_time",
            region_job=job,
        )
    )
    assert result.passed == passed
    assert result.checked_count == count
    if not passed:
        assert "1/3 message references missing" in result.message
