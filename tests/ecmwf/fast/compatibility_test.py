from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.typing import NDArray
from zarr.codecs import BytesCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from reformatters.common import template_utils
from reformatters.common.pydantic import replace
from reformatters.common.storage import StoreFactory
from reformatters.common.virtual_region_job import VirtualRef
from reformatters.ecmwf.aifs_ens.forecast_fast.region_job import (
    EcmwfAifsEnsForecastFastRegionJob,
)
from reformatters.ecmwf.aifs_ens.forecast_fast.template_config import (
    EcmwfAifsEnsForecastFastTemplateConfig,
)
from reformatters.ecmwf.fast.region_job import EcmwfFastRegionJob
from reformatters.ecmwf.ifs_ens.forecast_0_25_degree_fast.region_job import (
    EcmwfIfsEnsForecast025DegreeFastRegionJob,
)
from reformatters.ecmwf.ifs_ens.forecast_0_25_degree_fast.template_config import (
    EcmwfIfsEnsForecast025DegreeFastTemplateConfig,
)
from tests.common.template_refresh_concurrency_test import _factory
from tests.common.virtual_retention_test import _open


def _setup_seven_variable_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cls: type[EcmwfFastRegionJob]
) -> tuple[icechunk.Repository, EcmwfFastRegionJob, StoreFactory, NDArray[np.float64]]:
    config = (
        EcmwfIfsEnsForecast025DegreeFastTemplateConfig()
        if cls.model == "ifs"
        else EcmwfAifsEnsForecastFastTemplateConfig()
    )
    template = (
        config.get_template(pd.Timestamp("2026-11-05T01:00"))
        .sel(
            init_time=slice("2026-11-04", None),
            lead_time=[pd.Timedelta("6h")],
            ensemble_member=[0],
        )
        .isel(latitude=slice(0, 2), longitude=slice(0, 3))
        .to_dataset()
    )
    variables = [
        replace(
            v,
            encoding=replace(
                v.encoding, chunks=(1, 1, 1, 2, 3), serializer=BytesCodec().to_dict()
            ),
        )
        for v in config.data_vars
    ]
    for var in variables:
        template_utils.assign_var_metadata(template[var.name], var)
    names = [v.name for v in variables]
    old_template = template.drop_vars(names[7:])
    source = tmp_path / "fields.bin"
    values = np.arange(5 * 9 * 6, dtype="float64").reshape(5, 9, 2, 3)
    source.write_bytes(values.tobytes())
    repo_config = icechunk.RepositoryConfig.default()
    repo_config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            tmp_path.as_uri() + "/", icechunk.local_filesystem_store(str(tmp_path))
        )
    )
    repo = icechunk.Repository.create(
        icechunk.in_memory_storage(),
        config=repo_config,
        authorize_virtual_chunk_access=icechunk.containers_credentials(
            {tmp_path.as_uri() + "/": icechunk.credentials.LocalFileSystemAccess}
        ),
    )
    factory, _ = _factory(monkeypatch, repo)
    template_utils.write_metadata(
        xr.DataTree(old_template.isel(init_time=slice(0, 4))),
        repo.writable_session("main").store,
        mode="w-",
        consolidated=False,
    )
    job = cls(
        tmp_store=tmp_path / "work.zarr",
        template_ds=xr.DataTree(template),
        data_vars=variables,
        append_dim="init_time",
        region=slice(0, 5),
        reformat_job_name="test-add-variables",
        processing_mode="update",
        rolling_positions=3,
    )
    old_job = job.model_copy(
        update={"template_ds": xr.DataTree(old_template), "data_vars": variables[:7]}
    )
    coords = list(job.source_file_coords())
    assert len(coords) == 5
    old_batch = [
        (
            coord,
            [
                VirtualRef(var, coord.out_loc(), source.as_uri(), (i * 9 + j) * 48, 48)
                for j, var in enumerate(variables[:7])
            ],
        )
        for i, coord in enumerate(coords[:4])
    ]
    old_job._commit_virtual_batch(repo, "main", old_batch, overwrite_chunks=False)
    old_job.trim_rolling_window(repo, "main")
    return repo, job, factory, values


@pytest.mark.parametrize(
    "cls",
    [EcmwfIfsEnsForecast025DegreeFastRegionJob, EcmwfAifsEnsForecastFastRegionJob],
)
def test_add_variables_at_shifted_origin_then_append_and_trim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cls: type[EcmwfFastRegionJob]
) -> None:
    repo, job, factory, values = _setup_seven_variable_window(
        tmp_path, monkeypatch, cls
    )
    variables = job.data_vars
    names = [v.name for v in variables]
    coords = list(job.source_file_coords())
    source = tmp_path / "fields.bin"
    before = _open(repo).to_dataset().load()
    assert before.init_time.values[0] == np.datetime64("2026-11-04T06:00")
    old_snapshot = repo.readonly_session("main")
    job.refresh_metadata(factory, tmp_path / "refresh.zarr")
    refreshed = _open(repo).to_dataset().load()
    xr.testing.assert_equal(refreshed.drop_vars(names[7:]), before)
    for var in variables[:7]:
        key = f"{var.path}/zarr.json"
        old_metadata = sync(old_snapshot.store.get(key, default_buffer_prototype()))
        refreshed_metadata = sync(
            repo.readonly_session("main").store.get(key, default_buffer_prototype())
        )
        assert old_metadata is not None
        assert refreshed_metadata is not None
        assert old_metadata.to_bytes() == refreshed_metadata.to_bytes()
        assert (
            old_snapshot.chunk_type(f"/{var.path}", [0, 0, 0, 0, 0])
            == icechunk.ChunkType.virtual
        )
    for name in names[7:]:
        assert refreshed[name].isnull().all()
        assert np.isnan(refreshed[name].encoding["_FillValue"])
        assert np.isnan(refreshed[name].encoding["fill_value"])
    assert job.filter_already_present(
        coords[:4], repo.readonly_session("main").store
    ) == [coords[0]]
    refs = [
        VirtualRef(var, coords[4].out_loc(), source.as_uri(), (4 * 9 + j) * 48, 48)
        for j, var in enumerate(variables)
    ]
    job._commit_virtual_batch(repo, "main", [(coords[4], refs)], overwrite_chunks=False)
    job.trim_rolling_window(repo, "main")
    after = _open(repo).to_dataset().load()
    np.testing.assert_array_equal(after.init_time, job.template_ds.init_time[2:])
    xr.testing.assert_equal(
        after.drop_vars(names[7:]).isel(init_time=slice(0, 2)),
        before.isel(init_time=slice(1, None)),
    )
    for j, var in enumerate(variables):
        expected = values[4, j]
        if var.attrs.short_name in ("2t", "2d"):
            expected = expected - 273.15
        elif var.attrs.short_name == "tp" and cls.model == "ifs":
            expected = expected / 0.001
        np.testing.assert_array_equal(
            after[var.name].isel(init_time=-1).values.reshape(2, 3), expected
        )
        assert (
            repo.readonly_session("main").chunk_type(f"/{var.path}", [2, 0, 0, 0, 0])
            == icechunk.ChunkType.virtual
        )
    for name in names[7:]:
        assert after[name].isel(init_time=slice(0, 2)).isnull().all()
