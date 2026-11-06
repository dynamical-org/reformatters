from pathlib import Path

import icechunk
import numpy as np
import pytest
import xarray as xr
import zarr

from reformatters.common.storage import StoreFactory
from reformatters.common.template_utils import write_metadata
from tests.common.virtual_region_job_test import (
    VirtualTestRegionJob,
    _create_template_ds,
    _make_dataset,
    _make_region_job,
    _primary_repo,
)


def _setup(
    tmp_path: Path, *, positions: int = 3, size: int = 4, grouped: bool = False
) -> tuple[icechunk.Repository, VirtualTestRegionJob, StoreFactory]:
    dataset = _make_dataset(tmp_path, n_inits=size + 1)
    ds = _create_template_ds(size + 1).to_dataset()
    ds = ds.assign_coords(
        ingested_forecast_length=("init_time", np.full(size + 1, np.nan))
    )
    ds.ingested_forecast_length.encoding = {
        "dtype": "float64",
        "fill_value": np.nan,
        "chunks": (8,),
    }
    if grouped:
        ds["temperature_2m"] = ds.temperature_2m.transpose(
            "latitude", "init_time", "lead_time", "longitude"
        )
        ds.temperature_2m.encoding["chunks"] = (2, 1, 1, 3)
        ds["omitted"] = ds.temperature_2m.copy(deep=False)
        ds["native"] = xr.Variable(
            ("lead_time", "init_time"),
            np.arange(2 * (size + 1)).reshape(2, -1),
            encoding={"dtype": "int64", "fill_value": -1, "chunks": (1, 1)},
        )
        for coord in ds.coords.values():
            if "init_time" in coord.dims:
                coord.encoding["shards"] = tuple(
                    2 * n for n in coord.encoding["chunks"]
                )
    template = xr.DataTree.from_dict(
        {"/": ds, **({"/pressure_level": ds.copy(deep=False)} if grouped else {})}
    )
    factory = dataset.store_factory
    write_metadata(template.isel(init_time=slice(0, size)), factory, consolidated=False)
    repo = _primary_repo(factory)
    job = _make_region_job(
        template, region=slice(0, size), processing_mode="update"
    ).model_copy(update={"rolling_positions": positions})
    session = repo.writable_session("main")
    names = ["temperature_2m", "omitted"] if grouped else ["temperature_2m"]
    for node in template.subtree:
        prefix = node.path.strip("/")
        for batch in job.process_virtual_refs(job.source_file_coords()):
            for _, refs in batch:
                for ref in refs:
                    for name in names:
                        var = ref.data_var.model_copy(update={"name": name})
                        key = job.chunk_key(ref.out_loc, var)
                        assert key is not None
                        session.store.set_virtual_ref(
                            f"{prefix}/{name}/c/".lstrip("/") + "/".join(map(str, key)),
                            ref.location,
                            offset=ref.offset,
                            length=ref.length,
                        )
        zarr.open_array(
            session.store,
            path=f"{prefix}/ingested_forecast_length".lstrip("/"),
            mode="r+",
        )[:] = np.arange(size) + 1
        if grouped:
            zarr.open_array(
                session.store, path=f"{prefix}/native".lstrip("/"), mode="r+"
            )[:] = ds.native.values[:, :size]
    if session.has_uncommitted_changes:
        session.commit("Initialize retained window")
    return repo, job, factory


def _open(repo: icechunk.Repository) -> xr.DataTree:
    return xr.open_datatree(
        repo.readonly_session("main").store,  # ty: ignore[invalid-argument-type]
        engine="zarr",
        chunks=None,
        decode_timedelta=True,
        consolidated=False,
    )


@pytest.mark.parametrize("positions", [1, 12])
def test_generic_trim_preserves_labels_virtual_refs_native_data_and_coordinates(
    tmp_path: Path, positions: int
) -> None:
    repo, job, _ = _setup(
        tmp_path, positions=positions, size=positions + 2, grouped=True
    )
    template = job.template_ds.map_over_datasets(
        lambda ds: ds.drop_vars(["omitted", "native"])
    )
    job = job.model_copy(update={"template_ds": template})
    before = _open(repo).load()
    source = tmp_path / "messages.bin"
    payload = source.read_bytes()
    source.unlink()
    job.trim_retention(repo, "main")
    source.write_bytes(payload)
    xr.testing.assert_equal(_open(repo), before.isel(init_time=slice(2, None)))
    session = repo.readonly_session("main")
    for prefix in ("", "pressure_level/"):
        for name in ("temperature_2m", "omitted"):
            assert (
                session.chunk_type(f"/{prefix}{name}", [0, 0, 0, 0])
                == icechunk.ChunkType.virtual
            )
    snapshot = repo.lookup_branch("main")
    job.trim_retention(repo, "main")
    assert repo.lookup_branch("main") == snapshot
