from pathlib import Path

import icechunk
import obstore.store
import pandas as pd
import pytest
import xarray as xr
import zarr

from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.ecmwf import ecmwf_virtual_source as source_module
from reformatters.ecmwf.ifs_ens.forecast_6_day_0_1_degree_virtual.dynamical_dataset import (
    EcmwfIfsEnsForecast6Day01DegreeVirtualDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_1_degree_virtual.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day01DegreeVirtualDataset,
)
from reformatters.ecmwf.ifs_ens.virtual_dynamical_dataset import (
    EcmwfIfsEnsForecastVirtualDataset,
)
from reformatters.ecmwf.ifs_ens.virtual_region_job import (
    EcmwfIfsEnsForecastVirtualSourceFileCoord,
)


@pytest.fixture(
    params=[
        EcmwfIfsEnsForecast6Day01DegreeVirtualDataset,
        EcmwfIfsEnsForecast15Day01DegreeVirtualDataset,
    ]
)
def dataset(
    request: pytest.FixtureRequest, tmp_path: Path
) -> EcmwfIfsEnsForecastVirtualDataset:
    return request.param(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )


def test_persisted_fine_containers_match_catalog_without_changing_snapshot_or_references(
    dataset: EcmwfIfsEnsForecastVirtualDataset,
) -> None:
    factory = dataset.store_factory
    _, _, storage = factory._icechunk_storages()[0]
    original_config = icechunk.RepositoryConfig.default()
    original_config.clear_virtual_chunk_containers()
    original_config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer("gs://ecmwf-open-data/", icechunk.gcs_store())
    )
    repo = icechunk.Repository.create(storage, config=original_config)
    session = repo.writable_session("main")
    group = zarr.open_group(session.store, mode="w")
    group.create_array("values", shape=(1,), dtype="i4")[:] = [42]
    group.create_array("virtual", shape=(1,), dtype="f4", compressors=None)
    location = "gs://ecmwf-open-data/example.grib2"
    session.store.set_virtual_ref("virtual/c/0", location, offset=16, length=4)
    snapshot = session.commit("initial GCS references")

    factory.icechunk_repos(sort="primary-first")
    before = icechunk.Repository.fetch_config(storage)
    assert before is not None
    assert set(before.virtual_chunk_containers or ()) == {"gs://ecmwf-open-data/"}
    factory.persist_virtual_config()

    reader = icechunk.Repository.open(storage)
    assert set(reader.config.virtual_chunk_containers or ()) == {
        "gs://ecmwf-open-data/",
        "s3://ecmwf-forecasts/",
    }
    assert reader.lookup_branch("main") == snapshot
    readonly = reader.readonly_session("main")
    assert list(readonly.all_virtual_chunk_locations()) == [location]
    assert zarr.open_array(readonly.store, path="values", mode="r")[0] == 42


@pytest.mark.parametrize("gcs_present", [True, False])
def test_fine_discovery_never_switches_to_declared_s3_container(
    dataset: EcmwfIfsEnsForecastVirtualDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gcs_present: bool,
) -> None:
    coord = EcmwfIfsEnsForecastVirtualSourceFileCoord(
        init_time=pd.Timestamp("2026-11-06"),
        lead_time=pd.Timedelta("3h"),
        data_vars=[],
        resolution="0p1",
        family="oper",
        level_family="sfc",
        ensemble_members=(0,),
        pressure_levels=(),
    )
    calls: list[str] = []

    def listing(
        pending: list[EcmwfIfsEnsForecastVirtualSourceFileCoord],
        *,
        store: obstore.store.ObjectStore,
        location_prefix: str,
        require_index: bool,
    ) -> list[tuple[EcmwfIfsEnsForecastVirtualSourceFileCoord, int]]:
        calls.append(location_prefix)
        return (
            [(p, 100) for p in pending]
            if gcs_present or location_prefix.startswith("s3")
            else []
        )

    monkeypatch.setattr(source_module, "discover_available_by_obstore_listing", listing)
    job = dataset.region_job_class(
        tmp_store=tmp_path,
        template_ds=xr.DataTree(),
        data_vars=[],
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
    )
    available = job.discover_available([coord])

    assert calls == ["gs://ecmwf-open-data/"]
    assert available == ([(coord, 100)] if gcs_present else [])
    assert coord.get_url().startswith("gs://ecmwf-open-data/")
