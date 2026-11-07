import json
from pathlib import Path

import icechunk
import numpy as np
import obstore
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.pydantic import replace
from reformatters.common.storage import (
    DatasetFormat,
    IcechunkVirtualConfig,
    StorageConfig,
)
from reformatters.common.types import DatetimeLike
from reformatters.ecmwf.aifs_ens.forecast_fast import EcmwfAifsEnsForecastFastDataset
from reformatters.ecmwf.fast import region_job
from reformatters.ecmwf.fast.region_job import EcmwfFastRegionJob
from reformatters.ecmwf.fast.source import EcmwfFastSourceFileCoord
from reformatters.ecmwf.fast.template_config import EcmwfFastTemplateConfig, FastModel
from reformatters.ecmwf.ifs_ens.forecast_0_25_degree_fast import (
    EcmwfIfsEnsForecast025DegreeFastDataset,
)

FIXTURES = Path(__file__).parent / "fixtures"
VARIABLES = ("temperature_2m",)


@pytest.mark.parametrize("model", ["ifs", "aifs"])
def test_fixture_backfill_then_update_with_native_codec_snapshots(
    model: FastModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORKER_MEMORY_REQUEST_BYTES", "7000000000")
    cls = (
        EcmwfIfsEnsForecast025DegreeFastDataset
        if model == "ifs"
        else EcmwfAifsEnsForecastFastDataset
    )
    prefix = FIXTURES.as_uri() + "/"
    dataset = cls(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path / "output"), format=DatasetFormat.ICECHUNK
        ),
    )
    dataset = replace(
        dataset,
        icechunk_virtual_config=IcechunkVirtualConfig(
            containers=(
                icechunk.VirtualChunkContainer(
                    prefix, icechunk.local_filesystem_store(str(FIXTURES))
                ),
            ),
            manifest_split=dataset.icechunk_virtual_config.manifest_split,
        ),
    )
    variables = [v for v in dataset.template_config.data_vars if v.name in VARIABLES]
    get_template = dataset.template_config.get_template

    def template(self: EcmwfFastTemplateConfig, end_time: DatetimeLike) -> xr.DataTree:
        ds = (
            get_template(end_time)
            .sel(
                init_time=slice("2026-11-04", None),
                lead_time=pd.to_timedelta(
                    [0, 1] if model == "ifs" else [0, 6], unit="h"
                ),
                ensemble_member=[0, 1],
            )
            .to_dataset()
        )
        return xr.DataTree(ds[list(VARIABLES)])

    monkeypatch.setattr(
        type(dataset.template_config), "data_vars", property(lambda self: variables)
    )
    monkeypatch.setattr(type(dataset.template_config), "get_template", template)
    generate = EcmwfFastRegionJob.generate_source_file_coords

    def fixture_coords(
        self: EcmwfFastRegionJob, processing_region_ds: xr.Dataset, data_var_group: list
    ) -> list[EcmwfFastSourceFileCoord]:
        return [
            c.model_copy(
                update={
                    "source_params": frozenset({"2t"}),
                    "source_members": (0,) if c.family == "cf" else (1,),
                }
            )
            for c in generate(self, processing_region_ds, data_var_group)
        ]

    monkeypatch.setattr(
        EcmwfFastRegionJob, "generate_source_file_coords", fixture_coords
    )
    monkeypatch.setattr(EcmwfFastRegionJob, "download_concurrency", 2)
    monkeypatch.setattr(
        EcmwfFastSourceFileCoord, "get_url", lambda self: (FIXTURES / self.key).as_uri()
    )
    local = obstore.store.LocalStore(str(FIXTURES))
    monkeypatch.setattr(
        region_job, "s3_store_from_secret", lambda *args, **kwargs: local
    )

    def discover(
        self: EcmwfFastRegionJob, pending: list[EcmwfFastSourceFileCoord]
    ) -> list[tuple[EcmwfFastSourceFileCoord, int]]:
        self._listed_etags = self._listed_etags.copy()
        for coord in pending:
            meta = obstore.head(local, coord.key)
            self._listed_etags[coord.key] = (meta["size"], meta["e_tag"])
        return [(c, (FIXTURES / c.key).stat().st_size) for c in pending]

    monkeypatch.setattr(EcmwfFastRegionJob, "discover_available", discover)
    dataset.backfill_local(append_dim_end=pd.Timestamp("2026-11-04T01:00"))
    repo, _ = dataset.store_factory.icechunk_primary_and_replica_repos()
    initial_snapshot = repo.readonly_session("main").snapshot_id
    before = xr.open_zarr(
        repo.readonly_session(snapshot_id=initial_snapshot).store,
        chunks=None,
        decode_timedelta=True,
        consolidated=False,
    )
    assert before.sizes["init_time"] == 1
    expected = [
        r
        for r in json.loads((FIXTURES / "metadata.json").read_text())
        if r["model"] == model
    ]

    def assert_values(ds: xr.Dataset) -> None:
        for row in expected:
            if pd.Timestamp(row["init_time"]) not in pd.to_datetime(
                ds.init_time.values
            ):
                continue
            cell = ds.sel(
                init_time=pd.Timestamp(row["init_time"]),
                lead_time=pd.Timedelta(hours=row["hours"]),
                ensemble_member=row["members"][0],
                latitude=0,
                longitude=0,
            )
            np.testing.assert_allclose(
                float(cell.temperature_2m.values),
                row["point_lat0_lon0"]["2t"],
                rtol=1e-12,
                atol=1e-12,
            )

    assert_values(before)
    fire = pd.Timestamp("2026-11-04T11:08" if model == "ifs" else "2026-11-04T11:34")
    monkeypatch.setattr(
        pd.Timestamp, "now", classmethod(lambda cls, *args, **kwargs: fire)
    )
    dataset.update("test-fixture-update")
    final = repo.readonly_session("main")
    assert final.snapshot_id != initial_snapshot
    after = xr.open_zarr(
        final.store, chunks=None, decode_timedelta=True, consolidated=False
    )
    assert after.sizes["init_time"] == 2
    assert all(
        manifest.num_chunk_refs <= 4
        for manifest in repo.list_manifest_files(final.snapshot_id)
    )
    assert_values(after)
    for name in VARIABLES:
        assert after[name].encoding["chunks"] == (1, 1, 1, 721, 1440)
    historical = xr.open_zarr(
        repo.readonly_session(snapshot_id=initial_snapshot).store,
        chunks=None,
        decode_timedelta=True,
        consolidated=False,
    )
    assert historical.sizes["init_time"] == 1
    assert_values(historical)
