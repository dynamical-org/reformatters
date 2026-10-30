from pathlib import Path
from unittest.mock import Mock

import httpx
import icechunk
import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.codecs import BytesCodec, ZstdCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import MemoryStore

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import template_utils, validation
from reformatters.common.staging import rename_cronjob_for_staging
from reformatters.google.weathernext3.forecast_virtual.dynamical_dataset import (
    GoogleWeathernext3ForecastVirtualDataset,
)
from reformatters.google.weathernext3.forecast_virtual.region_job import (
    GoogleWeathernext3ForecastVirtualRegionJob,
    GoogleWeathernext3ForecastVirtualSourceFileCoord,
)
from reformatters.google.weathernext3.forecast_virtual.template_config import STATISTICS
from reformatters.google.weathernext_terms import (
    COPYRIGHT,
    LICENSE,
    REAL_TIME_ATTRIBUTION,
)
from reformatters.google.weathernext_virtual import listing
from reformatters.google.weathernext_virtual.listing import NativeObjectMetadata

DATASETS = [
    ds
    for ds in DYNAMICAL_DATASETS
    if isinstance(ds, GoogleWeathernext3ForecastVirtualDataset)
]


def job_for(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
) -> GoogleWeathernext3ForecastVirtualRegionJob:
    config = dataset.template_config
    return dataset.region_job_class(
        tmp_store=Path("unused"),
        template_ds=config.get_template(pd.Timestamp("2026-01-02")),
        data_vars=config.data_vars[:1],
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
    )


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
def test_terms_licence_and_attribution(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
) -> None:
    attributes = dataset.template_config.dataset_attributes
    assert attributes.license == LICENSE
    assert attributes.attribution == (
        "Google WeatherNext 3 forecast data processed by dynamical.org "
        "from Google's WeatherNext 3 experimental data. "
        f"{COPYRIGHT}. {REAL_TIME_ATTRIBUTION}"
    )
    assert all(
        "earth-engine" not in text and "Earth Engine" not in text
        for text in (attributes.attribution, attributes.license, attributes.description)
    )


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
def test_template_and_operations(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
) -> None:
    assert dataset.region_job_class.operational_update_window == pd.Timedelta("24h")
    config = dataset.template_config
    template = config.get_template(
        config.append_dim_start + config.append_dim_frequency
    ).to_dataset()
    assert tuple(template.temperature_2m.dims) == (
        "init_time",
        "statistic",
        "lead_time",
        "y",
        "x",
    )
    assert tuple(template.statistic.values) == STATISTICS
    assert template.statistic.encoding["chunks"] == (6,)
    assert template.sizes["lead_time"] == config.horizon_hours
    assert template.lead_time.values[0] == np.timedelta64(1, "h")
    assert template.expected_forecast_length.values[0] == np.timedelta64(
        config.horizon_hours, "h"
    )
    assert len(config.data_vars) == (19 if config.grid_degrees == 0.1 else 2)
    for var in config.data_vars:
        assert var.encoding.chunks == (
            1,
            1,
            1,
            template.sizes["y"],
            template.sizes["x"],
        )
        assert var.encoding.compressors == [
            ZstdCodec(level=0, checksum=False).to_dict()
        ]
        assert var.internal_attrs.keep_mantissa_bits == "no-rounding"
    (update,) = dataset.operational_kubernetes_resources("test")
    assert update.suspend
    assert (
        dataset._validation_monitor_name()
        == update.name.removesuffix("-update") + "-validate"
    )
    assert (
        len(rename_cronjob_for_staging(update, dataset.dataset_id, "0.1.0").name) <= 52
    )
    decode = dataset.validators()[-1]
    assert isinstance(decode, validation.CheckVirtualDecodeHealth)
    assert decode.sampled_levels == len(STATISTICS)
    assert decode.max_workers == 2


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
def test_source_mapping_and_horizons(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
) -> None:
    job = job_for(dataset)
    coords = job.source_file_coords()
    assert [coord.lead_time for coord in coords[:2]] == [
        pd.Timedelta("1h"),
        pd.Timedelta("2h"),
    ]
    coord = coords[1]
    assert coord.lead_index == 1
    assert coord.chunk_location(job.data_vars[0], "p90").endswith("_p90/c/1/0/0")
    template = job.template_ds.to_dataset()
    for hour, horizon in [(0, 360), (1, 48), (6, 360), (23, 48)]:
        leads = job._available_lead_times(
            pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour), template
        )
        assert len(leads) == min(horizon, dataset.template_config.horizon_hours)


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
@pytest.mark.parametrize("hour", [0, 1])
def test_listing_admission_and_restart(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
    hour: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = job_for(dataset)
    init = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour)
    # A synoptic init on the hourly product still uses the source's 360-hour horizon.
    coord = GoogleWeathernext3ForecastVirtualSourceFileCoord(
        init_time=init, lead_time=pd.Timedelta("1h"), data_vars=job.data_vars
    )
    (query,) = job._listing_queries(coord)
    assert query.delimiter is None
    assert query.match_glob is None
    locations = [chunk.location for chunk in job._source_chunks(coord)]
    missing = True
    calls = []

    def get(url: str, *, params: dict[str, str]) -> httpx.Response:
        calls.append(params)
        assert url == listing.OBJECTS_LOCATION
        assert "delimiter" not in params
        objects = locations[:-1] if missing else locations
        items = [
            {
                "name": loc.removeprefix(listing.PROXY_LOCATION_PREFIX),
                "size": "123",
                "md5Hash": "AAAAAAAAAAAAAAAAAAAAAA==",
            }
            for loc in objects
        ]
        items += [
            {"name": query.prefix + "mean/c/0/", "size": "0"},
            {
                "name": query.prefix + "mean/zarr.json",
                "size": "20",
                "md5Hash": "AAAAAAAAAAAAAAAAAAAAAA==",
            },
        ]
        return httpx.Response(
            200, json={"items": items}, request=httpx.Request("GET", url)
        )

    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.get.side_effect = get
    monkeypatch.setattr(listing.httpx, "Client", lambda **kwargs: client)
    assert job.discover_available([coord]) == []
    missing = False
    assert job.discover_available([coord]) == [(coord, 0)]
    assert len(coord.chunk_metadata) == 6
    if hour == 0 or dataset.template_config.horizon_hours == 48:
        refs = job.file_refs(coord, 0)
        assert len(refs) == 6
        assert all(
            ref.etag_checksum == '"00000000000000000000000000000000"' for ref in refs
        )
        repo = icechunk.Repository.create(
            icechunk.in_memory_storage(),
            config=icechunk.RepositoryConfig(
                virtual_chunk_containers={
                    "wn": listing.weathernext_virtual_chunk_containers()[0]
                }
            ),
        )
        session = repo.writable_session("main")
        template_utils.write_metadata(
            job.template_ds,
            session.store,
            "w-",
            consolidated=False,
            skip_icechunk_commit=True,
        )
        for index, ref in enumerate(refs):
            init_index = (
                job.template_ds.to_dataset().get_index("init_time").get_loc(init)
            )
            session.store.set_virtual_ref(
                f"{ref.data_var.path}/c/{init_index}/{index}/0/0/0",
                ref.location,
                offset=0,
                length=ref.length,
            )
        session.commit("six statistics")
        assert (
            job.filter_already_present(
                [coord.model_copy(update={"chunk_metadata": {}})],
                repo.readonly_session("main").store,
            )
            == []
        )
    calls.clear()
    second = coord.model_copy(update={"lead_time": pd.Timedelta("2h")})
    job.discover_available([coord, second])
    assert len(calls) == 1
    assert calls[0] == {"prefix": query.prefix, "maxResults": "1000"}


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
@pytest.mark.parametrize(
    ("fire", "newest_hourly", "newest_synoptic"),
    [
        ("2026-02-01T00:10", "2026-01-31T18:00", "2026-01-31T18:00"),
        ("2026-02-01T05:10", "2026-01-31T23:00", "2026-01-31T18:00"),
        ("2026-02-01T06:10", "2026-02-01T00:00", "2026-02-01T00:00"),
        ("2026-02-04T19:10", "2026-02-04T13:00", "2026-02-04T12:00"),
    ],
)
def test_update_window(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
    fire: str,
    newest_hourly: str,
    newest_synoptic: str,
) -> None:
    config = dataset.template_config
    jobs, template = dataset.region_job_class.operational_update_jobs(
        MemoryStore(),
        Path("unused"),
        config.get_template,
        "init_time",
        config.data_vars[:1],
        "test",
        pd.Timestamp(fire),
    )
    (job,) = jobs
    assert isinstance(job, GoogleWeathernext3ForecastVirtualRegionJob)
    newest = pd.Timestamp(
        newest_hourly
        if config.append_dim_frequency == pd.Timedelta("1h")
        else newest_synoptic
    )
    inits = template.to_dataset().get_index("init_time")
    assert inits[-1] == newest
    window = inits[job.region]
    assert window[0] == newest + config.append_dim_frequency - (
        job.operational_update_window
    )
    leads = pd.timedelta_range("1h", periods=config.horizon_hours, freq="1h")
    coords = job.source_file_coords()
    assert {(coord.init_time, coord.lead_time) for coord in coords} == {
        (init, lead) for init in window for lead in leads
    }
    assert all(query.match_glob is None for query in job._listing_queries(coords[-1]))


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
def test_backfill_includes_every_lead(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
) -> None:
    config = dataset.template_config
    jobs = dataset.region_job_class.get_jobs(
        tmp_store=Path("unused"),
        template_ds=config.get_template(pd.Timestamp("2026-01-02")),
        append_dim="init_time",
        all_data_vars=config.data_vars,
        reformat_job_name="backfill",
        filter_variable_names=[config.data_vars[0].name],
    )
    assert jobs
    assert {
        coord.lead_time for job in jobs for coord in job.source_file_coords()
    } == set(pd.timedelta_range("1h", periods=config.horizon_hours, freq="1h"))


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
def test_unrestricted_update_rejects_backlog_before_discovery(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = dataset.template_config
    split = dataset.region_job_class.manifest_init_split
    template = config.get_template(
        config.append_dim_start + (split + 1) * config.append_dim_frequency
    )
    job = dataset.region_job_class(
        tmp_store=Path("unused"),
        template_ds=template,
        data_vars=config.data_vars,
        append_dim="init_time",
        region=slice(0, split + 1),
        reformat_job_name="test",
        processing_mode="update",
        poll_deadline=pd.Timestamp("2000-01-01"),
    )
    coords = job.source_file_coords()
    bound = split * config.horizon_hours * len(config.data_vars) * len(STATISTICS)
    assert len(coords) * len(STATISTICS) > bound
    discovery = Mock(side_effect=AssertionError("discovery must not run"))
    monkeypatch.setattr(type(job), "discover_available", discovery)
    with pytest.raises(AssertionError, match=rf"exceeding bound {bound}"):
        list(job.process_virtual_refs(coords))
    discovery.assert_not_called()

    steady = [coord for coord in coords if coord.init_time == coords[-1].init_time]
    monkeypatch.setattr(type(job), "discover_available", Mock(return_value=[]))
    assert list(job.process_virtual_refs(steady)) == []


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
def test_ref_agreement_rejects_mismapping(
    dataset: GoogleWeathernext3ForecastVirtualDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = job_for(dataset)
    coord = job.source_file_coords()[0]
    chunks = job._source_chunks(coord)
    coord.chunk_metadata.update(
        {chunk.location: NativeObjectMetadata(100, '"etag"') for chunk in chunks}
    )
    bad = chunks[0]._replace(
        out_loc={**chunks[0].out_loc, "lead_time": pd.Timedelta("2h")}
    )
    monkeypatch.setattr(
        type(job), "_source_chunks", lambda self, coord: [bad, *chunks[1:]]
    )
    with pytest.raises(AssertionError):
        job.file_refs(coord, 0)


def test_source_zstd_bytes_decode_with_extra_singleton_dims() -> None:
    source = zarr.create_array(
        store=MemoryStore(),
        shape=(1, 3, 4),
        chunks=(1, 3, 4),
        dtype="float32",
        serializer=BytesCodec(endian="little"),
        compressors=[ZstdCodec(level=0, checksum=False)],
    )
    values = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    values[0, 1, 1] = np.nan
    source[:] = values
    raw = sync(source.store.get("c/0/0/0", prototype=default_buffer_prototype()))
    assert raw is not None
    target = zarr.create_array(
        store=MemoryStore(),
        shape=(1, 1, 1, 3, 4),
        chunks=(1, 1, 1, 3, 4),
        dtype="float32",
        serializer=BytesCodec(endian="little"),
        compressors=[ZstdCodec(level=0, checksum=False)],
    )
    sync(target.store.set("c/0/0/0/0/0", raw))
    np.testing.assert_array_equal(
        np.asarray(target[:]).reshape(values.shape), source[:]
    )


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
def test_template_regenerates_identically(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = dataset.template_config
    original = config.template_path()
    expected = {
        p.relative_to(original): p.read_bytes()
        for p in original.rglob("*")
        if p.is_file()
    }
    destination = tmp_path / "latest.zarr"
    monkeypatch.setattr(type(config), "template_path", lambda self: destination)
    config.update_template()
    actual = {
        p.relative_to(destination): p.read_bytes()
        for p in destination.rglob("*")
        if p.is_file()
    }
    assert actual == expected


def test_validators_decode_every_statistic_on_small_store() -> None:
    job = job_for(DATASETS[0])
    template = job.template_ds.isel(
        init_time=slice(0, 1), lead_time=slice(0, 1), y=slice(0, 2), x=slice(0, 2)
    )
    name = job.data_vars[0].path
    for var in job.data_vars:
        template[var.path].encoding["chunks"] = (1, 1, 1, 2, 2)
    job = job.model_copy(update={"template_ds": template})
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    session = repo.writable_session("main")
    template_utils.write_metadata(
        template, session.store, "w-", consolidated=False, skip_icechunk_commit=True
    )
    array = zarr.open_group(session.store, mode="r+")[name]
    assert isinstance(array, zarr.Array)
    array[:] = np.ones((1, 6, 1, 2, 2), dtype=np.float32)
    session.commit("all statistics")
    checks = [
        validation.CheckVirtualManifestCompleteness(),
        validation.CheckVirtualDecodeHealth(
            sampled_levels=len(STATISTICS), max_workers=2
        ),
    ]
    validation.validate_dataset(
        checks,
        store=repo.readonly_session("main").store,
        append_dim="init_time",
        dataset_id="test",
        region_job=job,
    )
    session = repo.writable_session("main")
    sync(
        session.store.set(
            f"{name}/c/0/1/0/0/0",
            default_buffer_prototype().buffer.from_bytes(b"corrupt p10"),
        )
    )
    session.commit("corrupt p10")
    with pytest.raises(validation.OperationalValidationError):
        validation.validate_dataset(
            checks,
            store=repo.readonly_session("main").store,
            append_dim="init_time",
            dataset_id="test",
            region_job=job,
        )


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda ds: ds.dataset_id)
def test_unit_conversions_decode_source_values(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
) -> None:
    for var in dataset.template_config.data_vars:
        source = zarr.create_array(
            store=MemoryStore(),
            shape=(1, 1, 2),
            chunks=(1, 1, 2),
            dtype="float32",
            serializer=BytesCodec(endian="little"),
            compressors=[ZstdCodec(level=0, checksum=False)],
        )
        raw = np.array([300, np.nan], dtype=np.float32).reshape(1, 1, 2)
        source[:] = raw
        payload = sync(
            source.store.get("c/0/0/0", prototype=default_buffer_prototype())
        )
        assert payload is not None
        assert var.encoding.serializer is not None
        target = zarr.create_array(
            store=MemoryStore(),
            shape=(1, 1, 1, 1, 2),
            chunks=(1, 1, 1, 1, 2),
            dtype="float32",
            serializer=var.encoding.serializer,
            compressors=var.encoding.compressors,
            filters=var.encoding.filters,
        )
        sync(target.store.set("c/0/0/0/0/0", payload))
        expected = raw
        if "temperature" in var.name:
            expected = raw - 273.15
        elif var.name.startswith("precipitation"):
            expected = raw * 1000 / 3600
        elif "radiation" in var.name:
            expected = raw / 3600
        elif "cloud" in var.name:
            expected = raw * 100
        np.testing.assert_allclose(
            np.asarray(target[:]).reshape(raw.shape),
            expected,
            rtol=1e-6,
            equal_nan=True,
        )
