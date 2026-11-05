import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from threading import Barrier
from unittest.mock import MagicMock

import icechunk
import pandas as pd
import pytest
import xarray as xr
from obstore.exceptions import GenericError, PermissionDeniedError

from reformatters.common import template_utils
from reformatters.common.download import SourceAccessError
from reformatters.common.storage import StoreFactory
from reformatters.ecmwf.aifs_ens.forecast_fast.region_job import (
    EcmwfAifsEnsForecastFastRegionJob,
)
from reformatters.ecmwf.aifs_ens.forecast_fast.template_config import (
    EcmwfAifsEnsForecastFastTemplateConfig,
)
from reformatters.ecmwf.fast import region_job
from reformatters.ecmwf.fast.grib import (
    InterruptedGribStreamError,
    SourceContractError,
)
from reformatters.ecmwf.fast.region_job import (
    EcmwfFastRegionJob,
)
from reformatters.ecmwf.fast.source import EcmwfFastSourceFileCoord
from reformatters.ecmwf.fast.template_config import EcmwfFastTemplateConfig
from reformatters.ecmwf.ifs_ens.forecast_0_25_degree_fast.region_job import (
    EcmwfIfsEnsForecast025DegreeFastRegionJob,
)
from reformatters.ecmwf.ifs_ens.forecast_0_25_degree_fast.template_config import (
    EcmwfIfsEnsForecast025DegreeFastTemplateConfig,
)
from tests.ecmwf.fast.grib_test import PARAMS, chunks, grib1, grib2, source_bytes

LAST_MODIFIED = pd.Timestamp("2026-11-04T00:00Z")


def make_job(
    tmp_path: Path, cls: type[EcmwfFastRegionJob], hour: int = 0
) -> EcmwfFastRegionJob:
    config = EcmwfFastTemplateConfig(model=cls.model)
    ds = xr.Dataset(
        coords={
            "init_time": [pd.Timestamp("2026-11-04") + pd.Timedelta(hours=hour)],
            "lead_time": config.dimension_coordinates()["lead_time"],
            "ensemble_member": [0, 1, 50],
        }
    )
    return cls(
        tmp_store=tmp_path,
        template_ds=xr.DataTree(ds),
        data_vars=config.data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test-fast",
    )


@pytest.mark.parametrize(
    ("cls", "hour", "expected"),
    [
        (EcmwfIfsEnsForecast025DegreeFastRegionJob, 0, 290),
        (EcmwfIfsEnsForecast025DegreeFastRegionJob, 6, 218),
        (EcmwfIfsEnsForecast025DegreeFastRegionJob, 12, 290),
        (EcmwfIfsEnsForecast025DegreeFastRegionJob, 18, 218),
        (EcmwfAifsEnsForecastFastRegionJob, 0, 122),
        (EcmwfAifsEnsForecastFastRegionJob, 6, 122),
        (EcmwfAifsEnsForecastFastRegionJob, 12, 122),
        (EcmwfAifsEnsForecastFastRegionJob, 18, 122),
    ],
)
def test_structural_holes_and_family_boundaries(
    tmp_path: Path, cls: type[EcmwfFastRegionJob], hour: int, expected: int
) -> None:
    job = make_job(tmp_path, cls, hour)
    coords = job.source_file_coords()
    assert len(coords) == expected
    assert len({c.key for c in coords}) == expected
    assert all(len(c.data_vars) == 7 for c in coords)
    assert {c.family for c in coords} == {"cf", "pf"}
    assert {c.out_loc()["ensemble_member"] for c in coords} == {0, 1}
    assert all(c.source_params == PARAMS for c in coords)
    assert all(
        list(c.members) == ([0] if c.family == "cf" else list(range(1, 51)))
        for c in coords
    )


def test_discovery_publishes_out_of_order_partial_families(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(tmp_path, EcmwfIfsEnsForecast025DegreeFastRegionJob)
    coords = job.source_file_coords()
    pf6 = next(
        c for c in coords if c.family == "pf" and c.lead_time == pd.Timedelta("6h")
    )
    cf7 = next(
        c for c in coords if c.family == "cf" and c.lead_time == pd.Timedelta("7h")
    )
    rows = {
        cf7.prefix: [
            (cf7.key, 100, LAST_MODIFIED),
            (cf7.prefix + "unrequested", 50, LAST_MODIFIED),
        ],
        pf6.prefix: [(pf6.key, 200, LAST_MODIFIED)],
    }
    monkeypatch.setattr(region_job, "listing_client", object)
    monkeypatch.setattr(
        region_job, "list_prefix_or_defer", lambda prefix, client: rows[prefix]
    )
    discovered = job.discover_available([pf6, cf7])
    assert [(id(c), size) for c, size in discovered] == [(id(cf7), 100), (id(pf6), 200)]
    rows[pf6.prefix] = []
    assert job.discover_available([pf6, cf7]) == []
    rows[pf6.prefix] = [(pf6.key, 200, LAST_MODIFIED)]
    assert job.discover_available([pf6]) == []
    rows[pf6.prefix] = [(pf6.key, 200, LAST_MODIFIED + pd.Timedelta("1s"))]
    assert job.discover_available([pf6]) == [(pf6, 200)]
    rows[cf7.prefix] = [(cf7.key, 0, LAST_MODIFIED + pd.Timedelta("1s"))]
    assert job.discover_available([cf7]) == []
    rows[cf7.prefix] = [(cf7.key, 100, LAST_MODIFIED + pd.Timedelta("1s"))]
    assert job.discover_available([cf7]) == [(cf7, 100)]


def test_all_nonempty_prefixes_list_concurrently_and_deduplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(tmp_path, EcmwfIfsEnsForecast025DegreeFastRegionJob)
    coords = job.source_file_coords()[:2]
    pending = [
        *coords,
        *(
            c.model_copy(update={"init_time": c.init_time + pd.Timedelta("6h")})
            for c in coords
        ),
    ]
    gate = Barrier(4, timeout=2)
    calls = []
    client = object()
    factory = MagicMock(return_value=client)

    def listed(prefix: str, client: object) -> list[tuple[str, int, datetime]]:
        calls.append(prefix)
        gate.wait()
        coord = next(c for c in pending if c.prefix == prefix)
        return [(coord.key, 100, LAST_MODIFIED), (coord.key, 100, LAST_MODIFIED)]

    monkeypatch.setattr(region_job, "listing_client", factory)
    monkeypatch.setattr(region_job, "list_prefix_or_defer", listed)
    assert len(job.discover_available(pending)) == 4
    assert set(calls) == {c.prefix for c in pending}
    factory.assert_called_once_with()
    calls.clear()
    assert job.discover_available([]) == []
    assert calls == []


def test_stream_attempts_restart_after_interruption_and_never_retry_source_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(tmp_path, EcmwfIfsEnsForecast025DegreeFastRegionJob)
    coord = next(
        c
        for c in job.source_file_coords()
        if c.family == "cf" and c.lead_time == pd.Timedelta("6h")
    )
    data = source_bytes("ifs")

    def response(payload: bytes) -> MagicMock:
        result = MagicMock()
        result.meta = {"size": len(data)}
        result.stream.side_effect = lambda **kwargs: chunks(payload, 31)
        return result

    getter = MagicMock(side_effect=[response(data[:-1]), response(data)])
    monkeypatch.setattr(
        region_job, "s3_store_from_secret", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(region_job.obstore, "get", getter)
    monkeypatch.setattr("reformatters.common.retry.time.sleep", lambda _: None)
    refs = job.file_refs(coord, len(data))
    assert len(refs) == 7
    assert getter.call_count == 2
    assert sum(r.length for r in refs) == len(data)
    getter.reset_mock(side_effect=True)
    getter.side_effect = [response(data[:-1]) for _ in range(2)]
    with pytest.raises(InterruptedGribStreamError):
        job.file_refs(coord, len(data))
    assert getter.call_count == 2
    getter.reset_mock(side_effect=True)
    getter.side_effect = [response(b"bad!" + data[4:])]
    with pytest.raises(SourceContractError, match="magic"):
        job.file_refs(coord, len(data))
    assert getter.call_count == 1


def test_provider_loop_and_default_download_concurrency() -> None:
    assert EcmwfFastRegionJob.download_concurrency == 64
    assert (
        EcmwfIfsEnsForecast025DegreeFastRegionJob.process_virtual_refs
        is EcmwfFastRegionJob.process_virtual_refs
    )
    assert (
        EcmwfAifsEnsForecastFastRegionJob.process_virtual_refs
        is EcmwfFastRegionJob.process_virtual_refs
    )
    assert "process_virtual_refs" in EcmwfFastRegionJob.__dict__


@pytest.fixture
def source_getter(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    getter = MagicMock()
    monkeypatch.setattr(
        region_job, "s3_store_from_secret", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(region_job.obstore, "get", getter)
    monkeypatch.setattr("reformatters.common.retry.time.sleep", lambda _: None)
    return getter


def source_response(payload: bytes) -> MagicMock:
    result = MagicMock()
    result.meta = {"size": len(payload)}
    result.stream.side_effect = lambda **kwargs: chunks(payload, 31)
    return result


@pytest.mark.parametrize(
    "message", ["503 Service Unavailable", "broken pipe", "Unsupported response format"]
)
@pytest.mark.parametrize("during_stream", [False, True])
def test_generic_errors_restart_the_whole_file(
    tmp_path: Path, source_getter: MagicMock, message: str, during_stream: bool
) -> None:
    job = make_job(tmp_path, EcmwfIfsEnsForecast025DegreeFastRegionJob)
    coord = next(
        c
        for c in job.source_file_coords()
        if c.family == "cf" and c.lead_time == pd.Timedelta("6h")
    )
    data = source_bytes("ifs")
    failure = GenericError(message)

    def interrupted_stream(**kwargs: object) -> Iterator[bytes]:
        yield data[:20]
        raise failure

    broken = source_response(data)
    broken.stream.side_effect = interrupted_stream
    source_getter.side_effect = [
        broken if during_stream else failure,
        source_response(data),
    ]
    assert len(job.file_refs(coord, len(data))) == 7
    assert source_getter.call_count == 2


@pytest.mark.parametrize("during_stream", [False, True])
def test_generic_errors_exhaust_exactly_two_attempts(
    tmp_path: Path, source_getter: MagicMock, during_stream: bool
) -> None:
    job = make_job(tmp_path, EcmwfIfsEnsForecast025DegreeFastRegionJob)
    coord = job.source_file_coords()[0]
    error = GenericError("Unsupported response format")

    def broken_stream(**kwargs: object) -> Iterator[bytes]:
        yield b"GRIB"
        raise error

    response = source_response(b"GRIB")
    response.stream.side_effect = broken_stream
    source_getter.side_effect = [response, response] if during_stream else error
    with pytest.raises(GenericError) as caught:
        job.file_refs(coord, 4)
    assert caught.value is error
    assert source_getter.call_count == 2


@pytest.mark.parametrize("failure", ["auth", "programming"])
@pytest.mark.parametrize("during_stream", [False, True])
def test_common_loop_propagates_auth_and_programming_failures_without_retry(
    tmp_path: Path, source_getter: MagicMock, failure: str, during_stream: bool
) -> None:
    job = make_job(tmp_path, EcmwfIfsEnsForecast025DegreeFastRegionJob)
    coord = job.source_file_coords()[0]
    error = (
        PermissionDeniedError("denied")
        if failure == "auth"
        else AssertionError("programming invariant")
    )

    def broken_stream(**kwargs: object) -> Iterator[bytes]:
        yield b"GRIB"
        raise error

    response = source_response(b"GRIB")
    response.stream.side_effect = broken_stream
    source_getter.side_effect = [response if during_stream else error]
    with pytest.raises(SourceAccessError if failure == "auth" else AssertionError):
        job._file_refs_or_skip(coord, 4)
    assert source_getter.call_count == 1


@pytest.mark.parametrize("damage", ["size", "magic", "date", "structure", "duplicate"])
def test_source_contract_failures_are_logged_and_skipped_without_retry(
    tmp_path: Path,
    source_getter: MagicMock,
    caplog: pytest.LogCaptureFixture,
    damage: str,
) -> None:
    job = make_job(tmp_path, EcmwfIfsEnsForecast025DegreeFastRegionJob)
    coord = next(
        c
        for c in job.source_file_coords()
        if c.family == "cf" and c.lead_time == pd.Timedelta("6h")
    )
    data = bytearray(source_bytes("ifs"))
    if damage == "magic":
        data[:4] = b"bad!"
    elif damage == "date":
        data[8 + 13] = 13
    elif damage == "structure":
        data[8 + 7] = 0
    elif damage == "duplicate":
        data.extend(grib1("2t", 0))
    source_getter.return_value = source_response(bytes(data))
    assert job._file_refs_or_skip(coord, len(data) + (damage == "size")) == []
    assert source_getter.call_count == 1
    assert any(
        record.exc_info and isinstance(record.exc_info[1], SourceContractError)
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "cls",
    [EcmwfIfsEnsForecast025DegreeFastRegionJob, EcmwfAifsEnsForecastFastRegionJob],
)
@pytest.mark.parametrize("partial_param", ["2t", "tp"])
def test_common_loop_partial_file_requires_temperature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_getter: MagicMock,
    cls: type[EcmwfFastRegionJob],
    partial_param: str,
) -> None:
    config = (
        EcmwfIfsEnsForecast025DegreeFastTemplateConfig()
        if cls.model == "ifs"
        else EcmwfAifsEnsForecastFastTemplateConfig()
    )
    template = config.get_template(pd.Timestamp("2026-11-04T01:00")).sel(
        init_time=[pd.Timestamp("2026-11-04")],
        lead_time=pd.to_timedelta([6, 12], unit="h"),
        ensemble_member=[0],
    )
    job = cls(
        tmp_store=tmp_path,
        template_ds=template,
        data_vars=config.data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test-skip-bad-file",
    )
    bad, healthy = job.source_file_coords()
    build = grib1 if cls.model == "ifs" else grib2
    files = {
        bad.key: build(partial_param, 0),
        healthy.key: source_bytes(cls.model, hours=12),
    }
    source_getter.side_effect = lambda store, key: source_response(files[key])
    monkeypatch.setattr(
        cls,
        "discover_available",
        lambda self, pending: [(c, len(files[c.key])) for c in pending],
    )
    monkeypatch.setattr(
        EcmwfFastSourceFileCoord, "get_url", lambda self: (tmp_path / self.key).as_uri()
    )
    repo_config = icechunk.RepositoryConfig.default()
    repo_config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            tmp_path.as_uri() + "/", icechunk.local_filesystem_store(str(tmp_path))
        )
    )
    repo = icechunk.Repository.create(icechunk.in_memory_storage(), config=repo_config)
    template_utils.write_metadata(
        template, repo.writable_session("main").store, mode="w-", consolidated=False
    )
    before = repo.readonly_session("main").snapshot_id
    job.process_virtual(repo, [], "main", [bad, healthy])
    final = repo.readonly_session("main")
    assert final.snapshot_id != before
    assert job.filter_already_present([bad, healthy], final.store) == (
        [] if partial_param == "2t" else [bad]
    )
    assert source_getter.call_count == 2
    refs = list(final.all_virtual_chunk_locations())
    assert refs.count(healthy.get_url()) == len(config.data_vars)
    assert refs.count(bad.get_url()) == (1 if partial_param == "2t" else 0)


@pytest.fixture(
    params=[
        EcmwfIfsEnsForecast025DegreeFastRegionJob,
        EcmwfAifsEnsForecastFastRegionJob,
    ]
)
def polling_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> tuple[EcmwfFastRegionJob, icechunk.Repository]:
    cls = request.param
    config = (
        EcmwfIfsEnsForecast025DegreeFastTemplateConfig()
        if cls.model == "ifs"
        else EcmwfAifsEnsForecastFastTemplateConfig()
    )
    template = config.get_template(pd.Timestamp("2026-11-04T01:00")).sel(
        init_time=[pd.Timestamp("2026-11-04")],
        lead_time=pd.to_timedelta([6, 12], unit="h"),
        ensemble_member=[0],
    )
    job = cls(
        tmp_store=tmp_path,
        template_ds=template,
        data_vars=config.data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test-poll-retry",
        processing_mode="update",
    )
    monkeypatch.setattr(
        EcmwfFastSourceFileCoord, "get_url", lambda self: (tmp_path / self.key).as_uri()
    )
    monkeypatch.setattr(region_job, "listing_client", object)
    monkeypatch.setattr(
        "reformatters.common.virtual_region_job.time.sleep", lambda _: None
    )
    repo_config = icechunk.RepositoryConfig.default()
    repo_config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            tmp_path.as_uri() + "/", icechunk.local_filesystem_store(str(tmp_path))
        )
    )
    repo = icechunk.Repository.create(icechunk.in_memory_storage(), config=repo_config)
    template_utils.write_metadata(
        template, repo.writable_session("main").store, mode="w-", consolidated=False
    )
    return job, repo


def test_changed_last_modified_retries_same_size_file_during_update(
    polling_job: tuple[EcmwfFastRegionJob, icechunk.Repository],
    monkeypatch: pytest.MonkeyPatch,
    source_getter: MagicMock,
) -> None:
    job, repo = polling_job
    bad, healthy = job.source_file_coords()
    repaired = source_bytes(job.model)
    files = {
        bad.key: b"bad!" + repaired[4:],
        healthy.key: source_bytes(job.model, hours=12),
    }
    source_getter.side_effect = lambda store, key: source_response(files[key])
    ticks = 0

    def listed(prefix: str, client: object) -> list[tuple[str, int, datetime]]:
        nonlocal ticks
        ticks += 1
        if ticks == 2:
            assert job.filter_already_present(
                [bad, healthy], repo.readonly_session("main").store
            ) == [bad]
            assert source_getter.call_count == 2
        if ticks == 3:
            assert source_getter.call_count == 2
            assert len(files[bad.key]) == len(repaired)
            files[bad.key] = repaired
        return [
            (
                coord.key,
                len(files[coord.key]),
                LAST_MODIFIED + pd.Timedelta(seconds=int(ticks >= 3 and coord is bad)),
            )
            for coord in (bad, healthy)
        ]

    monkeypatch.setattr(region_job, "list_prefix_or_defer", listed)
    assert job.process_virtual(repo, [], "main", [bad, healthy]) is True
    assert ticks == 3
    assert [call.args[1] for call in source_getter.call_args_list].count(bad.key) == 2
    assert [call.args[1] for call in source_getter.call_args_list].count(
        healthy.key
    ) == 1
    final = repo.readonly_session("main")
    assert job.filter_already_present([bad, healthy], final.store) == []
    assert len(list(final.all_virtual_chunk_locations())) == 2 * len(job.data_vars)


@pytest.mark.parametrize("failure", ["contract", "transient"])
def test_unchanged_failures_wait_for_deadline_and_fresh_instance_retries(
    polling_job: tuple[EcmwfFastRegionJob, icechunk.Repository],
    monkeypatch: pytest.MonkeyPatch,
    source_getter: MagicMock,
    failure: str,
) -> None:
    job, repo = polling_job
    coord = job.source_file_coords()[0]
    now = LAST_MODIFIED.tz_localize(None)
    deadline = now + pd.Timedelta("3s")
    job = job.model_copy(update={"poll_deadline": deadline})
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls, *a, **kw: now))
    ticks = 0

    def listed(prefix: str, client: object) -> list[tuple[str, int, datetime]]:
        nonlocal ticks, now
        ticks += 1
        now += pd.Timedelta("1s")
        return [(coord.key, 100, LAST_MODIFIED)]

    monkeypatch.setattr(region_job, "list_prefix_or_defer", listed)
    if failure == "contract":
        source_getter.return_value = source_response(b"bad!")
    else:
        source_getter.side_effect = GenericError("503 Service Unavailable")
    before = repo.readonly_session("main").snapshot_id
    assert job.process_virtual(repo, [], "main", [coord]) is False
    assert ticks == 3
    expected_attempts = 1 if failure == "contract" else 2
    assert source_getter.call_count == expected_attempts
    assert repo.readonly_session("main").snapshot_id == before

    fresh = type(job).model_validate(job.model_dump())
    assert fresh.discover_available([coord]) == [(coord, 100)]
    assert fresh._file_refs_or_skip(coord, 100) == []
    assert source_getter.call_count == 2 * expected_attempts


def test_worker_driver_version_state_does_not_leak_to_independent_runs(
    polling_job: tuple[EcmwfFastRegionJob, icechunk.Repository],
    monkeypatch: pytest.MonkeyPatch,
    source_getter: MagicMock,
) -> None:
    job, repo = polling_job
    job = job.model_copy(update={"processing_mode": "backfill"})
    coords = list(job.source_file_coords())
    monkeypatch.setattr(
        region_job,
        "list_prefix_or_defer",
        lambda prefix, client: [(coord.key, 100, LAST_MODIFIED) for coord in coords],
    )
    source_getter.return_value = source_response(b"bad!")
    factory = MagicMock(spec=StoreFactory)
    factory.icechunk_primary_and_replica_repos.return_value = (repo, [])

    for _ in range(2):
        assert (
            type(job).process_virtual_worker_jobs(
                [job], factory, "main", overwrite_chunks=False
            )
            is False
        )
    assert source_getter.call_count == 2 * len(coords)
    assert job.discover_available(coords) == [(coord, 100) for coord in coords]


@pytest.mark.parametrize(
    "cls",
    [EcmwfIfsEnsForecast025DegreeFastRegionJob, EcmwfAifsEnsForecastFastRegionJob],
)
def test_real_fixture_metadata_discrepancies_alert_without_changing_refs(
    tmp_path: Path,
    source_getter: MagicMock,
    caplog: pytest.LogCaptureFixture,
    cls: type[EcmwfFastRegionJob],
) -> None:
    config = (
        EcmwfIfsEnsForecast025DegreeFastTemplateConfig()
        if cls.model == "ifs"
        else EcmwfAifsEnsForecastFastTemplateConfig()
    )
    job = make_job(tmp_path, cls).model_copy(
        update={
            "template_ds": config.get_template(pd.Timestamp("2026-11-04T01:00")).sel(
                init_time=[pd.Timestamp("2026-11-04")]
            )
        }
    )
    coord = next(
        c
        for c in job.source_file_coords()
        if c.family == "cf" and c.lead_time == pd.Timedelta(0)
    ).model_copy(update={"source_params": frozenset({"2t"})})
    original = (Path(__file__).parent / "fixtures" / coord.key).read_bytes()
    source_getter.return_value = source_response(original)
    baseline = job._file_refs_or_skip(coord, len(original))
    assert len(baseline) == 1
    assert not [r for r in caplog.records if r.levelname == "ERROR"]
    changed = bytearray(original)
    if cls.model == "ifs":
        changed[8 + 4] = 99
    else:
        offset = 16
        while offset < len(changed) - 4:
            size = int.from_bytes(changed[offset : offset + 4], "big")
            number = changed[offset + 4]
            if number == 1:
                changed[offset + 5 : offset + 7] = (99).to_bytes(2, "big")
            elif number == 3:
                changed[offset + 14] = 5
            elif number == 4:
                changed[offset + 36] = 101
            offset += size
    caplog.clear()
    source_getter.return_value = source_response(bytes(changed))
    assert job._file_refs_or_skip(coord, len(changed)) == baseline
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1
    assert coord.key in errors[0]
    assert "centre=99 (expected 98)" in errors[0]
    if cls.model == "aifs":
        assert "earth shape=5 (expected 6)" in errors[0]
        assert "ensemble size=101 (expected 51)" in errors[0]
    caplog.clear()
    changed[-4:] = b"bad!"
    source_getter.return_value = source_response(bytes(changed))
    assert job._file_refs_or_skip(coord, len(changed)) == []
    assert "Source discrepancies" not in caplog.text


@pytest.mark.parametrize("last_modified", [None, LAST_MODIFIED])
def test_stream_request_event_preserves_optional_last_modified(
    tmp_path: Path,
    source_getter: MagicMock,
    caplog: pytest.LogCaptureFixture,
    last_modified: datetime | None,
) -> None:
    job = make_job(tmp_path, EcmwfIfsEnsForecast025DegreeFastRegionJob).model_copy(
        update={"processing_mode": "update"}
    )
    coord = next(
        c
        for c in job.source_file_coords()
        if c.family == "cf" and c.lead_time == pd.Timedelta("6h")
    )
    if last_modified is not None:
        job._last_modified[coord.key] = last_modified
    data = source_bytes("ifs")
    source_getter.return_value = source_response(data)
    with caplog.at_level("INFO"):
        assert job.file_refs(coord, len(data))
    events = [
        json.loads(r.getMessage().split("Virtual pipeline ", 1)[1])
        for r in caplog.records
        if '"event": "request"' in r.getMessage()
    ]
    assert len(events) == 1
    assert events[0]["file"] == coord.get_url()
    assert events[0]["key"] == coord.key
    assert events[0]["bytes"] == len(data)
    assert events[0]["last_modified"] == (
        None if last_modified is None else last_modified.isoformat()
    )
    assert datetime.fromisoformat(events[0]["requested_at"]).tzinfo is not None
