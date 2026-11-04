from collections.abc import Iterator
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
from reformatters.common.virtual_region_job import VirtualRegionJob
from reformatters.ecmwf.aifs_ens.forecast_fast.region_job import (
    EcmwfAifsEnsForecastFastRegionJob,
)
from reformatters.ecmwf.aifs_ens.forecast_fast.template_config import (
    EcmwfAifsEnsForecastFastTemplateConfig,
)
from reformatters.ecmwf.fast import region_job
from reformatters.ecmwf.fast.grib import (
    PARAMS,
    InterruptedGribStreamError,
    SourceContractError,
)
from reformatters.ecmwf.fast.region_job import (
    EcmwfFastRegionJob,
    TransientSourceReadError,
)
from reformatters.ecmwf.fast.source import EcmwfFastSourceFileCoord
from reformatters.ecmwf.fast.template_config import EcmwfFastTemplateConfig
from reformatters.ecmwf.ifs_ens.forecast_0_25_degree_fast.region_job import (
    EcmwfIfsEnsForecast025DegreeFastRegionJob,
)
from reformatters.ecmwf.ifs_ens.forecast_0_25_degree_fast.template_config import (
    EcmwfIfsEnsForecast025DegreeFastTemplateConfig,
)
from tests.ecmwf.fast.grib_test import chunks, grib1, grib2, source_bytes


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
        cf7.prefix: [(cf7.key, 100), (cf7.prefix + "unrequested", 50)],
        pf6.prefix: [(pf6.key, 200)],
    }
    monkeypatch.setattr(region_job, "listing_client", lambda _: object())
    monkeypatch.setattr(
        region_job, "list_prefix_or_defer", lambda prefix, client: rows[prefix]
    )
    discovered = job.discover_available([pf6, cf7])
    assert [(id(c), size) for c, size in discovered] == [(id(cf7), 100), (id(pf6), 200)]
    rows[pf6.prefix] = []
    assert job.discover_available([pf6, cf7]) == [(cf7, 100)]
    rows[pf6.prefix] = [(pf6.key, 200)]
    assert job.discover_available([pf6]) == [(pf6, 200)]
    rows[cf7.prefix] = [(cf7.key, 0)]
    assert job.discover_available([cf7]) == []


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

    def listed(prefix: str, client: object) -> list[tuple[str, int]]:
        calls.append(prefix)
        gate.wait()
        coord = next(c for c in pending if c.prefix == prefix)
        return [(coord.key, 100), (coord.key, 100)]

    monkeypatch.setattr(region_job, "listing_client", factory)
    monkeypatch.setattr(region_job, "list_prefix_or_defer", listed)
    assert len(job.discover_available(pending)) == 4
    assert set(calls) == {c.prefix for c in pending}
    factory.assert_called_once_with(4)
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
    getter.side_effect = [response(data[:-1]) for _ in range(3)]
    with pytest.raises(InterruptedGribStreamError):
        job.file_refs(coord, len(data))
    assert getter.call_count == 3
    getter.reset_mock(side_effect=True)
    getter.side_effect = [response(b"bad!" + data[4:])]
    with pytest.raises(SourceContractError, match="magic"):
        job.file_refs(coord, len(data))
    assert getter.call_count == 1


def test_common_loop_and_default_download_concurrency_are_inherited() -> None:
    assert EcmwfFastRegionJob.download_concurrency == 64
    assert (
        EcmwfFastRegionJob.process_virtual_refs is VirtualRegionJob.process_virtual_refs
    )


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
    "message",
    [
        "request timed out",
        "request timeout",
        "connection reset by peer",
        "connection closed",
        "connection refused",
        "error sending request",
        "error decoding response body",
        "incomplete message",
        "500 Internal Server Error",
        "502 Bad Gateway",
        "503 Service Unavailable",
        "504 Gateway Timeout",
        "429 Too Many Requests",
    ],
)
@pytest.mark.parametrize("during_stream", [False, True])
def test_generic_transient_errors_restart_the_whole_file(
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


def test_generic_errors_retry_only_known_transients(
    tmp_path: Path, source_getter: MagicMock
) -> None:
    job = make_job(tmp_path, EcmwfIfsEnsForecast025DegreeFastRegionJob)
    coord = job.source_file_coords()[0]
    unexpected = GenericError("Unsupported response format")
    source_getter.side_effect = unexpected
    with pytest.raises(GenericError) as caught:
        job.file_refs(coord, 100)
    assert caught.value is unexpected
    assert source_getter.call_count == 1
    source_getter.reset_mock()
    transient = GenericError("503 Service Unavailable")
    source_getter.side_effect = transient
    with pytest.raises(TransientSourceReadError) as exhausted:
        job.file_refs(coord, 100)
    assert exhausted.value.__cause__ is transient
    assert source_getter.call_count == 3


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


@pytest.mark.parametrize(
    "damage", ["size", "magic", "date", "structure", "duplicate", "missing"]
)
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
    elif damage == "missing":
        data = bytearray(grib1("2t", 0))
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
def test_common_loop_commits_healthy_file_and_leaves_incomplete_file_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_getter: MagicMock,
    cls: type[EcmwfFastRegionJob],
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
    files = {bad.key: build("2t", 0), healthy.key: source_bytes(cls.model, hours=12)}
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
    assert job.filter_already_present([bad, healthy], final.store) == [bad]
    assert source_getter.call_count == 2
    refs = list(final.all_virtual_chunk_locations())
    assert refs == [healthy.get_url()] * len(config.data_vars)
