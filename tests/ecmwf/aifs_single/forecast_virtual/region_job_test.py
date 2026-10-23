import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import icechunk
import obstore.store
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.storage import (
    IcechunkVirtualConfig,
    _repository_config_and_credentials,
    manifest_append_dim_split,
)
from reformatters.ecmwf.aifs_single.forecast_virtual import (
    region_job as region_job_module,
)
from reformatters.ecmwf.aifs_single.forecast_virtual.region_job import (
    S3_FALLBACK_BEFORE,
    EcmwfAifsSingleForecastVirtualRegionJob,
    EcmwfAifsSingleForecastVirtualSourceFileCoord,
    aifs_single_virtual_chunk_containers,
)
from reformatters.ecmwf.aifs_single.forecast_virtual.template_config import (
    AIFS_2026_UPGRADE_DATE,
    PRESSURE_LEVELS,
    EcmwfAifsSingleForecastVirtualTemplateConfig,
    EcmwfAifsSingleVirtualDataVar,
)

FIXTURES_DIR = Path(__file__).parents[2] / "fixtures"

TEMPLATE_CONFIG = EcmwfAifsSingleForecastVirtualTemplateConfig()
_ERA1_INIT = pd.Timestamp("2024-06-01T00:00")
_ERA2_INIT = pd.Timestamp("2025-03-01T00:00")
_LEAD_6H = pd.Timedelta("6h")


def get_var(path: str) -> EcmwfAifsSingleVirtualDataVar:
    return next(v for v in TEMPLATE_CONFIG.data_vars if v.path == path)


@pytest.fixture(scope="module")
def template_ds() -> xr.DataTree:
    return TEMPLATE_CONFIG.get_template(pd.Timestamp("2024-04-02T00:00"))


def make_job(
    template_ds: xr.DataTree,
    data_vars: Sequence[EcmwfAifsSingleVirtualDataVar] | None = None,
    region: slice = slice(0, 1),
    processing_mode: Literal["backfill", "update"] = "backfill",
) -> EcmwfAifsSingleForecastVirtualRegionJob:
    return EcmwfAifsSingleForecastVirtualRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=data_vars or TEMPLATE_CONFIG.data_vars,
        append_dim="init_time",
        region=region,
        reformat_job_name="test",
        processing_mode=processing_mode,
    )


def _coord(
    data_vars: Sequence[EcmwfAifsSingleVirtualDataVar],
    init_time: pd.Timestamp = _ERA2_INIT,
    lead_time: pd.Timedelta = _LEAD_6H,
) -> EcmwfAifsSingleForecastVirtualSourceFileCoord:
    return EcmwfAifsSingleForecastVirtualSourceFileCoord(
        init_time=init_time,
        lead_time=lead_time,
        data_vars=data_vars,
    )


def _index_line(
    param: str,
    levtype: str,
    offset: int,
    length: int,
    levelist: str | None = None,
) -> str:
    entry = {
        "domain": "g",
        "date": "20250301",
        "time": "0000",
        "expver": "0001",
        "class": "ai",
        "type": "fc",
        "stream": "oper",
        "step": "6",
        "levtype": levtype,
        "param": param,
        "_offset": offset,
        "_length": length,
    }
    if levelist is not None:
        entry["levelist"] = levelist
    return json.dumps(entry) + "\n"


def _fake_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str | dict[str, str]
) -> list[str]:
    """Serve index contents, optionally by URL; return the downloaded URLs."""
    downloaded: list[str] = []

    def fake_download(url: str, dataset_id: str, **kwargs: str) -> Path:
        if url.startswith("s3://"):
            assert kwargs == {"region": "eu-central-1"}
        else:
            assert not kwargs
        downloaded.append(url)
        path = tmp_path / (url.rsplit("/", 1)[-1])
        path.write_text(content[url] if isinstance(content, dict) else content)
        return path

    monkeypatch.setattr(region_job_module, "gcs_download_to_disk", fake_download)
    monkeypatch.setattr(region_job_module, "s3_download_to_disk", fake_download)
    return downloaded


def _fake_listing(
    monkeypatch: pytest.MonkeyPatch,
    *,
    on_gcs: set[tuple[pd.Timestamp, pd.Timedelta]],
    on_s3: set[tuple[pd.Timestamp, pd.Timedelta]],
) -> list[str]:
    """Serve each source's listing from the given (init, lead) sets; returns the
    location prefixes listed, in order."""
    listed: list[str] = []
    present = {"gs://ecmwf-open-data/": on_gcs, "s3://ecmwf-forecasts/": on_s3}

    def fake(
        pending: list[EcmwfAifsSingleForecastVirtualSourceFileCoord],
        *,
        store: obstore.store.ObjectStore,
        location_prefix: str,
        require_index: bool,
    ) -> list[tuple[EcmwfAifsSingleForecastVirtualSourceFileCoord, int]]:
        assert require_index is True
        assert all(coord.get_url().startswith(location_prefix) for coord in pending)
        if location_prefix.startswith("s3://"):
            assert isinstance(store, obstore.store.S3Store)
            assert store.config["region"] == "eu-central-1"
        else:
            assert isinstance(store, obstore.store.GCSStore)
        listed.append(location_prefix)
        return [
            (coord, 9000)
            for coord in pending
            if (coord.init_time, coord.lead_time) in present[location_prefix]
        ]

    monkeypatch.setattr(
        region_job_module, "discover_available_by_obstore_listing", fake
    )
    return listed


# --- URLs and out_loc ---


def test_source_file_coord_url_era1_uses_aifs_path() -> None:
    coord = _coord([get_var("temperature_2m")], init_time=_ERA1_INIT)
    assert coord.get_url() == (
        "gs://ecmwf-open-data/20240601/00z/aifs/0p25/oper/"
        "20240601000000-6h-oper-fc.grib2"
    )
    assert coord.get_index_url() == (
        "gs://ecmwf-open-data/20240601/00z/aifs/0p25/oper/"
        "20240601000000-6h-oper-fc.index"
    )


def test_source_file_coord_url_era2_uses_aifs_single_path() -> None:
    coord = _coord(
        [get_var("temperature_2m")],
        init_time=pd.Timestamp("2025-03-01T12:00"),
        lead_time=pd.Timedelta("360h"),
    )
    assert coord.get_url() == (
        "gs://ecmwf-open-data/20250301/12z/aifs-single/0p25/oper/"
        "20250301120000-360h-oper-fc.grib2"
    )


@pytest.mark.parametrize(
    ("init_time", "expected_stream_path"),
    [
        ("2025-02-24T00:00", "aifs/0p25/oper"),
        ("2025-02-24T06:00", "aifs-single/0p25/experimental/oper"),
        ("2025-02-25T00:00", "aifs-single/0p25/experimental/oper"),
        ("2025-02-25T06:00", "aifs-single/0p25/oper"),
    ],
)
def test_source_file_coord_url_spans_the_three_source_stream_paths(
    init_time: str, expected_stream_path: str
) -> None:
    """The aifs-single stream is served from an experimental/ path for its first 36 hours."""
    coord = _coord([get_var("temperature_2m")], init_time=pd.Timestamp(init_time))
    stamp = pd.Timestamp(init_time)
    assert coord.get_url() == (
        f"gs://ecmwf-open-data/{stamp.strftime('%Y%m%d')}/{stamp.strftime('%H')}z/"
        f"{expected_stream_path}/"
        f"{stamp.strftime('%Y%m%d%H')}0000-6h-oper-fc.grib2"
    )


def test_source_file_coord_s3_source_uses_same_key_in_s3_bucket() -> None:
    coord = _coord([get_var("temperature_2m")], init_time=_ERA1_INIT)
    coord.fall_back_to_s3()
    assert coord.source == "s3"
    assert coord.get_url() == (
        "s3://ecmwf-forecasts/20240601/00z/aifs/0p25/oper/"
        "20240601000000-6h-oper-fc.grib2"
    )
    assert coord.get_index_url() == (
        "s3://ecmwf-forecasts/20240601/00z/aifs/0p25/oper/"
        "20240601000000-6h-oper-fc.index"
    )


def test_s3_url_refuses_inits_on_or_after_gate() -> None:
    coord = _coord([get_var("temperature_2m")], init_time=S3_FALLBACK_BEFORE)
    coord.fall_back_to_s3()
    with pytest.raises(AssertionError, match="only older inits may reference S3"):
        coord.get_url()
    # A retimed copy carries the source with it; the URL boundary still refuses.
    old = _coord([get_var("temperature_2m")], init_time=_ERA1_INIT)
    old.fall_back_to_s3()
    retimed = old.model_copy(update={"init_time": S3_FALLBACK_BEFORE})
    with pytest.raises(AssertionError, match="only older inits may reference S3"):
        retimed.get_index_url()


def test_out_loc_pins_init_and_lead_only(template_ds: xr.DataTree) -> None:
    coord = _coord([get_var("temperature_2m"), get_var("pressure_level/temperature")])
    assert dict(coord.out_loc()) == {
        "init_time": _ERA2_INIT,
        "lead_time": _LEAD_6H,
    }
    # A group-only coord's manifest probe supplements the first level.
    group_only = _coord([get_var("pressure_level/temperature")])
    job = make_job(template_ds, data_vars=group_only.data_vars)
    probe_loc = job.representative_probe_loc(
        group_only, job.representative_var(group_only)
    )
    assert dict(probe_loc) == {
        "init_time": _ERA2_INIT,
        "lead_time": _LEAD_6H,
        "pressure_level": 1000,
    }


# --- file_refs ---


def test_file_refs_routes_root_and_soil_messages(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index = (
        _index_line("2t", "sfc", 0, 1000)
        + _index_line("sot", "sol", 1000, 500, levelist="1")
        + _index_line("sot", "sol", 1500, 500, levelist="2")
        + _index_line("skt", "sfc", 2000, 1000)  # not in data_vars -> not emitted
    )
    downloaded = _fake_index(monkeypatch, tmp_path, index)
    data_vars = [
        get_var("temperature_2m"),
        get_var("soil_temperature_layer_1"),
        get_var("soil_temperature_layer_2"),
    ]
    job = make_job(template_ds, data_vars=data_vars)
    refs = job.file_refs(_coord(data_vars), file_size=3000)

    assert downloaded == [
        (
            "gs://ecmwf-open-data/20250301/00z/aifs-single/0p25/oper/"
            "20250301000000-6h-oper-fc.index"
        )
    ]
    by_name = {r.data_var.name: r for r in refs}
    assert set(by_name) == {
        "temperature_2m",
        "soil_temperature_layer_1",
        "soil_temperature_layer_2",
    }
    assert (by_name["temperature_2m"].offset, by_name["temperature_2m"].length) == (
        0,
        1000,
    )
    layer_1 = by_name["soil_temperature_layer_1"]
    layer_2 = by_name["soil_temperature_layer_2"]
    assert (layer_1.offset, layer_1.length) == (1000, 500)
    assert (layer_2.offset, layer_2.length) == (1500, 500)
    for ref in refs:
        assert ref.out_loc == {"init_time": _ERA2_INIT, "lead_time": _LEAD_6H}
        assert ref.location == (
            "gs://ecmwf-open-data/20250301/00z/aifs-single/0p25/oper/"
            "20250301000000-6h-oper-fc.grib2"
        )


def test_file_refs_pressure_group_routes_each_level(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index = _index_line("t", "pl", 0, 1200, levelist="500") + _index_line(
        "t", "pl", 1200, 1300, levelist="50"
    )
    _fake_index(monkeypatch, tmp_path, index)
    var = get_var("pressure_level/temperature")
    job = make_job(template_ds, data_vars=[var])
    refs = job.file_refs(_coord([var]), file_size=2500)

    assert len(refs) == 2
    by_level = {ref.out_loc["pressure_level"]: ref for ref in refs}
    assert (by_level[500].offset, by_level[500].length) == (0, 1200)
    assert (by_level[50].offset, by_level[50].length) == (1200, 1300)
    for ref in refs:
        assert ref.data_var.path == "pressure_level/temperature"
        assert ref.out_loc["init_time"] == _ERA2_INIT


@pytest.mark.parametrize("lead_hour", [0, 6])
@pytest.mark.parametrize("numeric_levels", [False, True])
def test_file_refs_accepts_real_index_level_selectors(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    lead_hour: int,
    numeric_levels: bool,
) -> None:
    rows = [
        json.loads(line)
        for line in (FIXTURES_DIR / f"aifs_single_20250301_00z_{lead_hour}h.index")
        .read_text()
        .splitlines()
    ]
    if numeric_levels:
        for row in rows:
            if "levelist" in row:
                row["levelist"] = int(row["levelist"])
    data_vars = [get_var("temperature_2m"), get_var("pressure_level/temperature")]
    _fake_index(monkeypatch, tmp_path, "\n".join(json.dumps(row) for row in rows))
    file_size = max(row["_offset"] + row["_length"] for row in rows)
    refs = make_job(template_ds, data_vars).file_refs(
        _coord(data_vars, lead_time=pd.Timedelta(hours=lead_hour)), file_size
    )
    assert len(refs) == len(PRESSURE_LEVELS)
    assert not [record for record in caplog.records if record.levelname == "ERROR"]


def test_file_refs_missing_level_yields_no_ref(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # q has no 10 hPa level in the source; only listed levels produce refs.
    index = _index_line("q", "pl", 0, 1000, levelist="1000")
    _fake_index(monkeypatch, tmp_path, index)
    var = get_var("pressure_level/specific_humidity")
    job = make_job(template_ds, data_vars=[var])
    refs = job.file_refs(_coord([var]), file_size=1000)

    assert [ref.out_loc["pressure_level"] for ref in refs] == [1000]


def test_file_refs_skips_stale_index_past_eof(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_index(
        monkeypatch,
        tmp_path,
        _index_line("2t", "sfc", 0, 100) + _index_line("10u", "sfc", 500, 1000),
    )
    data_vars = [get_var("temperature_2m")]
    job = make_job(template_ds, data_vars=data_vars)
    # An unselected row past EOF invalidates the earlier, in-bounds ref too.
    assert job.file_refs(_coord(data_vars), file_size=1200) == []


def test_file_refs_quarantines_decreasing_byte_ranges(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _fake_index(
        monkeypatch,
        tmp_path,
        _index_line("2t", "sfc", 200, 100) + _index_line("10u", "sfc", 0, 100),
    )
    data_vars = [get_var("temperature_2m")]
    assert make_job(template_ds, data_vars).file_refs(_coord(data_vars), 300) == []
    assert "untrustworthy index byte geometry" in caplog.text


@pytest.mark.parametrize("processing_mode", ["update", "backfill"])
@pytest.mark.parametrize(
    ("bad_row", "quarantine"),
    [
        ("{bad json", False),
        ("[]", False),
        ("null", False),
        *[
            (
                json.dumps(
                    {
                        "param": "10u",
                        "levtype": "sfc",
                        "_offset": 100,
                        "_length": 100,
                        **change,
                    }
                ),
                "_offset" in change or "_length" in change,
            )
            for change in [
                {"param": ["10u"]},
                {"levtype": {}},
                {"levelist": []},
                {"levelist": "bad"},
                {"levelist": 1.5},
                {"param": None},
                {"_offset": -1},
                {"_offset": 0},
                {"_offset": 50},
                {"_offset": 1.5},
                {"_offset": True},
                {"_offset": "bad"},
                {"_length": 0},
                {"_length": -1},
                {"_length": []},
                {"_length": 10000},
            ]
        ],
        ('{"param":"10u","levtype":"sfc"}', True),
        (_index_line("unselected", "sfc", 50, 100), True),
        (_index_line("", "sfc", 50, 100), True),
    ],
)
def test_source_errors_preserve_committed_sibling_files(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    processing_mode: Literal["update", "backfill"],
    bad_row: str,
    quarantine: bool,
) -> None:
    data_vars = [
        get_var(name)
        for name in ("temperature_2m", "wind_u_10m", "skin_temperature_surface")
    ]
    init_time = pd.Timestamp(template_ds.to_dataset().get_index("init_time")[0])
    template = xr.DataTree(
        template_ds.to_dataset()[[var.name for var in data_vars]].sel(
            init_time=[init_time]
        )
    )
    job = make_job(template, data_vars, processing_mode=processing_mode)
    coords = [
        _coord(data_vars, init_time=init_time, lead_time=pd.Timedelta(hours=hour))
        for hour in (0, 6)
    ]
    _fake_index(
        monkeypatch,
        tmp_path,
        {
            coords[0].get_index_url(): _index_line("2t", "sfc", 0, 100)
            + bad_row
            + "\n"
            + _index_line("skt", "sfc", 200, 100),
            coords[1].get_index_url(): _index_line("2t", "sfc", 0, 100)
            + _index_line("10u", "sfc", 100, 100)
            + _index_line("skt", "sfc", 200, 100),
        },
    )
    monkeypatch.setattr(
        EcmwfAifsSingleForecastVirtualRegionJob,
        "discover_available",
        lambda self, pending: [(coord, 300) for coord in pending],
    )
    config, credentials = _repository_config_and_credentials(
        IcechunkVirtualConfig(
            containers=aifs_single_virtual_chunk_containers(),
            manifest_split=manifest_append_dim_split(split_size=32, dim="init_time"),
        )
    )
    repo = icechunk.Repository.create(
        icechunk.in_memory_storage(),
        config=config,
        authorize_virtual_chunk_access=credentials,
    )
    session = repo.writable_session("main")
    template.to_zarr(session.store, compute=False, consolidated=False)
    session.commit("template")

    job.process_virtual(repo, [], "main", coords)

    store = repo.readonly_session("main").store
    for lead_index in (0, 1):
        assert asyncio.run(store.exists(f"temperature_2m/c/0/{lead_index}/0/0")) == (
            lead_index == 1 or not quarantine
        )
        assert asyncio.run(
            store.exists(f"skin_temperature_surface/c/0/{lead_index}/0/0")
        ) == (lead_index == 1 or not quarantine)
        assert asyncio.run(store.exists(f"wind_u_10m/c/0/{lead_index}/0/0")) == (
            lead_index == 1
        )
    assert job.filter_already_present(coords, store) == (
        [coords[0]] if quarantine else []
    )
    assert any(
        record.levelname == "ERROR" and "Skipping" in record.message
        for record in caplog.records
    )
    assert ("Missing 1 expected messages" in caplog.text) == (not quarantine)
    assert ("untrustworthy index byte geometry" in caplog.text) == quarantine


@pytest.mark.parametrize("init_time", [_ERA2_INIT, AIFS_2026_UPGRADE_DATE])
@pytest.mark.parametrize("param", ["t", "q"])
def test_missing_message_logging_accounts_for_known_pressure_levels(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    init_time: pd.Timestamp,
    param: str,
) -> None:
    var = get_var(
        "pressure_level/temperature"
        if param == "t"
        else "pressure_level/specific_humidity"
    )
    content = "".join(
        _index_line(param, "pl", index * 100, 100, str(level))
        for index, level in enumerate(PRESSURE_LEVELS)
        if level != 10
    )
    _fake_index(monkeypatch, tmp_path, content)
    job = make_job(template_ds, [var])
    assert (
        len(job.file_refs(_coord([var], init_time=init_time), 10000))
        == len(PRESSURE_LEVELS) - 1
    )
    assert ("Missing 1 expected messages" in caplog.text) == (
        param == "t" and init_time >= AIFS_2026_UPGRADE_DATE
    )


def test_file_refs_s3_source_reads_index_and_points_refs_at_s3(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    downloaded = _fake_index(monkeypatch, tmp_path, _index_line("2t", "sfc", 0, 1000))
    data_vars = [get_var("temperature_2m")]
    coord = _coord(data_vars, init_time=_ERA1_INIT)
    coord.fall_back_to_s3()
    job = make_job(template_ds, data_vars=data_vars)

    (ref,) = job.file_refs(coord, file_size=1000)

    assert downloaded == [
        "s3://ecmwf-forecasts/20240601/00z/aifs/0p25/oper/20240601000000-6h-oper-fc.index"
    ]
    assert ref.location == (
        "s3://ecmwf-forecasts/20240601/00z/aifs/0p25/oper/20240601000000-6h-oper-fc.grib2"
    )


# --- discover_available ---


def test_discover_available_lists_gcs_requiring_index(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_vars = [get_var("temperature_2m")]
    coord = _coord(data_vars)
    captured: dict[str, object] = {}

    def fake(
        pending: list[EcmwfAifsSingleForecastVirtualSourceFileCoord], **kwargs: object
    ) -> list[tuple[EcmwfAifsSingleForecastVirtualSourceFileCoord, int]]:
        captured.update(kwargs)
        return [(pending[0], 9000)]

    monkeypatch.setattr(
        region_job_module, "discover_available_by_obstore_listing", fake
    )
    job = make_job(template_ds, data_vars=data_vars)

    result = job.discover_available([coord])

    assert len(result) == 1
    assert result[0][0] is coord
    # AIFS data files always land with a .index sidecar; a file isn't ready until both exist.
    assert captured["require_index"] is True
    assert captured["location_prefix"] == "gs://ecmwf-open-data/"
    assert isinstance(captured["store"], obstore.store.GCSStore)


def test_discover_available_falls_back_to_s3_before_gate(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_vars = [get_var("temperature_2m")]
    init = S3_FALLBACK_BEFORE - pd.Timedelta("6h")
    coord = _coord(data_vars, init_time=init)
    listed = _fake_listing(monkeypatch, on_gcs=set(), on_s3={(init, _LEAD_6H)})
    job = make_job(template_ds, data_vars=data_vars)

    result = job.discover_available([coord])

    assert listed == ["gs://ecmwf-open-data/", "s3://ecmwf-forecasts/"]
    assert len(result) == 1
    # The write loop drops discovered coords by identity, so the pending coord
    # itself is returned, switched to its S3 URL.
    assert result[0][0] is coord
    assert coord.source == "s3"
    assert coord.get_url().startswith("s3://ecmwf-forecasts/")


def test_discover_available_never_falls_back_on_or_after_gate(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_vars = [get_var("temperature_2m")]
    coord = _coord(data_vars, init_time=S3_FALLBACK_BEFORE)
    listed = _fake_listing(
        monkeypatch, on_gcs=set(), on_s3={(S3_FALLBACK_BEFORE, _LEAD_6H)}
    )
    job = make_job(template_ds, data_vars=data_vars)

    assert job.discover_available([coord]) == []
    assert listed == ["gs://ecmwf-open-data/"]
    assert coord.source == "gcs"


def test_discover_available_keeps_gcs_url_when_s3_also_lacks_the_file(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_vars = [get_var("temperature_2m")]
    coord = _coord(data_vars, init_time=_ERA1_INIT)
    listed = _fake_listing(monkeypatch, on_gcs=set(), on_s3=set())
    job = make_job(template_ds, data_vars=data_vars)

    assert job.discover_available([coord]) == []
    assert listed == ["gs://ecmwf-open-data/", "s3://ecmwf-forecasts/"]
    assert coord.source == "gcs"


def test_discover_available_partial_init_falls_back_only_for_the_absent_file(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An init GCS holds all but one file of reads that one file from S3."""
    data_vars = [get_var("temperature_2m")]
    init = pd.Timestamp("2024-04-17T12:00")
    lead_0 = _coord(data_vars, init_time=init, lead_time=pd.Timedelta(0))
    lead_6 = _coord(data_vars, init_time=init, lead_time=_LEAD_6H)
    listed = _fake_listing(
        monkeypatch,
        on_gcs={(init, _LEAD_6H)},
        on_s3={(init, pd.Timedelta(0)), (init, _LEAD_6H)},
    )
    job = make_job(template_ds, data_vars=data_vars)

    result = job.discover_available([lead_0, lead_6])

    assert listed == ["gs://ecmwf-open-data/", "s3://ecmwf-forecasts/"]
    assert {id(coord) for coord, _ in result} == {id(lead_6), id(lead_0)}
    assert lead_6.source == "gcs"
    assert lead_0.source == "s3"


def test_discover_available_skips_s3_when_gcs_serves_every_file(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_vars = [get_var("temperature_2m")]
    coord = _coord(data_vars, init_time=_ERA1_INIT)
    listed = _fake_listing(
        monkeypatch, on_gcs={(_ERA1_INIT, _LEAD_6H)}, on_s3={(_ERA1_INIT, _LEAD_6H)}
    )
    job = make_job(template_ds, data_vars=data_vars)

    assert job.discover_available([coord]) == [(coord, 9000)]
    assert listed == ["gs://ecmwf-open-data/"]
    assert coord.source == "gcs"


def test_update_loop_offers_only_unresolved_gcs_coords_on_later_ticks(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Over two polling ticks, a coord served from S3 on the first tick leaves the
    loop, and only the still-pending GCS-sourced coord reaches the second listing."""
    data_vars = [get_var("temperature_2m")]
    init = _ERA1_INIT
    on_s3 = _coord(data_vars, init_time=init, lead_time=pd.Timedelta(0))
    late_on_gcs = _coord(data_vars, init_time=init, lead_time=_LEAD_6H)
    _fake_index(monkeypatch, tmp_path, _index_line("2t", "sfc", 0, 1000))
    listings: list[tuple[str, list[str]]] = []

    def fake(
        pending: list[EcmwfAifsSingleForecastVirtualSourceFileCoord],
        *,
        store: obstore.store.ObjectStore,
        location_prefix: str,
        require_index: bool,
    ) -> list[tuple[EcmwfAifsSingleForecastVirtualSourceFileCoord, int]]:
        listings.append((location_prefix, [c.get_url() for c in pending]))
        gcs_calls = sum(prefix.startswith("gs://") for prefix, _ in listings)
        if location_prefix.startswith("s3://"):
            return [(c, 1000) for c in pending if c.lead_time == pd.Timedelta(0)]
        if gcs_calls >= 2:
            return [(c, 1000) for c in pending if c.lead_time == _LEAD_6H]
        return []

    monkeypatch.setattr(
        region_job_module, "discover_available_by_obstore_listing", fake
    )
    job = make_job(template_ds, data_vars=data_vars, processing_mode="update")
    job = job.model_copy(update={"tick_interval": pd.Timedelta(0)})

    batches = list(job.process_virtual_refs([on_s3, late_on_gcs]))

    assert [[c for c, _ in batch] for batch in batches] == [[on_s3], [late_on_gcs]]
    assert batches[0][0][1][0].location.startswith("s3://ecmwf-forecasts/")
    assert batches[1][0][1][0].location.startswith("gs://ecmwf-open-data/")
    assert [prefix for prefix, _ in listings] == [
        "gs://ecmwf-open-data/",
        "s3://ecmwf-forecasts/",
        "gs://ecmwf-open-data/",
    ]
    # The S3 listing probed copies of both files GCS lacked; the still-pending
    # coord kept its GCS URL for the second GCS listing.
    assert listings[1][1] == [
        on_s3.get_url(),
        late_on_gcs.get_url().replace("gs://ecmwf-open-data/", "s3://ecmwf-forecasts/"),
    ]
    assert listings[2][1] == [late_on_gcs.get_url()]


# --- generate_source_file_coords ---


def test_generate_source_file_coords_filters_by_era_and_lead_0(
    template_ds: xr.DataTree,
) -> None:
    data_vars = [
        get_var("temperature_2m"),  # all eras, all leads
        get_var("total_precipitation_run_total_surface"),  # from 2025-02-26
        get_var("land_sea_mask_surface"),  # all eras, lead 0 only
        get_var("pressure_level/temperature"),  # all eras, all leads
    ]
    job = make_job(template_ds, data_vars=data_vars)
    # The template starts 2024-04-01 (before the 2025-02-26 format change).
    region_ds = (
        template_ds.to_dataset()
        .isel(init_time=slice(0, 1))
        .sel(lead_time=[pd.Timedelta("0h"), pd.Timedelta("6h")])
    )

    coords = job.generate_source_file_coords(region_ds, data_vars)
    by_lead = {c.lead_time: c for c in coords}

    assert len(coords) == 2
    assert {v.name for v in by_lead[pd.Timedelta("0h")].data_vars} == {
        "temperature_2m",
        "land_sea_mask_surface",
        "temperature",
    }
    assert {v.name for v in by_lead[pd.Timedelta("6h")].data_vars} == {
        "temperature_2m",
        "temperature",
    }


def test_generate_source_file_coords_era2_includes_expanded_vars(
    template_ds: xr.DataTree,
) -> None:
    data_vars = [
        get_var("temperature_2m"),
        get_var("total_precipitation_run_total_surface"),
    ]
    job = make_job(template_ds, data_vars=data_vars)
    region_ds = (
        template_ds.to_dataset()
        .isel(init_time=slice(0, 1))
        .sel(lead_time=[pd.Timedelta("6h")])
    )
    # Relabel the region's init to an era2 init; only coordinate values matter here.
    region_ds = region_ds.assign_coords(init_time=[_ERA2_INIT])

    (coord,) = job.generate_source_file_coords(region_ds, data_vars)
    assert {v.name for v in coord.data_vars} == {
        "temperature_2m",
        "total_precipitation_run_total_surface",
    }


# --- file_refs against a real index ---


@pytest.mark.parametrize(
    ("lead_time", "index_fixture", "file_size"),
    [
        # Real sidecars for the 2025-03-01T00 init: lead 0 carries the statics, lead 6
        # carries the accumulations. file_size is the real .grib2 Content-Length, so a
        # byte range running past it would be rejected as a stale index.
        (pd.Timedelta("0h"), "aifs_single_20250301_00z_0h.index", 79_888_489),
        (pd.Timedelta("6h"), "aifs_single_20250301_00z_6h.index", 77_642_846),
    ],
)
def test_file_refs_covers_every_variable_in_a_real_index(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lead_time: pd.Timedelta,
    index_fixture: str,
    file_size: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every variable a real source file carries gets a ref from that file's real index.

    The file_refs tests above feed synthetic index lines, so they check our routing
    against our *assumption* of how messages are labelled. This checks the assumption
    itself -- that the published param / levtype / levelist match each variable's
    grib_index_* attrs -- across every declared variable, rather than the handful the
    slow integration test samples.
    """
    _fake_index(
        monkeypatch, tmp_path, (FIXTURES_DIR / index_fixture).read_text("utf-8")
    )
    data_vars = TEMPLATE_CONFIG.data_vars
    job = make_job(template_ds, data_vars=data_vars)
    region_ds = (
        template_ds.to_dataset()
        .isel(init_time=slice(0, 1))
        .sel(lead_time=[lead_time])
        .assign_coords(init_time=[_ERA2_INIT])
    )

    (coord,) = job.generate_source_file_coords(region_ds, data_vars)
    refs = job.file_refs(coord, file_size=file_size)

    expected = {var.path for var in coord.data_vars}
    assert expected, "the coord should carry variables at this lead time"
    assert {ref.data_var.path for ref in refs} == expected
    assert not [record for record in caplog.records if record.levelname == "ERROR"]


def test_operational_update_jobs_single_polling_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A production fire time: init+5h42m for the 2025-03-01T00 init.
    now = pd.Timestamp("2025-03-01T05:42")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda *a, **kw: now))

    jobs, template_ds = EcmwfAifsSingleForecastVirtualRegionJob.operational_update_jobs(
        primary_store=Mock(),
        tmp_store=Path("unused-tmp.zarr"),
        get_template_fn=TEMPLATE_CONFIG.get_template,
        append_dim="init_time",
        all_data_vars=TEMPLATE_CONFIG.data_vars,
        reformat_job_name="test",
    )
    (job,) = jobs
    assert isinstance(job, EcmwfAifsSingleForecastVirtualRegionJob)
    assert job.processing_mode == "update"
    init_times = template_ds.to_dataset().get_index("init_time")
    assert init_times[-1] <= now
    assert now - init_times[-1] < TEMPLATE_CONFIG.append_dim_frequency
    # 20h window at a fire time = the just-publishing init + 2 prior cycles.
    assert job.region == slice(len(init_times) - 3, len(init_times))
