import hashlib
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import icechunk
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from zarr.storage import LocalStore, MemoryStore

from reformatters.common import shared_memory_utils
from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.common.iterating import item
from reformatters.common.materialized_region_job import MaterializedRegionJob
from reformatters.common.pydantic import replace
from reformatters.common.region_job import SourceFileResult, SourceFileStatus
from reformatters.common.storage import StoreFactory
from reformatters.common.types import ArrayFloat32
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.source_file_coord import (
    MARS_OPEN_DATA_CUTOVER,
    IfsEnsSourceFileCoord,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
)
from scripts import retained_ifs_repair_stage as stage


@dataclass
class _Unit:
    template: xr.DataTree
    temporary: Path
    staging: LocalStore
    requests: Path
    variable: str

    def run(self) -> dict[str, Any]:
        return stage.regenerate_unit(
            self.template, self.variable, 1, self.temporary, self.staging, self.requests
        )

    def receipt(self) -> dict[str, Any]:
        return json.loads(
            item(
                self.temporary.parent.glob("retained-ifs-stage-*/receipt.json")
            ).read_bytes()
        )


@pytest.fixture(autouse=True)
def forbid_template_builders_and_writable_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = Mock(side_effect=AssertionError("Forbidden production entry point"))
    monkeypatch.setattr(
        EcmwfIfsEnsForecast15Day025DegreeTemplateConfig, "get_template", forbidden
    )
    monkeypatch.setattr(
        EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
        "dimension_coordinates",
        forbidden,
    )
    monkeypatch.setattr(StoreFactory, "primary_store", forbidden)
    monkeypatch.setattr(StoreFactory, "replica_stores", forbidden)
    monkeypatch.setattr(MaterializedRegionJob, "process", forbidden)
    monkeypatch.setattr(MaterializedRegionJob, "process_worker_jobs", forbidden)
    monkeypatch.setattr(icechunk.Repository, "writable_session", forbidden)
    monkeypatch.setattr(icechunk.Repository, "create_branch", forbidden)


@pytest.fixture(autouse=True)
def local_buffer_and_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    buffers: dict[str, bytearray] = {}

    class Buffer:
        def __init__(
            self, name: str | None = None, create: bool = False, size: int = 0
        ) -> None:
            self.name = name or "synthetic-buffer"
            if create:
                buffers[self.name] = bytearray(size)
            self.buf = buffers[self.name]

        def close(self) -> None:
            pass

        def unlink(self) -> None:
            del buffers[self.name]

    def executor(
        max_workers: int, initializer: Callable[[int], None], initargs: tuple[int]
    ) -> ThreadPoolExecutor:
        assert max_workers <= 4
        assert initargs == (1,)
        return ThreadPoolExecutor(
            max_workers=max_workers, initializer=initializer, initargs=initargs
        )

    monkeypatch.setattr(shared_memory_utils, "SharedMemory", Buffer)
    monkeypatch.setattr(stage, "ProcessPoolExecutor", executor)


@pytest.fixture
def unit_factory(tmp_path: Path) -> Iterator[Callable[..., _Unit]]:
    opened: list[xr.DataTree] = []

    def make(variable: str = "temperature_2m", *, mars: bool = False) -> _Unit:
        data_var = item(
            var
            for var in EcmwfIfsEnsForecast15Day025DegreeTemplateConfig().data_vars
            if var.name == variable
        )
        root = tmp_path / str(len(opened))
        root.mkdir()
        init = MARS_OPEN_DATA_CUTOVER + pd.Timedelta(days=-2 if mars else 2)
        ds = xr.Dataset(
            {
                variable: (
                    (
                        "init_time",
                        "lead_time",
                        "ensemble_member",
                        "latitude",
                        "longitude",
                    ),
                    np.full((2, 3, 2, 2, 2), 91, dtype=np.float32),
                    {"units": data_var.attrs.units, "_FillValue": np.float32(np.nan)},
                ),
            },
            coords={
                "init_time": pd.date_range(init, periods=2),
                "lead_time": pd.to_timedelta([0, 3, 6], unit="h"),
                "ensemble_member": np.array([0, 1], dtype=np.int64),
                "latitude": [1.0, 0.0],
                "longitude": [0.0, 1.0],
                "expected_forecast_length": (
                    "init_time",
                    pd.to_timedelta([6, 6], unit="h"),
                ),
                "ingested_forecast_length": (
                    "init_time",
                    pd.to_timedelta([6, 6], unit="h"),
                ),
                "spatial_ref": 0,
            },
            attrs={"dataset_id": "ecmwf-ifs-ens-forecast-15-day-0-25-degree"},
        )
        ds = ds.assign_coords(valid_time=ds.init_time + ds.lead_time)
        baseline = root / "baseline.zarr"
        ds.to_zarr(
            baseline,
            zarr_format=3,
            consolidated=False,
            encoding={
                variable: {
                    "chunks": (1, 1, 1, 1, 1),
                    "shards": (1, 3, 2, 1, 1),
                    "fill_value": np.nan,
                }
            },
        )
        template = xr.open_datatree(
            baseline, engine="zarr", chunks={}, consolidated=False
        )
        opened.append(template)
        temporary = root / "temporary.zarr"
        staging_root = root / "staging.zarr"
        for path in (temporary, staging_root):
            shutil.copytree(baseline, path)
            shutil.rmtree(path / variable / "c")
        job = stage._StrictRegionJob(
            tmp_store=temporary,
            template_ds=template,
            data_vars=[data_var],
            append_dim="init_time",
            region=slice(1, 2),
            reformat_job_name="synthetic",
        )
        coords = job.generate_source_file_coords(ds.isel(init_time=[1]), [data_var])
        indexes: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        entries: list[dict[str, Any]] = []
        for coord in coords:
            selected = item(coord.data_var_group).internal_attrs
            lead, member = stage._key(coord)
            offset = lead * 100 + member * 10
            row: dict[str, Any] = {
                "param": selected.grib_index_param,
                "levtype": selected.grib_index_level_type,
                "number": member,
                "step": lead,
                "type": "cf" if member == 0 else "pf",
                "_offset": offset,
                "_length": 4,
            }
            if not np.isnan(selected.grib_index_level_value):
                row["levelist"] = selected.grib_index_level_value
            url = stage._url(coord)
            indexes[url].append(row)
            entries.append(
                {
                    "lead_hours": lead,
                    "member": member,
                    "url": url,
                    "starts": [offset],
                    "ends": [offset + 4],
                }
            )
        (root / "indexes").mkdir()
        for url, rows in indexes.items():
            raw = "\n".join(json.dumps(row) for row in rows).encode()
            digest = hashlib.sha256(raw).hexdigest()
            relative = f"indexes/{digest}.index"
            (root / relative).write_bytes(raw)
            for entry in entries:
                if entry["url"] == url:
                    entry.update(index_path=relative, index_sha256=digest)
        requests = root / "requests.json"
        requests.write_text(json.dumps(entries))
        return _Unit(template, temporary, LocalStore(staging_root), requests, variable)

    yield make
    for tree in opened:
        tree.close()


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch) -> Mock:
    def download(
        url: str,
        dataset_id: str,
        byte_ranges: tuple[Sequence[int], Sequence[int]] | None = None,
        local_path_suffix: str = "",
        disk_cache: bool = False,
    ) -> Path:
        assert byte_ranges is not None
        assert not disk_cache
        assert dataset_id == "ecmwf-ifs-ens-forecast-15-day-0-25-degree"
        assert Path.cwd().name.startswith("retained-ifs-stage-")
        path = Path("data") / (
            hashlib.sha256(url.encode()).hexdigest() + local_path_suffix
        )
        assert not path.exists()
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"GRIB")
        return path

    mock = Mock(side_effect=download)
    monkeypatch.setattr(stage, "http_download_to_disk", mock)
    return mock


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> Mock:
    def read(coord: IfsEnsSourceFileCoord, data_var: EcmwfDataVar) -> ArrayFloat32:
        assert coord.downloaded_path is not None
        assert coord.downloaded_path.read_bytes() == b"GRIB"
        return np.full(
            (2, 2),
            1 + stage._key(coord)[0] / 3 + coord.ensemble_member,
            dtype=np.float32,
        )

    mock = Mock(side_effect=read)
    monkeypatch.setattr(stage._StrictRegionJob, "read_data", mock)
    return mock


@pytest.mark.parametrize("mars", [False, True])
def test_regenerate_b_unit_with_pinned_real_selector(
    unit_factory: Callable[..., _Unit], downloads: Mock, reads: Mock, mars: bool
) -> None:
    unit = unit_factory(mars=mars)
    before = stage._protected_hashes(unit.temporary, unit.template)
    receipt = unit.run()
    assert receipt["status"] == "succeeded"
    assert receipt["expected_count"] == 6
    assert receipt["read_count"] == 6
    assert receipt["intentional_absent_coords"] == []
    assert receipt["transform"] == {"status": "succeeded"}
    assert receipt["metadata_unchanged"]
    assert receipt["derived_coords_checked"] == [
        "expected_forecast_length",
        "ingested_forecast_length",
        "spatial_ref",
        "valid_time",
    ]
    assert downloads.call_count == 6
    assert reads.call_count == 6
    assert stage._protected_hashes(unit.temporary, unit.template) == before
    assert stage._protected_hashes(unit.staging.root, unit.template) == before
    assert len(receipt["written_keys"]) == 4
    assert all(
        key.startswith(f"{unit.variable}/c/1/") for key in receipt["written_keys"]
    )
    for source, call in zip(receipt["sources"], downloads.call_args_list, strict=True):
        assert Path(source["path"]).read_bytes() == b"GRIB"
        assert Path(source["path"]).stat().st_mode & 0o222 == 0
        assert source["sha256"] == hashlib.sha256(b"GRIB").hexdigest()
        assert (
            call.kwargs["local_path_suffix"]
            == "-"
            + hashlib.sha256(
                f"{source['starts'][0]}-{source['ends'][0]}".encode()
            ).hexdigest()
        )
    assert Path(receipt["receipt_path"]).stat().st_mode & 0o222 == 0
    assert unit.receipt() == receipt
    actual = xr.open_zarr(unit.staging, chunks=None, consolidated=False)
    try:
        assert np.isnan(actual[unit.variable].isel(init_time=0)).all()
        np.testing.assert_array_equal(
            actual[unit.variable].isel(init_time=1).values[:, :, 0, 0],
            [[1, 2], [2, 3], [3, 4]],
        )
    finally:
        actual.close()


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "extra",
        "url",
        "sha",
        "selector",
        "index",
        "absolute",
        "escape",
    ],
)
def test_all_request_gates_precede_any_download(
    unit_factory: Callable[..., _Unit], downloads: Mock, reads: Mock, fault: str
) -> None:
    unit = unit_factory()
    requests = json.loads(unit.requests.read_bytes())
    last = requests[-1]
    if fault == "missing":
        requests.pop()
    elif fault == "duplicate":
        requests.append(last)
    elif fault == "extra":
        requests.append(last | {"member": 99})
    elif fault == "url":
        last["url"] = last["url"].replace(
            "storage.googleapis.com", "ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
        )
    elif fault == "sha":
        last["index_sha256"] = "0" * 64
    elif fault == "selector":
        last["starts"][0] += 1
    elif fault == "index":
        index = unit.requests.parent / last["index_path"]
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        rows[-1]["param"] = "wrong"
        index.write_text("\n".join(json.dumps(row) for row in rows))
        last["index_sha256"] = stage._hash(index)
    elif fault == "absolute":
        last["index_path"] = str(unit.requests.parent / last["index_path"])
    else:
        last["index_path"] = "../outside.index"
    unit.requests.write_text(json.dumps(requests))
    with pytest.raises((AssertionError, KeyError)):
        unit.run()
    downloads.assert_not_called()
    reads.assert_not_called()
    assert unit.receipt()["status"] == "failed"
    assert unit.receipt()["metadata_unchanged"]


def test_shared_indexes_snapshotted_once_with_real_selection_per_coordinate(
    unit_factory: Callable[..., _Unit],
    downloads: Mock,
    reads: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit = unit_factory(mars=True)
    entries = json.loads(unit.requests.read_bytes())
    original_paths = {
        (unit.requests.parent / entry["index_path"]).resolve() for entry in entries
    }
    source_reads: Counter[Path] = Counter()
    read_bytes = Path.read_bytes

    def counted_read(path: Path) -> bytes:
        if path in original_paths:
            source_reads[path] += 1
        return read_bytes(path)

    selector = Mock(wraps=stage.grib_message_byte_ranges_from_index)
    monkeypatch.setattr(Path, "read_bytes", counted_read)
    monkeypatch.setattr(stage, "grib_message_byte_ranges_from_index", selector)
    receipt = unit.run()
    assert source_reads == Counter(dict.fromkeys(original_paths, 1))
    assert selector.call_count == receipt["expected_count"] == 6
    pinned_paths = {call.args[0] for call in selector.call_args_list}
    assert len(pinned_paths) == len(original_paths) == 2
    assert pinned_paths.isdisjoint(original_paths)
    assert all(path.stat().st_mode & 0o222 == 0 for path in pinned_paths)


def test_conflicting_index_sha_for_same_resolved_path_blocks_downloads(
    unit_factory: Callable[..., _Unit],
    downloads: Mock,
    reads: Mock,
) -> None:
    unit = unit_factory(mars=True)
    entries = json.loads(unit.requests.read_bytes())
    assert entries[0]["index_path"] == entries[2]["index_path"]
    entries[2]["index_sha256"] = "0" * 64
    unit.requests.write_text(json.dumps(entries))
    with pytest.raises(AssertionError, match="Conflicting SHA256"):
        unit.run()
    downloads.assert_not_called()
    reads.assert_not_called()


@pytest.mark.parametrize("failure", ["download", "read", "transform", "coordinates"])
def test_failures_propagate_without_stage_writes(
    unit_factory: Callable[..., _Unit],
    downloads: Mock,
    reads: Mock,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    unit = unit_factory()
    before = stage._protected_hashes(unit.staging.root, unit.template)
    error = ValueError(f"injected {failure}")
    if failure == "download":
        downloads.side_effect = error
    elif failure == "read":
        reads.side_effect = error
    elif failure == "transform":
        monkeypatch.setattr(
            stage._StrictRegionJob,
            "apply_data_transformations",
            Mock(side_effect=error),
        )
    else:

        def changed_coordinates(
            self: stage._StrictRegionJob, results: dict[str, list[SourceFileResult]]
        ) -> xr.DataTree:
            changed = self.template_ds.copy(deep=True)
            changed["ingested_forecast_length"] = (
                changed.ingested_forecast_length + np.timedelta64(1, "h")
            )
            return changed

        monkeypatch.setattr(
            stage._StrictRegionJob, "update_template_with_results", changed_coordinates
        )
    with pytest.raises((ValueError, AssertionError)):
        unit.run()
    receipt = unit.receipt()
    assert receipt["status"] == "failed"
    assert receipt["metadata_unchanged"]
    assert receipt["derived_coords_checked"] == []
    assert not (unit.staging.root / unit.variable / "c").exists()
    assert stage._protected_hashes(unit.staging.root, unit.template) == before
    assert "error" in receipt
    if failure == "read":
        assert receipt["read_count"] == 0
        assert len(receipt["sources"]) == 1
        assert Path(receipt["sources"][0]["path"]).is_file()
    if failure == "download":
        assert downloads.call_count == 1
        reads.assert_not_called()


@pytest.mark.parametrize("kind", ["invalid", "clamp", "mixed"])
def test_deaccumulation_counts_and_error_are_retained(
    unit_factory: Callable[..., _Unit], downloads: Mock, reads: Mock, kind: str
) -> None:
    unit = unit_factory("precipitation_surface")

    def decreasing(
        coord: IfsEnsSourceFileCoord, data_var: EcmwfDataVar
    ) -> ArrayFloat32:
        lead = stage._key(coord)[0]
        if lead < 6:
            return np.full((2, 2), 1e-5 if lead else 0, dtype=np.float32)
        result = np.full((2, 2), -1 if kind == "invalid" else 0, dtype=np.float32)
        if kind == "mixed":
            result[0, :] = -1
        return result

    reads.side_effect = decreasing
    with pytest.raises(
        ValueError, match="clamped" if kind == "clamp" else "below threshold"
    ):
        unit.run()
    receipt = unit.receipt()
    assert receipt["read_count"] == 6
    assert receipt["transform"]["status"] == "failed"
    assert (
        receipt["transform"]["invalid_negative_count"]
        == {"invalid": 8, "clamp": 0, "mixed": 4}[kind]
    )
    assert (
        receipt["transform"]["clamped_count"]
        == {"invalid": 0, "clamp": 8, "mixed": 4}[kind]
    )
    assert receipt["transform"]["error"] in receipt["error"]
    assert not (unit.staging.root / unit.variable / "c").exists()


def test_generated_lead_zero_omissions_are_enumerated(
    unit_factory: Callable[..., _Unit], downloads: Mock, reads: Mock
) -> None:
    unit = unit_factory("wind_gust_10m", mars=True)
    receipt = unit.run()
    assert receipt["intentional_absent_coords"] == [
        {"lead_hours": 0, "member": 0},
        {"lead_hours": 0, "member": 1},
    ]
    assert receipt["expected_count"] == 4
    assert receipt["read_count"] == 4
    actual = xr.open_zarr(unit.staging, chunks=None, consolidated=False)
    try:
        assert np.isnan(actual[unit.variable].isel(init_time=1, lead_time=0)).all()
        assert np.isfinite(
            actual[unit.variable].isel(init_time=1, lead_time=slice(1, None))
        ).all()
    finally:
        actual.close()


def test_unexpected_generated_omission_blocks_before_download(
    unit_factory: Callable[..., _Unit],
    downloads: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit = unit_factory()
    generate = stage._StrictRegionJob.generate_source_file_coords

    def omit(
        self: stage._StrictRegionJob, ds: xr.Dataset, variables: Sequence[EcmwfDataVar]
    ) -> Sequence[IfsEnsSourceFileCoord]:
        return list(generate(self, ds, variables))[:-1]

    monkeypatch.setattr(stage._StrictRegionJob, "generate_source_file_coords", omit)
    with pytest.raises(AssertionError, match="Unexpected missing"):
        unit.run()
    downloads.assert_not_called()


def test_strict_transform_matches_production_order_and_rounding() -> None:
    config = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    var = item(var for var in config.data_vars if var.name == "precipitation_surface")
    array = xr.DataArray(
        np.array([0, 0.02123456, 0.05012345], dtype=np.float32).reshape(1, 3, 1),
        dims=("init_time", "lead_time", "x"),
        coords={"lead_time": pd.to_timedelta([0, 3, 6], unit="h")},
        attrs={"units": var.attrs.units},
    )
    expected = array.copy(deep=True)
    factor = var.internal_attrs.scale_factor
    reset = var.internal_attrs.window_reset_frequency
    threshold = var.internal_attrs.deaccumulation_invalid_below_threshold_rate
    assert factor is not None
    assert reset is not None
    assert threshold is not None
    expected *= factor
    deaccumulate_to_rates_inplace(
        expected,
        dim="lead_time",
        reset_frequency=reset,
        invalid_below_threshold_rate=threshold,
        expected_invalid_fraction=0.01,
        expected_clamp_fraction=0.08,
    )
    job = stage._StrictRegionJob.model_construct()
    MaterializedRegionJob.apply_data_transformations(job, expected, var)
    job.apply_data_transformations(array, var)
    np.testing.assert_array_equal(
        array.values.view(np.uint32), expected.values.view(np.uint32)
    )
    assert job.transform_receipt == {
        "status": "succeeded",
        "invalid_negative_count": 0,
        "clamped_count": 0,
    }


def test_remote_staging_rejected_before_download(
    unit_factory: Callable[..., _Unit], downloads: Mock
) -> None:
    unit = unit_factory()
    with pytest.raises(AssertionError):
        stage.regenerate_unit(
            unit.template,
            unit.variable,
            1,
            unit.temporary,
            MemoryStore(),  # ty: ignore[invalid-argument-type]
            unit.requests,
        )
    downloads.assert_not_called()


def test_non_succeeded_generated_status_rejected(
    unit_factory: Callable[..., _Unit],
    downloads: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit = unit_factory()
    generate = stage._StrictRegionJob.generate_source_file_coords

    def failed(
        self: stage._StrictRegionJob, ds: xr.Dataset, variables: Sequence[EcmwfDataVar]
    ) -> Sequence[IfsEnsSourceFileCoord]:
        coords = list(generate(self, ds, variables))
        coords[-1] = replace(coords[-1], status=SourceFileStatus.ReadFailed)
        return coords

    monkeypatch.setattr(stage._StrictRegionJob, "generate_source_file_coords", failed)
    with pytest.raises(AssertionError):
        unit.run()
    downloads.assert_not_called()


def test_native_shared_memory_and_process_writes(
    unit_factory: Callable[..., _Unit],
    downloads: Mock,
    reads: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def executor(
        max_workers: int, initializer: Callable[[int], None], initargs: tuple[int]
    ) -> ProcessPoolExecutor:
        assert max_workers == 4
        return ProcessPoolExecutor(
            max_workers=2, initializer=initializer, initargs=initargs
        )

    monkeypatch.setattr(shared_memory_utils, "SharedMemory", SharedMemory)
    monkeypatch.setattr(stage, "ProcessPoolExecutor", executor)
    unit = unit_factory()
    receipt = unit.run()
    assert receipt["status"] == "succeeded"
    assert receipt["read_count"] == receipt["expected_count"] == 6
    assert receipt["metadata_unchanged"] is True
