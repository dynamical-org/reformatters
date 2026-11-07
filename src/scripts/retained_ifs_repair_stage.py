import hashlib
import json
import math
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import chdir
from itertools import product
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numba import set_num_threads
from pydantic import ConfigDict, Field, TypeAdapter
from zarr.storage import LocalStore

from reformatters.common.deaccumulation import (
    VALID_OUTPUT_UNITS_FOR_DEACCUMULATION,
    _deaccumulate_to_rates_numba,
)
from reformatters.common.download import byte_ranges_digest, http_download_to_disk
from reformatters.common.iterating import item
from reformatters.common.materialized_region_job import MaterializedRegionJob
from reformatters.common.pydantic import FrozenBaseModel, replace
from reformatters.common.region_job import SourceFileResult, SourceFileStatus
from reformatters.common.shared_memory_utils import (
    create_data_array_and_template,
    make_shared_buffer,
    write_shards,
)
from reformatters.common.time_utils import whole_hours
from reformatters.common.zarr import copy_data_var
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar
from reformatters.ecmwf.ecmwf_grib_index import grib_message_byte_ranges_from_index
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.region_job import (
    EcmwfIfsEnsForecast15Day025DegreeRegionJob,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.source_file_coord import (
    IfsEnsSourceFileCoord,
    MarsSourceFileCoord,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
)


class _Request(FrozenBaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    lead_hours: int
    member: int
    url: str
    index_path: str
    index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    starts: list[int]
    ends: list[int]

    @property
    def key(self) -> tuple[int, int]:
        return self.lead_hours, self.member


class _LocalShardStore(LocalStore):
    @property
    def supports_consolidated_metadata(self) -> bool:
        return False


class _StrictRegionJob(EcmwfIfsEnsForecast15Day025DegreeRegionJob):
    transform_receipt: dict[str, Any] = Field(
        default_factory=lambda: {"status": "not_run"}
    )

    def apply_data_transformations(
        self, data_array: xr.DataArray, data_var: EcmwfDataVar
    ) -> None:
        outcome = self.transform_receipt
        outcome["status"] = "started"
        if data_var.internal_attrs.scale_factor is not None:
            data_array *= data_var.internal_attrs.scale_factor

        if data_var.internal_attrs.deaccumulate_to_rate:
            _strict_deaccumulate(data_array, data_var, outcome)

        MaterializedRegionJob.apply_data_transformations(self, data_array, data_var)
        outcome["status"] = "succeeded"


def _strict_deaccumulate(
    data_array: xr.DataArray, data_var: EcmwfDataVar, outcome: dict[str, Any]
) -> None:
    outcome.update(invalid_negative_count=None, clamped_count=None)
    assert data_array.attrs["units"] in VALID_OUTPUT_UNITS_FOR_DEACCUMULATION
    reset = data_var.internal_attrs.window_reset_frequency
    threshold = data_var.internal_attrs.deaccumulation_invalid_below_threshold_rate
    assert reset is not None
    assert threshold is not None
    times = data_array["lead_time"].values
    assert np.issubdtype(times.dtype, np.timedelta64)
    seconds = times.astype("timedelta64[s]").astype(np.int64)
    assert seconds[0] == 0
    assert np.all(np.diff(seconds) > 0)
    axis = data_array.dims.index("lead_time")
    values = data_array.values.reshape(
        math.prod(data_array.shape[:axis]),
        data_array.shape[axis],
        math.prod(data_array.shape[axis + 1 :]),
    )
    invalid, clamped = _deaccumulate_to_rates_numba(
        values,
        seconds,
        seconds % reset.total_seconds() == 0,
        np.zeros(seconds.shape, dtype=np.bool),
        threshold,
        0,
    )
    outcome.update(invalid_negative_count=int(invalid), clamped_count=int(clamped))
    invalid_fraction = invalid / values.size
    if invalid_fraction > 0.01:
        raise ValueError(
            f"Found {invalid} values ({invalid_fraction:.1%}) below threshold, expected at most 1.0%"
        )
    clamped_fraction = clamped / values.size
    if clamped_fraction > 0.08:
        raise ValueError(
            f"Over 8% ({clamped} total, {clamped_fraction:.1%}) values were clamped to 0"
        )


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _protected_hashes(root: Path, template: xr.DataTree) -> dict[str, str]:
    paths = set(root.rglob("zarr.json"))
    for node in template.subtree:
        for coord in node.to_dataset().coords:
            directory = root / node.path.lstrip("/") / str(coord)
            paths.update(path for path in directory.rglob("*") if path.is_file())
    return {str(path.relative_to(root)): _hash(path) for path in sorted(paths)}


def _key(coord: IfsEnsSourceFileCoord) -> tuple[int, int]:
    return whole_hours(coord.lead_time), coord.ensemble_member


def _url(coord: IfsEnsSourceFileCoord) -> str:
    return (
        coord.get_url("s3-source-coop")
        if isinstance(coord, MarsSourceFileCoord)
        else coord.get_url("gcs")
    )


def _approve_requests(
    requests_path: Path,
    coords: Sequence[IfsEnsSourceFileCoord],
    work: Path,
) -> tuple[dict[tuple[int, int], _Request], str]:
    raw = requests_path.read_bytes()
    requests = TypeAdapter(list[_Request]).validate_json(raw)
    by_key = {request.key: request for request in requests}
    generated = [_key(coord) for coord in coords]
    assert len(by_key) == len(requests), "Duplicate approved request coordinates"
    assert len(set(generated)) == len(generated), "Duplicate generated coordinates"
    assert set(by_key) == set(generated), "Approved/generated roster mismatch"
    identities = {
        (request.url, tuple(request.starts), tuple(request.ends))
        for request in requests
    }
    assert len(identities) == len(requests), "Duplicate approved source ranges"

    sha_by_path: dict[Path, str] = {}
    snapshots: dict[tuple[Path, str], Path] = {}
    for coord in coords:
        request = by_key[_key(coord)]
        assert request.url == _url(coord), "Approved source URL mismatch"
        relative = Path(request.index_path)
        assert not relative.is_absolute(), "Index path must be relative"
        index_path = (requests_path.parent / relative).resolve()
        assert index_path.is_relative_to(requests_path.parent.resolve())
        if index_path in sha_by_path:
            assert sha_by_path[index_path] == request.index_sha256, (
                "Conflicting SHA256 pins for one index path"
            )
        cache_key = (index_path, request.index_sha256)
        if cache_key not in snapshots:
            index_bytes = index_path.read_bytes()
            assert hashlib.sha256(index_bytes).hexdigest() == request.index_sha256, (
                "Index SHA256 mismatch"
            )
            pinned = work / f"index-{request.index_sha256}.jsonl"
            if not pinned.exists():
                with pinned.open("xb") as stream:
                    stream.write(index_bytes)
                pinned.chmod(0o444)
            snapshots[cache_key] = pinned
            sha_by_path[index_path] = request.index_sha256
        starts, ends = grib_message_byte_ranges_from_index(
            snapshots[cache_key],
            coord.data_var_group,
            coord.ensemble_member,
            step=coord.index_step,
        )
        assert (request.starts, request.ends) == (starts, ends), (
            "Approved/index selector range mismatch"
        )
        assert starts
        assert all(0 <= start < end for start, end in zip(starts, ends, strict=True))
    return by_key, hashlib.sha256(raw).hexdigest()


def _unit_coords(tree: xr.DataTree, init_index: int) -> xr.Dataset:
    unit = tree.to_dataset().isel(init_time=[init_index])
    return xr.Dataset(coords=unit.coords).load().copy(deep=True)


def _check_result_coords(
    job: _StrictRegionJob,
    coords: Sequence[IfsEnsSourceFileCoord],
    baseline: xr.Dataset,
) -> list[str]:
    results = {
        job.data_vars[0].path: [
            SourceFileResult(
                status=coord.status, out_loc=dict(coord.out_loc()), url=_url(coord)
            )
            for coord in coords
        ]
    }
    updated = job.update_template_with_results(results)
    actual = xr.Dataset(coords=updated.to_dataset().coords).sel(
        init_time=baseline.init_time
    )
    xr.testing.assert_identical(actual, baseline)
    assert all(actual[name].dtype == baseline[name].dtype for name in baseline.coords)
    return sorted(str(name) for name in baseline.coords if name not in baseline.dims)


def _write_receipt(work: Path, receipt: dict[str, Any]) -> None:
    path = work / "receipt.json"
    with path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    path.chmod(0o444)


def _prepare_local_unit(
    template: xr.DataTree,
    variable: str,
    init_index: int,
    temporary: Path,
    staging: LocalStore,
) -> tuple[EcmwfDataVar, set[str], dict[str, Any], dict[str, str], dict[str, str]]:
    assert type(staging) is LocalStore
    assert not staging.read_only
    assert template.groups == ("/",), "IFS15 staging requires a root-only template"
    assert type(init_index) is int
    assert 0 <= init_index < template.sizes["init_time"]
    stage_root = staging.root.resolve()
    assert temporary.is_dir()
    assert stage_root.is_dir()
    assert not temporary.is_relative_to(stage_root)
    assert not stage_root.is_relative_to(temporary)
    config = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    data_var = item(var for var in config.data_vars if var.name == variable)
    assert variable in template.to_dataset().data_vars
    da = template[variable]
    assert da.dims == (
        "init_time",
        "lead_time",
        "ensemble_member",
        "latitude",
        "longitude",
    )
    shards = da.encoding["shards"]
    assert shards[0] == 1, "A whole unit must occupy its own init shard"
    assert da.dtype == np.float32
    tmp_before = _protected_hashes(temporary, template)
    stage_before = _protected_hashes(stage_root, template)
    assert tmp_before
    assert tmp_before == stage_before, "Local metadata/coordinate clones differ"
    metadata = json.loads((temporary / variable / "zarr.json").read_bytes())
    assert metadata["chunk_key_encoding"] == {
        "name": "default",
        "configuration": {"separator": "/"},
    }
    assert tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"]) == tuple(
        shards
    )
    expected_keys = {
        f"{variable}/c/" + "/".join(map(str, indexes))
        for indexes in product(
            [init_index],
            *(
                range(math.ceil(size / shard))
                for size, shard in zip(da.shape[1:], shards[1:], strict=True)
            ),
        )
    }
    assert not (temporary / variable / "c").exists(), (
        "Temporary variable contains stale data"
    )
    assert not any((stage_root / key).exists() for key in expected_keys), (
        "Staging unit contains stale data"
    )
    return data_var, expected_keys, metadata, tmp_before, stage_before


def _generated_coords(
    job: _StrictRegionJob, processing: xr.Dataset, data_var: EcmwfDataVar
) -> tuple[list[IfsEnsSourceFileCoord], list[dict[str, int]]]:
    leads = [whole_hours(pd.Timedelta(value)) for value in processing.lead_time.values]
    members = [int(value) for value in processing.ensemble_member.values]
    assert len(set(leads)) == len(leads)
    assert len(set(members)) == len(members)
    assert leads[0] == 0
    coords = list(job.generate_source_file_coords(processing, [data_var]))
    all_pairs = set(product(leads, members))
    omitted = (
        {(0, member) for member in members}
        if not data_var.has_hour_0_values()
        else set()
    )
    assert {_key(coord) for coord in coords} == all_pairs - omitted, (
        "Unexpected missing generated source coordinates"
    )
    assert coords
    assert all(coord.status == SourceFileStatus.Processing for coord in coords)
    assert all(
        coord.init_time == pd.Timestamp(processing.init_time.values[0])
        for coord in coords
    )
    return coords, [
        {"lead_hours": lead, "member": member} for lead, member in sorted(omitted)
    ]


def _read_sources(
    job: _StrictRegionJob,
    out: xr.DataArray,
    coords: Sequence[IfsEnsSourceFileCoord],
    requests: dict[tuple[int, int], _Request],
    work: Path,
    receipt: dict[str, Any],
) -> list[IfsEnsSourceFileCoord]:
    results = []
    for coord in coords:
        request = requests[_key(coord)]
        suffix = byte_ranges_digest(request.starts, request.ends)
        assert (
            suffix
            == hashlib.sha256(
                ",".join(
                    f"{start}-{end}"
                    for start, end in zip(request.starts, request.ends, strict=True)
                ).encode()
            ).hexdigest()
        )
        with chdir(work):
            downloaded = http_download_to_disk(
                request.url,
                job.dataset_id,
                byte_ranges=(request.starts, request.ends),
                local_path_suffix=f"-{suffix}",
                disk_cache=False,
            ).resolve()
        assert downloaded.is_relative_to(work)
        assert downloaded.is_file()
        downloaded.chmod(0o444)
        source = request.model_dump() | {
            "path": str(downloaded),
            "sha256": _hash(downloaded),
            "size": downloaded.stat().st_size,
            "read_status": "pending",
        }
        receipt["sources"].append(source)
        assert source["size"] == sum(
            end - start for start, end in zip(request.starts, request.ends, strict=True)
        ), "Downloaded range size mismatch"
        downloaded_coord = replace(coord, downloaded_path=downloaded)
        values = job.read_data(downloaded_coord, job.data_vars[0])
        assert values.dtype == np.float32, "Decoded source dtype mismatch"
        assert values.shape == (out.sizes["latitude"], out.sizes["longitude"]), (
            "Decoded source shape mismatch"
        )
        out.loc[coord.out_loc()] = values
        assert _hash(downloaded) == source["sha256"], "Source changed during read"
        results.append(replace(downloaded_coord, status=SourceFileStatus.Succeeded))
        source["read_status"] = "succeeded"
        receipt["read_count"] += 1
    return results


def regenerate_unit(
    template: xr.DataTree,
    variable: str,
    init_index: int,
    temporary: Path,
    staging: LocalStore,
    requests_path: Path,
) -> dict[str, Any]:
    """Regenerate one whole IFS15 unit locally, requiring a pinned complete source roster."""
    if sys.flags.optimize:
        raise RuntimeError("Local staging requires Python assertions enabled")
    temporary = temporary.resolve()
    data_var, expected_keys, metadata, tmp_before, stage_before = _prepare_local_unit(
        template, variable, init_index, temporary, staging
    )
    stage_root = staging.root.resolve()
    unit_coords = _unit_coords(template, init_index)
    work = Path(mkdtemp(prefix="retained-ifs-stage-", dir=temporary.parent)).resolve()
    receipt: dict[str, Any] = {
        "status": "started",
        "variable": variable,
        "init_index": init_index,
        "init_time": str(unit_coords.init_time.values[0]),
        "receipt_path": str(work / "receipt.json"),
        "expected_count": 0,
        "read_count": 0,
        "sources": [],
        "transform": {},
        "derived_coords_checked": [],
    }
    job = _StrictRegionJob(
        tmp_store=temporary,
        template_ds=template.copy(deep=False),
        data_vars=[data_var],
        append_dim="init_time",
        region=slice(init_index, init_index + 1),
        reformat_job_name="local-staging",
        download_parallelism=1,
        read_parallelism=1,
    )
    receipt["transform"] = job.transform_receipt
    try:
        processing, output = job._get_region_datasets()  # noqa: SLF001
        coords, absent = _generated_coords(job, processing, data_var)
        receipt["expected_count"] = len(coords)
        receipt["intentional_absent_coords"] = absent
        requests, requests_sha = _approve_requests(
            requests_path.resolve(), coords, work
        )
        receipt["requests_sha256"] = requests_sha
        set_num_threads(1)
        with make_shared_buffer(processing) as shared_buffer:
            out, out_template = create_data_array_and_template(
                processing,
                variable,
                shared_buffer,
                fill_value=float(metadata["fill_value"]),
            )
            coords = _read_sources(job, out, coords, requests, work, receipt)
            assert receipt["read_count"] == receipt["expected_count"]
            assert all(coord.status == SourceFileStatus.Succeeded for coord in coords)
            job.transform_receipt["status"] = "started"
            job.apply_data_transformations(out, data_var)
            receipt["derived_coords_checked"] = _check_result_coords(
                job, coords, unit_coords
            )
            with ProcessPoolExecutor(
                max_workers=4, initializer=set_num_threads, initargs=(1,)
            ) as executor:
                write_shards(
                    out_template,
                    shared_buffer,
                    "init_time",
                    output,
                    _LocalShardStore(temporary),
                    executor,
                )
        written = {
            str(path.relative_to(temporary))
            for path in (temporary / variable / "c").rglob("*")
            if path.is_file()
        }
        assert written == expected_keys, "Written shard keys differ from B grid"
        assert _protected_hashes(temporary, template) == tmp_before, (
            "Temporary metadata/coordinates changed"
        )
        copy_data_var(
            variable,
            job.region,
            output,
            "init_time",
            temporary,
            staging,
            replica_stores=[],
        )
        assert all((stage_root / key).is_file() for key in expected_keys)
        assert _protected_hashes(stage_root, template) == stage_before, (
            "Staging metadata/coordinates changed"
        )
        receipt.update(
            status="succeeded",
            metadata_unchanged=True,
            written_keys=sorted(expected_keys),
        )
    except Exception as error:
        receipt.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            metadata_unchanged=(
                _protected_hashes(temporary, template) == tmp_before
                and _protected_hashes(stage_root, template) == stage_before
            ),
        )
        if job.transform_receipt.get("status") == "started":
            job.transform_receipt.update(status="failed", error=str(error))
        _write_receipt(work, receipt)
        error.add_note(f"Local staging receipt: {receipt['receipt_path']}")
        raise
    _write_receipt(work, receipt)
    return receipt
