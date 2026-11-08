import argparse
import asyncio
import hashlib
import json
import platform
import re
import sys
import tomllib
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from importlib.metadata import version
from itertools import islice, product
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import httpx
import icechunk
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import zarr
from numba import set_num_threads
from numcodecs import blosc
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zarr.abc.store import Store
from zarr.storage import LocalStore

import reformatters
import scripts
from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.common.logging import get_logger
from reformatters.common.materialized_region_job import MaterializedRegionJob
from reformatters.common.pydantic import replace
from reformatters.common.zarr import sync_to_store
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.region_job import (
    EcmwfIfsEnsForecast15Day025DegreeRegionJob as ProductionJob,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.source_file_coord import (
    MarsSourceFileCoord,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeTemplateConfig as ProductionConfig,
)
from scripts import retained_ifs_repair as helpers
from scripts.icechunk_utils import open_repo

log = get_logger(__name__)
require = helpers.require
BASE_SNAPSHOT = "VZSXDB55KG2P774ZHT8G"
PROTECTED_BRANCH = "ifs-ens-full-20261102T215837Z-187dd836"
REPOSITORY = "s3://dynamical-ecmwf-ifs-ens/ecmwf-ifs-ens-forecast-15-day-0-25-degree/v0.1.0.icechunk"
SCIENCE_SHA256 = "12ef8c70444f6146c4d3f312568407f980f7afaeb544d07c78e21b2ae84792f3"
SCOPE = {
    ("precipitation_surface", "2023-11-05T00:00:00"): (
        (45, 111, "a116e297d2295874b725946b08446265b116adf51a41d4b5ed65258a6f08e45b"),
        (45, 114, "7a89d4bd372cfe12b80ead10b199623e6366cc0609f1e8e8cc9f87ac0b594d74"),
    ),
    ("pressure_reduced_to_mean_sea_level", "2023-10-08T00:00:00"): (
        (30, 48, "bbbcfc689e8767f857078bc670ef63f6f0aee6dae2bba946b3595768ad7ee52d"),
    ),
    ("wind_v_10m", "2023-12-30T00:00:00"): (
        (11, 3, "8f408ec60d323221c8f7d6b6a253d3d79bcdd4afc05acef3d5ac4cddf4623c7d"),
    ),
    ("pressure_surface", "2024-01-17T00:00:00"): (
        (25, 90, "5f7bf886234e2d39c3b6b77ea7681b26ddcbc8495d4c4def63a2daa674b4f3db"),
    ),
    ("temperature_2m", "2024-02-03T00:00:00"): (
        (35, 90, "c7a7a3e2625df2350cd03db52a315dc1d545844304b0dccf919579d17142a183"),
    ),
    ("categorical_precipitation_type_surface", "2025-10-13T00:00:00"): (
        (3, 258, "773ea6ded9e4c9f2f50ab14a2f2ec08ee672ba083c64f992c47dfbe102415829"),
    ),
}
PARAMETERS = {
    "precipitation_surface": 228,
    "pressure_reduced_to_mean_sea_level": 151,
    "wind_v_10m": 166,
    "pressure_surface": 134,
    "temperature_2m": 167,
    "categorical_precipitation_type_surface": 19,
}
LEADS = np.r_[np.arange(0, 145, 3), np.arange(150, 361, 6)]
MEMBERS = np.arange(51)
LATITUDE = np.linspace(90, -90, 721)
LONGITUDE = np.arange(1440) / 4 - 180

type Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
type Array = np.ndarray[Any, np.dtype[np.float32]]
type Fields = dict[tuple[int, int], Array]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ExpectedField(Model):
    member: int
    lead_hours: int
    expected_sha256: Sha256


class Source(Model):
    member: int
    lead_hours: int
    url: str
    start: int = Field(ge=0)
    end_exclusive: int = Field(gt=0)
    sha256: Sha256
    header: dict[str, int | str]

    @model_validator(mode="after")
    def check_range(self) -> Self:
        require(
            0 < self.end_exclusive - self.start <= 16 * 1024 * 1024,
            "Invalid source range size",
        )
        return self


class Unit(Model):
    variable: str
    init: str
    fields: tuple[ExpectedField, ...]
    sources: tuple[Source, ...]


class ShardHashes(Model):
    encoded_sha256: Sha256
    decoded_sha256: Sha256


class Manifest(Model):
    version: Literal[2]
    dataset_id: Literal["ecmwf-ifs-ens-forecast-15-day-0-25-degree"]
    repository: str
    base_snapshot: str
    protected_branch: str
    candidate_branch: str
    runtime_sha256: Sha256
    metadata_sha256: Sha256
    scientific_manifest_sha256: Sha256
    units: tuple[Unit, ...]
    approved_shards: dict[str, ShardHashes] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_scope(self) -> Self:
        require(self.repository == REPOSITORY, "Repository is outside immutable scope")
        require(
            self.base_snapshot == BASE_SNAPSHOT,
            "Base snapshot is outside immutable scope",
        )
        require(
            self.protected_branch == PROTECTED_BRANCH,
            "Protected branch is outside immutable scope",
        )
        require(
            self.scientific_manifest_sha256 == SCIENCE_SHA256,
            "Scientific manifest pin mismatch",
        )
        require(
            bool(
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.candidate_branch)
            ),
            "Invalid candidate branch",
        )
        require(
            self.candidate_branch not in {"main", PROTECTED_BRANCH},
            "Forbidden candidate branch",
        )
        require(len(self.units) == len(SCOPE), "Exact six-unit scope required")
        require(
            {(u.variable, u.init) for u in self.units} == set(SCOPE),
            "Unit scope mismatch",
        )
        for unit in self.units:
            expected = SCOPE[unit.variable, unit.init]
            require(
                len(unit.fields) == len(expected), "Exact permitted fields required"
            )
            require(
                {(f.member, f.lead_hours, f.expected_sha256) for f in unit.fields}
                == set(expected),
                "Frozen expected-field scope/hash mismatch",
            )
            source_pairs = {(m, h) for m, h, _ in expected}
            if unit.variable == "precipitation_surface":
                source_pairs.add((45, 108))
            require(
                len(unit.sources) == len(source_pairs),
                "Exact own-source roster required",
            )
            require(
                {(s.member, s.lead_hours) for s in unit.sources} == source_pairs,
                "Source scope mismatch",
            )
            for source in unit.sources:
                require(source.url == source_url(unit, source), "Source URL mismatch")
                require(
                    source.header == expected_header(unit, source),
                    "Header identity pin mismatch",
                )
        require(
            all(
                re.fullmatch(r"[a-z][a-z0-9_]*/c/[0-9]+/0/0/[0-9]+/[0-9]+", key)
                for key in self.approved_shards
            ),
            "Invalid approved shard key",
        )
        return self


def source_url(unit: Unit, source: Source) -> str:
    if unit.variable == "categorical_precipitation_type_surface":
        return "https://storage.googleapis.com/ecmwf-open-data/20251013/00z/ifs/0p25/enfo/20251013000000-258h-enfo-ef.grib2"
    request_type = "pf_sfc_0" if source.member <= 25 else "pf_sfc_1"
    return f"https://data.source.coop/dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens/{unit.init[:10]}/{request_type}.grib"


def expected_header(unit: Unit, source: Source) -> dict[str, int | str]:
    ptype = unit.variable == "categorical_precipitation_type_surface"
    common: dict[str, int | str] = {
        "edition": 2 if ptype else 1,
        "centre": 98,
        "parameter": PARAMETERS[unit.variable],
        "level_type": 1,
        "member": source.member,
        "lead_hours": source.lead_hours,
        "init": unit.init,
        "ensemble_size": 51,
        "time_unit": 1,
    }
    return common | (
        {
            "discipline": 0,
            "category": 1,
            "product_definition_template": 1,
            "ensemble_type": 255,
        }
        if ptype
        else {
            "table": 128,
            "local_definition": 30,
            "mars_type": 11,
            "time_range_indicator": 0,
        }
    )


def parse_header(body: bytes) -> dict[str, int | str]:
    require(
        len(body) >= 61 and body[:4] == b"GRIB" and body[-4:] == b"7777",
        "Invalid GRIB message framing",
    )
    if body[7] == 1:
        require(int.from_bytes(body[4:7], "big") == len(body), "GRIB1 length mismatch")
        section_size = int.from_bytes(body[8:11], "big")
        require(
            51 <= section_size <= len(body) - 12, "Invalid GRIB1 product section size"
        )
        indicator = body[28]
        require(indicator in (0, 10), "Unsupported GRIB1 time range indicator")
        require(indicator != 0 or body[27] == 0, "Unexpected GRIB1 P2")
        return {
            "edition": 1,
            "centre": body[12],
            "table": body[11],
            "parameter": body[16],
            "level_type": body[17],
            "local_definition": body[48],
            "mars_type": body[50],
            "member": body[57],
            "ensemble_size": body[58],
            "time_unit": body[25],
            "time_range_indicator": indicator,
            "lead_hours": body[26]
            if indicator == 0
            else int.from_bytes(body[26:28], "big"),
            "init": datetime(
                (body[32] - 1) * 100 + body[20], body[21], body[22], body[23], body[24]
            ).isoformat(),
        }
    require(
        body[7] == 2 and int.from_bytes(body[8:16], "big") == len(body),
        "GRIB2 edition/length mismatch",
    )
    sections = {}
    position = 16
    while position < len(body) - 4:
        require(position + 5 <= len(body) - 4, "Truncated GRIB2 section header")
        length = int.from_bytes(body[position : position + 4], "big")
        number = body[position + 4]
        require(
            length >= 5 and position + length <= len(body) - 4,
            "Invalid GRIB2 section size",
        )
        require(number not in sections, "Multiple GRIB2 fields/duplicate sections")
        sections[number] = body[position : position + length]
        position += length
    require(
        position == len(body) - 4 and {1, 3, 4, 5, 6, 7} <= set(sections),
        "Incomplete GRIB2 message",
    )
    reference, field = sections[1], sections[4]
    require(
        len(reference) >= 21 and len(field) == 37,
        "Unexpected GRIB2 identification/product section",
    )
    return {
        "edition": 2,
        "centre": int.from_bytes(reference[5:7], "big"),
        "discipline": body[6],
        "category": field[9],
        "parameter": field[10],
        "level_type": field[22],
        "member": field[35],
        "ensemble_size": field[36],
        "ensemble_type": field[34],
        "product_definition_template": int.from_bytes(field[7:9], "big"),
        "time_unit": field[17],
        "lead_hours": int.from_bytes(field[18:22], "big"),
        "init": datetime(
            int.from_bytes(reference[12:14], "big"),
            reference[14],
            reference[15],
            reference[16],
            reference[17],
            reference[18],
        ).isoformat(),
    }


def raw_digest(values: Array) -> str:
    require(values.dtype == np.dtype("<f4"), "Expected little-endian float32")
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def fetch_source(source: Source, path: Path) -> dict[str, Any]:
    size = source.end_exclusive - source.start
    with (
        httpx.Client(trust_env=False, follow_redirects=False, timeout=90) as client,
        client.stream(
            "GET",
            source.url,
            headers={
                "Range": f"bytes={source.start}-{source.end_exclusive - 1}",
                "Accept-Encoding": "identity",
            },
        ) as response,
    ):
        require(response.status_code == 206, "Source did not return exact 206 range")
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", content_range)
        require(match is not None, "Invalid Content-Range")
        assert match is not None
        require(
            tuple(map(int, match.groups()[:2]))
            == (source.start, source.end_exclusive - 1)
            and int(match[3]) >= source.end_exclusive,
            "Content-Range mismatch",
        )
        require(
            response.headers.get("Content-Encoding", "identity") == "identity",
            "Unexpected source content encoding",
        )
        require(
            int(response.headers.get("Content-Length", size)) == size,
            "Source Content-Length mismatch",
        )
        body = bytearray()
        for chunk in response.iter_raw():
            require(len(body) + len(chunk) <= size, "Source range too long")
            body.extend(chunk)
    require(len(body) == size, "Source range size mismatch")
    require(
        hashlib.sha256(body).hexdigest() == source.sha256, "Source raw SHA256 mismatch"
    )
    require(parse_header(bytes(body)) == source.header, "Source header mismatch")
    with path.open("xb") as stream:
        stream.write(body)
    return source.model_dump(mode="json") | {
        "http_status": 206,
        "content_range": content_range,
        "size": size,
    }


def reconstruct_fields(unit: Unit, work: Path) -> tuple[Fields, list[dict[str, Any]]]:
    work.mkdir()
    var = next(v for v in ProductionConfig().data_vars if v.name == unit.variable)
    ordered = sorted(
        unit.sources, key=lambda source: (source.lead_hours, source.member)
    )
    ds = xr.Dataset(
        coords={
            "init_time": [pd.Timestamp(unit.init)],
            "lead_time": pd.to_timedelta([s.lead_hours for s in ordered], unit="h"),
            "ensemble_member": [ordered[0].member],
        },
        attrs={"dataset_id": helpers.DATASET_ID},
    )
    job = ProductionJob(
        tmp_store=work,
        template_ds=xr.DataTree(ds),
        data_vars=[var],
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="retained-field-patch",
        download_parallelism=1,
        read_parallelism=1,
    )
    coords = {
        (c.ensemble_member, int(c.lead_time / pd.Timedelta(hours=1))): c
        for c in job.generate_source_file_coords(ds, [var])
    }
    arrays, receipts = [], []
    for number, source in enumerate(ordered):
        path = work / f"source-{number}.grib"
        receipt = fetch_source(source, path)
        coord = coords[source.member, source.lead_hours]
        actual_url = (
            coord.get_url("s3-source-coop")
            if isinstance(coord, MarsSourceFileCoord)
            else coord.get_url("gcs")
        )
        require(source.url == actual_url, "Production source coordinate URL mismatch")
        with rasterio.open(path) as reader:
            tags, transform = reader.tags(1), reader.transform
            require(
                reader.shape == (len(LATITUDE), len(LONGITUDE)),
                "Source grid shape mismatch",
            )
            require(
                transform.b == transform.d == 0
                and np.array_equal(
                    transform.f + (np.arange(reader.height) + 0.5) * transform.e,
                    LATITUDE,
                )
                and np.array_equal(
                    transform.c + (np.arange(reader.width) + 0.5) * transform.a,
                    LONGITUDE,
                ),
                "Source grid coordinates mismatch",
            )
            require(
                int(tags["GRIB_REF_TIME"])
                == int(pd.Timestamp(unit.init, tz="UTC").timestamp())
                and int(tags["GRIB_FORECAST_SECONDS"]) == source.lead_hours * 3600,
                "Rasterio time identity mismatch",
            )
        values = job.read_data(replace(coord, downloaded_path=path), var)
        require(
            values.dtype == np.dtype("<f4")
            and values.shape == (len(LATITUDE), len(LONGITUDE)),
            "Production read_data shape/dtype mismatch",
        )
        require(
            helpers.file_digest(path) == source.sha256, "Source changed during decode"
        )
        arrays.append(values)
        receipts.append(receipt)
        path.unlink()
        log.info(
            f"status=source-decoded variable={unit.variable} member={source.member} lead_hours={source.lead_hours}"
        )
    data = xr.DataArray(
        np.stack(arrays)[None, :, None],
        dims=helpers.DIMS,
        coords=ds.coords,
        attrs={"units": var.attrs.units},
    )
    if var.internal_attrs.deaccumulate_to_rate:
        require(
            unit.variable == "precipitation_surface", "Unexpected accumulated quantity"
        )
        if var.internal_attrs.scale_factor is not None:
            data *= var.internal_attrs.scale_factor
        reset = var.internal_attrs.window_reset_frequency
        threshold = var.internal_attrs.deaccumulation_invalid_below_threshold_rate
        require(
            reset is not None and threshold is not None,
            "Missing deaccumulation configuration",
        )
        assert reset is not None
        assert threshold is not None
        deaccumulate_to_rates_inplace(
            data,
            dim="lead_time",
            reset_frequency=reset,
            invalid_below_threshold_rate=threshold,
            expected_invalid_fraction=0.01,
            expected_clamp_fraction=0.08,
        )
        MaterializedRegionJob.apply_data_transformations(job, data, var)
    else:
        job.apply_data_transformations(data, var)
    fields = {}
    for field in unit.fields:
        position = next(
            i
            for i, source in enumerate(ordered)
            if (source.member, source.lead_hours) == (field.member, field.lead_hours)
        )
        expected = np.ascontiguousarray(data.values[0, position, 0])
        require(
            raw_digest(expected) == field.expected_sha256,
            "Transformed expected raw float32 SHA256 mismatch",
        )
        fields[field.member, field.lead_hours] = expected
    return fields, receipts


def verify_runtime(manifest: Manifest, code_root: Path) -> None:
    require(
        sys.flags.optimize == 0 and sys.byteorder == "little",
        "Unapproved optimized/big-endian runtime",
    )
    require(
        helpers.runtime_digest(code_root) == manifest.runtime_sha256,
        "Runtime source/lock hash mismatch",
    )
    for module in (reformatters, scripts):
        require(
            module.__file__ is not None
            and Path(module.__file__)
            .resolve()
            .is_relative_to((code_root / "src").resolve()),
            "Imported runtime does not match source tree",
        )
    locked = tomllib.loads((code_root / "uv.lock").read_text())
    for package in locked["package"]:
        if package["name"] in {
            "numpy",
            "pandas",
            "xarray",
            "zarr",
            "icechunk",
            "rasterio",
            "numba",
            "numcodecs",
            "rioxarray",
            "affine",
            "httpx",
            "pydantic",
        }:
            require(
                version(package["name"]) == package["version"],
                f"Installed dependency differs from lock: {package['name']}",
            )
    set_num_threads(1)
    blosc.set_nthreads(1)
    zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})


@contextmanager
def new_work_directory(work: Path) -> Generator[None]:
    work.mkdir(mode=0o700, exist_ok=False)
    log.info("status=started")
    try:
        yield
    except BaseException as error:
        helpers.write_record(
            work / "STOP.json",
            {
                "status": "stopped",
                "error_type": type(error).__name__,
                "message": str(error),
                "automatic_retry_allowed": False,
                "base_snapshot": BASE_SNAPSHOT,
                "protected_branch": PROTECTED_BRANCH,
            },
        )
        log.exception("status=stopped automatic_retry_allowed=false")
        raise


def validate_refs(repo: icechunk.Repository, manifest: Manifest) -> None:
    require(
        repo.lookup_branch(PROTECTED_BRANCH) == BASE_SNAPSHOT, "Protected branch moved"
    )
    require(
        manifest.candidate_branch not in repo.list_branches(),
        "Candidate branch already exists; investigate without retry",
    )


def resolve_units(manifest: Manifest, tree: xr.DataTree) -> list[tuple[Unit, int]]:
    require(
        tree.groups == ("/",) and tree.attrs.get("dataset_id") == helpers.DATASET_ID,
        "Dataset structure/identity mismatch",
    )
    require(
        np.array_equal(tree.lead_time.values, LEADS.astype("timedelta64[h]"))
        and np.array_equal(tree.ensemble_member.values, MEMBERS),
        "Lead/member coordinates mismatch",
    )
    require(
        np.array_equal(tree.latitude.values, LATITUDE)
        and np.array_equal(tree.longitude.values, LONGITUDE),
        "Retained spatial coordinates mismatch",
    )
    resolved = []
    for unit in manifest.units:
        indexes = np.flatnonzero(
            tree.init_time.values == pd.Timestamp(unit.init).to_datetime64()
        )
        require(len(indexes) == 1, "Init must resolve exactly once")
        array = tree[unit.variable]
        require(
            tuple(array.dims) == helpers.DIMS and array.dtype == np.dtype("<f4"),
            "Variable dimension/dtype mismatch",
        )
        require(
            array.shape[1:]
            == (len(LEADS), len(MEMBERS), len(LATITUDE), len(LONGITUDE)),
            "Unit shape mismatch",
        )
        resolved.append((unit, int(indexes[0])))
    return resolved


def shard_selections(
    store: Store, unit: Unit, index: int
) -> Iterator[tuple[str, tuple[int | slice, ...]]]:
    metadata = json.loads(helpers.read_bytes(store, f"{unit.variable}/zarr.json"))
    require(
        metadata["chunk_key_encoding"]
        == {"name": "default", "configuration": {"separator": "/"}},
        "Unsupported chunk key encoding",
    )
    shape = metadata["shape"]
    shards = metadata["chunk_grid"]["configuration"]["chunk_shape"]
    require(
        len(shards) == 5
        and shards[0] == 1
        and shards[1] >= shape[1]
        and shards[2] >= shape[2],
        "Unsupported shard layout",
    )
    for y, x in product(
        range((shape[3] + shards[3] - 1) // shards[3]),
        range((shape[4] + shards[4] - 1) // shards[4]),
    ):
        yield (
            f"{unit.variable}/c/{index}/0/0/{y}/{x}",
            (
                index,
                slice(None),
                slice(None),
                slice(y * shards[3], min((y + 1) * shards[3], shape[3])),
                slice(x * shards[4], min((x + 1) * shards[4], shape[4])),
            ),
        )


def compare_shard(
    left: Array, right: Array, fields: Fields, selection: tuple[int | slice, ...]
) -> dict[str, int]:
    require(
        left.shape == right.shape and left.dtype == right.dtype == np.dtype("<f4"),
        "Shard shape/dtype mismatch",
    )
    y, x = selection[-2:]
    total = changed = outside = 0
    for li, lead in enumerate(LEADS):
        for mi, member in enumerate(MEMBERS):
            original, candidate = left[li, mi], right[li, mi]
            differences = int(
                np.count_nonzero(original.view("<u4") != candidate.view("<u4"))
            )
            expected = fields.get((int(member), int(lead)))
            if expected is None:
                require(
                    differences == 0,
                    "Unexpected uint32 change outside permitted fields",
                )
                outside += original.size
            else:
                require(
                    np.array_equal(candidate.view("<u4"), expected[y, x].view("<u4")),
                    "Expected field uint32 mismatch",
                )
            total += original.size
            changed += differences
    require(total == left.size, "Incomplete shard comparison")
    return {"compared_cells": total, "changed_cells": changed, "outside_cells": outside}


def add_counts(total: dict[str, int], counts: Mapping[str, int]) -> None:
    for key, value in counts.items():
        total[key] = total.get(key, 0) + value


def stage_unit(
    base: Store, stage: Store, work: Path, unit: Unit, index: int, fields: Fields
) -> tuple[dict[str, ShardHashes], dict[str, int], set[str]]:
    original = zarr.open_array(base, path=unit.variable, mode="r")
    staged = zarr.open_array(stage, path=unit.variable, mode="r+")
    changed, totals, all_keys = {}, {}, set()
    for key, selection in shard_selections(base, unit, index):
        all_keys.add(key)
        left = np.asarray(original[selection])
        patched = left.copy()
        for (member, lead), expected in fields.items():
            li, mi = (
                int(np.flatnonzero(LEADS == lead)[0]),
                int(np.flatnonzero(MEMBERS == member)[0]),
            )
            patched[li, mi] = expected[selection[-2], selection[-1]]
        decoded_sha = raw_digest(patched)
        staged[selection] = patched
        del patched
        decoded = np.asarray(staged[selection])
        require(
            raw_digest(decoded) == decoded_sha, "Codec roundtrip changed uint32 cells"
        )
        counts = compare_shard(left, decoded, fields, selection)
        add_counts(totals, counts)
        if counts["changed_cells"]:
            path = work / "stage" / key
            require(
                path.is_file() and not path.is_symlink(), "Missing/unsafe staged shard"
            )
            changed[key] = ShardHashes(
                encoded_sha256=helpers.file_digest(path), decoded_sha256=decoded_sha
            )
            path.chmod(0o444)
        log.info(
            f"status=shard-verified key={key} changed_cells={counts['changed_cells']}"
        )
        del left, decoded
    require(
        totals["compared_cells"] == int(np.prod(original.shape[1:])),
        "Incomplete full-unit comparison",
    )
    require(
        totals["compared_cells"] - totals["outside_cells"]
        == len(fields) * len(LATITUDE) * len(LONGITUDE),
        "Incomplete permitted mask",
    )
    require(totals["changed_cells"] > 0, "Unit has no correction")
    return changed, totals, all_keys


def verify_sealed(work: Path, shards: Mapping[str, ShardHashes]) -> None:
    for key, hashes in shards.items():
        path = work / "stage" / key
        require(
            path.is_file()
            and not path.is_symlink()
            and helpers.file_digest(path) == hashes.encoded_sha256,
            "Sealed staged shard changed",
        )


def post_compare(
    base: Store,
    committed: Store,
    resolved: list[tuple[Unit, int, Fields]],
    approved: Mapping[str, ShardHashes],
) -> list[dict[str, int]]:
    totals = []
    for unit, index, fields in resolved:
        before = zarr.open_array(base, path=unit.variable, mode="r")
        after = zarr.open_array(committed, path=unit.variable, mode="r")
        counts = {}
        for key, selection in shard_selections(base, unit, index):
            left, right = np.asarray(before[selection]), np.asarray(after[selection])
            add_counts(counts, compare_shard(left, right, fields, selection))
            if key in approved:
                require(
                    raw_digest(right) == approved[key].decoded_sha256,
                    "Committed decoded shard SHA256 mismatch",
                )
                require(
                    hashlib.sha256(helpers.read_bytes(committed, key)).hexdigest()
                    == approved[key].encoded_sha256,
                    "Committed encoded shard SHA256 mismatch",
                )
            del left, right
        require(
            counts["compared_cells"] == int(np.prod(before.shape[1:])),
            "Incomplete postcommit full-unit comparison",
        )
        totals.append(counts)
    return totals


def run_in_work_directory(  # noqa: PLR0915
    repo: icechunk.Repository, manifest: Manifest, work: Path, *, commit: bool
) -> dict[str, Any]:
    validate_refs(repo, manifest)
    require(
        not commit or bool(manifest.approved_shards),
        "Commit requires preapproved C1 shard keys and encoded/decoded hashes",
    )
    base = repo.readonly_session(snapshot_id=BASE_SNAPSHOT).store
    tree = helpers.open_tree(base)
    metadata = helpers.metadata_bytes(base, tree)
    require(
        helpers.metadata_digest(metadata) == manifest.metadata_sha256,
        "Baseline metadata/coordinate hash mismatch",
    )
    units = resolve_units(manifest, tree)
    helpers.clone_metadata(metadata, work / "stage")
    stage = LocalStore(work / "stage")
    changed: dict[str, ShardHashes] = {}
    records, resolved, all_keys = [], [], set(metadata)
    for number, (unit, index) in enumerate(units):
        fields, sources = reconstruct_fields(unit, work / f"sources-{number}")
        for field in unit.fields:
            require(
                raw_digest(fields[field.member, field.lead_hours])
                == field.expected_sha256,
                "Expected field hash mismatch before staging",
            )
        shards, comparison, unit_keys = stage_unit(
            base, stage, work, unit, index, fields
        )
        changed.update(shards)
        all_keys.update(unit_keys)
        resolved.append((unit, index, fields))
        records.append(
            {
                "variable": unit.variable,
                "init": unit.init,
                "init_index": index,
                "sources": sources,
                "comparison": comparison,
            }
        )
    require(
        set(asyncio.run(helpers.keys(stage))) == all_keys,
        "Unexpected/missing staged store keys",
    )
    require(
        helpers.metadata_bytes(stage, helpers.open_tree(stage)) == metadata,
        "Staged metadata/coordinates changed",
    )
    if manifest.approved_shards:
        require(
            changed == manifest.approved_shards,
            "C1 preapproved shard keys or encoded/decoded hashes mismatch",
        )
    verify_sealed(work, changed)
    result: dict[str, Any] = {
        "status": "prepared",
        "base_snapshot": BASE_SNAPSHOT,
        "candidate_branch": manifest.candidate_branch,
        "manifest_sha256": helpers.object_digest(manifest.model_dump(mode="json")),
        "runtime_sha256": manifest.runtime_sha256,
        "metadata_sha256": manifest.metadata_sha256,
        "scientific_manifest_sha256": manifest.scientific_manifest_sha256,
        "machine": platform.machine(),
        "changed_shards": {
            key: hashes.model_dump() for key, hashes in sorted(changed.items())
        },
        "units": records,
    }
    helpers.write_record(work / "precommit.json", result)
    log.info(f"status=prepared changed_shards={len(changed)}")
    validate_refs(repo, manifest)
    if commit:
        verify_sealed(work, changed)
        require(
            helpers.metadata_bytes(stage, helpers.open_tree(stage)) == metadata,
            "Sealed metadata/coordinates changed",
        )
        repo.create_branch(manifest.candidate_branch, BASE_SNAPSHOT)
        session = repo.writable_session(manifest.candidate_branch)
        require(
            session.snapshot_id == BASE_SNAPSHOT
            and repo.lookup_branch(manifest.candidate_branch) == BASE_SNAPSHOT,
            "Candidate base changed",
        )
        for key, hashes in sorted(changed.items()):
            value = helpers.read_bytes(stage, key)
            require(
                hashlib.sha256(value).hexdigest() == hashes.encoded_sha256,
                "Sealed staged shard changed before copy",
            )
            sync_to_store(session.store, key, value)
        helpers.validate_diff(session.status(), set(changed))
        require(
            repo.lookup_branch(PROTECTED_BRANCH) == BASE_SNAPSHOT,
            "Protected branch moved before commit",
        )
        snapshot = session.commit(
            f"Retained patch manifest={result['manifest_sha256']} runtime={manifest.runtime_sha256} precommit={helpers.object_digest(result)}"
        )
        helpers.write_record(
            work / "committed.json",
            {"snapshot": snapshot, "candidate_branch": manifest.candidate_branch},
        )
        ancestry = list(islice(repo.ancestry(snapshot_id=snapshot), 2))
        require(
            len(ancestry) == 2
            and ancestry[0].id == snapshot
            and ancestry[1].id == BASE_SNAPSHOT,
            "Unexpected single-commit ancestry",
        )
        helpers.validate_diff(
            repo.diff(from_snapshot_id=BASE_SNAPSHOT, to_snapshot_id=snapshot),
            set(changed),
        )
        committed = repo.readonly_session(snapshot_id=snapshot).store
        require(
            helpers.metadata_bytes(committed, helpers.open_tree(committed)) == metadata,
            "Committed metadata/coordinates changed",
        )
        result["post_comparisons"] = post_compare(base, committed, resolved, changed)
        require(
            repo.lookup_branch(PROTECTED_BRANCH) == BASE_SNAPSHOT,
            "Protected branch moved after commit",
        )
        require(
            repo.lookup_branch(manifest.candidate_branch) == snapshot,
            "Candidate moved after commit",
        )
        result.update(status="committed-and-verified", snapshot=snapshot)
    helpers.write_record(work / "result.json", result)
    log.info(f"status={result['status']}")
    tree.close()
    return result


def run(
    repo: icechunk.Repository, manifest: Manifest, work: Path, *, commit: bool = False
) -> dict[str, Any]:
    with new_work_directory(work):
        manifest = Manifest.model_validate_json(manifest.model_dump_json())
        verify_runtime(manifest, Path(__file__).resolve().parents[2])
        return run_in_work_directory(repo, manifest, work, commit=commit)


def check_sources(manifest: Manifest, work: Path) -> dict[str, Any]:
    source_count = field_count = 0
    records = []
    for number, unit in enumerate(manifest.units):
        fields, sources = reconstruct_fields(unit, work / f"sources-{number}")
        source_count += len(sources)
        field_count += len(fields)
        records.append(
            {
                "variable": unit.variable,
                "init": unit.init,
                "sources": sources,
                "fields": [field.model_dump() for field in unit.fields],
            }
        )
    result = {
        "status": "sources-verified",
        "runtime_sha256": manifest.runtime_sha256,
        "manifest_sha256": helpers.object_digest(manifest.model_dump(mode="json")),
        "machine": platform.machine(),
        "source_count": source_count,
        "field_count": field_count,
        "units": records,
    }
    helpers.write_record(work / "result.json", result)
    log.info(
        f"status=sources-verified source_count={source_count} field_count={field_count}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--auth", choices=("anonymous", "env", "secret"), default="anonymous"
    )
    parser.add_argument("--k8s-secret", default="")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--commit-to-new-branch", action="store_true")
    mode.add_argument("--check-sources-only", action="store_true")
    args = parser.parse_args()
    with new_work_directory(args.work_dir):
        raw = args.manifest.read_bytes()
        require(
            hashlib.sha256(raw).hexdigest() == args.manifest_sha256,
            "Manifest file SHA256 mismatch",
        )
        manifest = Manifest.model_validate_json(raw)
        require(
            args.commit_to_new_branch
            or (args.auth == "anonymous" and not args.k8s_secret),
            "Readonly modes require anonymous access without credentials",
        )
        require(
            not args.commit_to_new_branch
            or (
                args.auth in {"env", "secret"}
                and bool(args.k8s_secret) == (args.auth == "secret")
            ),
            "Commit requires explicit writable authentication",
        )
        verify_runtime(manifest, Path(__file__).resolve().parents[2])
        if args.check_sources_only:
            check_sources(manifest, args.work_dir)
        else:
            repo = open_repo(manifest.repository, args.auth, args.k8s_secret)
            run_in_work_directory(
                repo, manifest, args.work_dir, commit=args.commit_to_new_branch
            )


if __name__ == "__main__":
    main()
