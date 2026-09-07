"""The shared NOAA GRIB-index ref builder: byte ranges, message matching, and the
chunk indices the production HRRR virtual datasets resolve."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.config_models import DataVar
from reformatters.common.region_job import CoordinateValue
from reformatters.common.types import Dim, Timedelta, Timestamp
from reformatters.noaa import noaa_virtual_region_job as shared_region_job_module
from reformatters.noaa.hrrr.forecast_48_hour_virtual.region_job import (
    NoaaHrrrForecast48HourVirtualRegionJob,
)
from reformatters.noaa.hrrr.forecast_48_hour_virtual.template_config import (
    NoaaHrrrForecast48HourVirtualTemplateConfig,
)
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
from reformatters.noaa.hrrr.virtual_region_job import (
    NoaaHrrrForecastVirtualSourceFileCoord,
)
from reformatters.noaa.models import NoaaInternalAttrs
from reformatters.noaa.noaa_virtual_region_job import (
    NoaaVirtualRegionJob,
    NoaaVirtualSourceFileCoord,
)
from tests.noaa.grib_index_fixtures import (
    grib_section_0,
    stub_grib_source_file_reads,
)

TEMPLATE_CONFIG = NoaaHrrrForecast48HourVirtualTemplateConfig()
# The archive's first init, so its position along init_time is 0.
INIT_TIME = pd.Timestamp("2018-07-13T12:00")
TEMPLATE_END = pd.Timestamp("2018-07-14T00:00")
LEAD_TIME = pd.Timedelta("6h")


def get_var(path: str) -> NoaaHrrrDataVar:
    return next(v for v in TEMPLATE_CONFIG.data_vars if v.path == path)


@pytest.fixture(scope="module")
def template_ds() -> xr.DataTree:
    return TEMPLATE_CONFIG.get_template(TEMPLATE_END)


def make_job(
    template_ds: xr.DataTree, data_vars: Sequence[NoaaHrrrDataVar]
) -> NoaaHrrrForecast48HourVirtualRegionJob:
    return NoaaHrrrForecast48HourVirtualRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=template_ds,
        data_vars=data_vars,
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
    )


def coord(
    file_type: Literal["sfc", "prs", "nat"],
    data_vars: Sequence[NoaaHrrrDataVar],
    lead_time: pd.Timedelta = LEAD_TIME,
) -> NoaaHrrrForecastVirtualSourceFileCoord:
    return NoaaHrrrForecastVirtualSourceFileCoord(
        init_time=INIT_TIME,
        lead_time=lead_time,
        domain="conus",
        file_type=file_type,
        data_vars=data_vars,
    )


def fake_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    content: str,
    data_file_size: int | None = None,
) -> None:
    stub_grib_source_file_reads(
        monkeypatch,
        shared_region_job_module,
        tmp_path,
        lambda _url: content,
        data_file_size=data_file_size,
    )


# --- Byte ranges and message matching ---

_SFC_INDEX = (
    "1:0:d=2018071312:REFC:entire atmosphere:6 hour fcst:\n"
    "2:500:d=2018071312:TMP:2 m above ground:6 hour fcst:\n"
    "3:1500:d=2018071312:var discipline=0 center=7 local_table=1 parmcat=16 parm=201:entire atmosphere:6 hour fcst:\n"
    "4:2000:d=2018071312:APCP:surface:0-6 hour acc fcst:\n"
    "5:3000:d=2018071312:APCP:surface:5-6 hour acc fcst:\n"
)


def test_file_refs_window_disambiguation_and_skips_unmatched(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_index(monkeypatch, tmp_path, _SFC_INDEX)
    data_vars = [
        get_var("temperature_2m"),
        get_var("total_precipitation_run_total_surface"),  # 0-6 hour acc window
        get_var("total_precipitation_surface"),  # 5-6 hour acc window
    ]
    job = make_job(template_ds, data_vars)
    refs = job.file_refs(coord("sfc", data_vars), file_size=9000)

    by_name = {r.data_var.name: r for r in refs}
    # REFC and the unnamed experimental message are not in data_vars -> not emitted.
    assert set(by_name) == {
        "temperature_2m",
        "total_precipitation_run_total_surface",
        "total_precipitation_surface",
    }
    # A message's end byte is the next message's start; the last message ends at EOF.
    assert (by_name["temperature_2m"].offset, by_name["temperature_2m"].length) == (
        500,
        1000,
    )
    run_total = by_name["total_precipitation_run_total_surface"]
    one_hour = by_name["total_precipitation_surface"]
    assert (run_total.offset, run_total.length) == (2000, 1000)
    assert (one_hour.offset, one_hour.length) == (3000, 9000 - 3000)
    for ref in refs:
        assert ref.out_loc == {"init_time": INIT_TIME, "lead_time": LEAD_TIME}
        assert ref.location == (
            "s3://noaa-hrrr-bdp-pds/hrrr.20180713/conus/hrrr.t12z.wrfsfcf06.grib2"
        )


def test_file_refs_one_message_fills_two_variables_sharing_a_window(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # At lead 1h a run-total accumulation window (0->1) and the per-hour bucket (0->1)
    # render the identical idx window string, so the single matching message must
    # populate both variables rather than one silently displacing the other.
    fake_index(
        monkeypatch,
        tmp_path,
        "1:0:d=2018071312:APCP:surface:0-1 hour acc fcst:\n",
        data_file_size=1000,
    )
    data_vars = [
        get_var("total_precipitation_run_total_surface"),
        get_var("total_precipitation_surface"),
    ]
    job = make_job(template_ds, data_vars)
    refs = job.file_refs(
        coord("sfc", data_vars, lead_time=pd.Timedelta("1h")), file_size=1000
    )

    assert {r.data_var.name for r in refs} == {
        "total_precipitation_run_total_surface",
        "total_precipitation_surface",
    }
    for ref in refs:
        assert (ref.offset, ref.length) == (0, 1000)


def test_file_refs_duplicate_messages_keep_the_first_and_still_fill_every_variable(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A source may publish the same field twice, byte-distinct but identical in value.
    # Each variable takes the first message; the two variables that legitimately share
    # this window must still both be filled.
    fake_index(
        monkeypatch,
        tmp_path,
        "1:0:d=2018071312:APCP:surface:0-1 hour acc fcst:\n"
        "2:1000:d=2018071312:APCP:surface:0-1 hour acc fcst:\n",
    )
    data_vars = [
        get_var("total_precipitation_run_total_surface"),
        get_var("total_precipitation_surface"),
    ]
    job = make_job(template_ds, data_vars)
    refs = job.file_refs(
        coord("sfc", data_vars, lead_time=pd.Timedelta("1h")), file_size=2000
    )

    assert sorted(r.data_var.name for r in refs) == [
        "total_precipitation_run_total_surface",
        "total_precipitation_surface",
    ]
    for ref in refs:
        assert (ref.offset, ref.length) == (0, 1000)


def test_file_refs_skips_index_whose_bad_range_is_on_an_unmatched_message(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Rejecting the index discards the whole file, so every message's range is
    # checked - a corrupt range on a message no variable wants condemns it too.
    fake_index(
        monkeypatch,
        tmp_path,
        "1:0:d=2018071312:TMP:2 m above ground:6 hour fcst:\n"
        "2:1000:d=2018071312:REFC:entire atmosphere:6 hour fcst:\n",
    )
    data_vars = [get_var("temperature_2m")]  # REFC is not requested
    job = make_job(template_ds, data_vars)
    # REFC starts at the last byte, so its range is empty; TMP's range is fine.
    assert job.file_refs(coord("sfc", data_vars), file_size=1000) == []


def test_file_refs_matches_element_alternative_spellings(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Index element spellings vary by era and by the wgrib2 build that wrote the index
    # (deprecated GRIB parameters like TCOLWold, NCEP-local params as raw
    # "var discipline=..." strings), so grib_element_alternatives must also match.
    fake_index(
        monkeypatch,
        tmp_path,
        "1:0:d=2018071312:TCOLWold:entire atmosphere:6 hour fcst:\n"
        "2:1000:d=2018071312:var discipline=0 center=7 local_table=1 parmcat=7 parm=200"
        ":5000-2000 m above ground:5-6 hour min fcst:\n",
    )
    data_vars = [
        get_var("total_column_cloud_water_atmosphere"),  # TCOLW, alt TCOLWold
        get_var("minimum_updraft_helicity_5000_2000m"),  # MNUPHL, alt raw var string
    ]
    job = make_job(template_ds, data_vars)
    refs = job.file_refs(coord("sfc", data_vars), file_size=2500)

    by_name = {r.data_var.name: r for r in refs}
    assert set(by_name) == {
        "total_column_cloud_water_atmosphere",
        "minimum_updraft_helicity_5000_2000m",
    }
    tcolw = by_name["total_column_cloud_water_atmosphere"]
    mnuphl = by_name["minimum_updraft_helicity_5000_2000m"]
    assert (tcolw.offset, tcolw.length) == (0, 1000)
    assert (mnuphl.offset, mnuphl.length) == (1000, 2500 - 1000)


def test_file_refs_lead_0_instant_uses_anl_window(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_index(
        monkeypatch,
        tmp_path,
        "1:0:d=2018071312:TMP:2 m above ground:anl:\n",
        data_file_size=1000,
    )
    data_vars = [get_var("temperature_2m")]
    job = make_job(template_ds, data_vars)
    refs = job.file_refs(
        coord("sfc", data_vars, lead_time=pd.Timedelta("0h")), file_size=1000
    )

    assert [r.data_var.name for r in refs] == ["temperature_2m"]


def test_file_refs_skips_index_with_non_increasing_offsets(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_index(
        monkeypatch,
        tmp_path,
        "1:500:d=2018071312:TMP:2 m above ground:6 hour fcst:\n"
        "2:500:d=2018071312:REFC:entire atmosphere:6 hour fcst:\n",
    )
    data_vars = [get_var("temperature_2m")]
    job = make_job(template_ds, data_vars)
    assert job.file_refs(coord("sfc", data_vars), file_size=9000) == []


def test_file_refs_skips_index_reaching_past_the_data_file(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_index(monkeypatch, tmp_path, _SFC_INDEX)
    data_vars = [get_var("temperature_2m")]  # index says bytes 500..1500
    job = make_job(template_ds, data_vars)
    assert job.file_refs(coord("sfc", data_vars), file_size=1200) == []


def test_stubbed_source_file_reads_are_all_keyed_on_the_index_url(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # file_refs downloads `<data url>.idx` and reads the data url's header, so a
    # make_index keyed on the url must see the index's for both. Handed the data url it
    # would fetch the whole GRIB file where a test supplies a real index.
    requested: list[str] = []

    def make_index(url: str) -> str:
        requested.append(url)
        return _SFC_INDEX

    stub_grib_source_file_reads(
        monkeypatch, shared_region_job_module, tmp_path, make_index
    )
    data_vars = [get_var("temperature_2m")]
    job = make_job(template_ds, data_vars)

    assert job.file_refs(coord("sfc", data_vars), file_size=9000)
    assert requested
    assert all(url.endswith(".idx") for url in requested), requested


def test_file_refs_skips_index_whose_offsets_drifted_but_stayed_in_bounds(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A re-uploaded object leaves its sidecar index naming byte ranges that are no
    # longer message boundaries but are still inside the file, so the ranges alone
    # cannot condemn it: refs built from them exist and do not decode.
    fake_index(monkeypatch, tmp_path, _SFC_INDEX)
    # The last entry is at 3000 in a 9000-byte file, so a message declaring more than
    # the 6000 bytes that remain cannot be the one the index describes.
    monkeypatch.setattr(
        shared_region_job_module,
        "s3_read_bytes",
        lambda url, **kwargs: grib_section_0(7000),
    )
    data_vars = [get_var("temperature_2m")]
    job = make_job(template_ds, data_vars)
    assert job.file_refs(coord("sfc", data_vars), file_size=9000) == []


def test_file_refs_skips_index_whose_middle_message_was_resized(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The case a first-entry check cannot see: message 1 is untouched, so its length still
    # agrees, but a later message changed size and shifted every offset after it. The
    # index's last offset then lands mid-message, where there is no GRIB magic.
    fake_index(monkeypatch, tmp_path, _SFC_INDEX)
    monkeypatch.setattr(
        shared_region_job_module,
        "s3_read_bytes",
        lambda url, *, region, start, end: (
            grib_section_0(500) if start == 0 else bytes(16)
        ),
    )
    data_vars = [get_var("temperature_2m")]
    job = make_job(template_ds, data_vars)

    assert job.file_refs(coord("sfc", data_vars), file_size=9000) == []


def test_file_refs_skips_index_whose_last_offset_is_past_the_file_end(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A stale index can name a last offset beyond the object, where a ranged GET would
    # 416. The guard must reject without reading rather than raise into the pool.
    fake_index(monkeypatch, tmp_path, _SFC_INDEX)

    def must_not_read(url: str, **kwargs: object) -> bytes:
        raise AssertionError("read attempted past the end of the file")

    monkeypatch.setattr(shared_region_job_module, "s3_read_bytes", must_not_read)
    data_vars = [get_var("temperature_2m")]
    job = make_job(template_ds, data_vars)

    assert job.file_refs(coord("sfc", data_vars), file_size=3005) == []


def test_file_refs_accepts_an_index_that_omits_trailing_messages(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Real early-HRRR indexes list fewer messages than the object holds, so the last
    # entry's message ends well before EOF. Requiring it to end AT the file end rejects
    # 7 of 845 healthy real objects, so the guard only requires that it fit.
    fake_index(monkeypatch, tmp_path, _SFC_INDEX)
    monkeypatch.setattr(
        shared_region_job_module,
        "s3_read_bytes",
        lambda url, **kwargs: grib_section_0(500),  # 500 of the 6000 bytes remaining
    )
    data_vars = [get_var("temperature_2m")]
    job = make_job(template_ds, data_vars)

    assert [
        r.data_var.name for r in job.file_refs(coord("sfc", data_vars), file_size=9000)
    ] == ["temperature_2m"]


def test_file_refs_skips_index_whose_offsets_are_uniformly_shifted(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Displacing every offset by the same amount leaves each span the right length and
    # inside the file, so neither the bounds check nor a message length read at byte 0
    # sees anything wrong. Reading where the index says the message starts does: the
    # offset lands mid-message, which carries no GRIB magic.
    shifted = "".join(
        line.replace(f":{line.split(':')[1]}:", f":{int(line.split(':')[1]) + 100}:", 1)
        + "\n"
        for line in _SFC_INDEX.splitlines()
    )
    fake_index(monkeypatch, tmp_path, shifted)
    monkeypatch.setattr(
        shared_region_job_module,
        "s3_read_bytes",
        lambda url, *, region, start, end: (
            grib_section_0(500) if start == 0 else bytes(16)
        ),
    )
    data_vars = [get_var("temperature_2m")]
    job = make_job(template_ds, data_vars)

    assert job.file_refs(coord("sfc", data_vars), file_size=9000) == []


def test_file_refs_skips_an_object_too_short_to_hold_a_grib_header(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A truncated or non-GRIB object is untrusted source state, so it discards the one
    # file like any other stale index rather than raising and taking the worker with it.
    fake_index(monkeypatch, tmp_path, _SFC_INDEX)
    monkeypatch.setattr(
        shared_region_job_module,
        "s3_read_bytes",
        lambda url, *, region, start, end: b"GRI",
    )
    data_vars = [get_var("temperature_2m")]
    job = make_job(template_ds, data_vars)

    assert job.file_refs(coord("sfc", data_vars), file_size=9000) == []


def test_file_refs_skips_empty_index(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_index(monkeypatch, tmp_path, "")
    data_vars = [get_var("temperature_2m")]
    job = make_job(template_ds, data_vars)
    assert job.file_refs(coord("sfc", data_vars), file_size=9000) == []


# --- Chunk indices the production HRRR virtual datasets resolve ---
#
# Vertical labels travel from the template coordinate into the ref, so these pin the
# chunks the refs actually address. INIT_TIME is init_time index 0, lead 6h is
# lead_time index 6, and y/x are single-chunk.


def test_root_var_chunk_index(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_index(
        monkeypatch,
        tmp_path,
        "1:0:d=2018071312:TMP:2 m above ground:6 hour fcst:\n",
        data_file_size=1000,
    )
    var = get_var("temperature_2m")
    job = make_job(template_ds, [var])

    (ref,) = job.file_refs(coord("sfc", [var]), file_size=1000)

    assert job.chunk_key(ref.out_loc, ref.data_var) == (0, 6, 0, 0)


def test_pressure_level_chunk_indices(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_index(
        monkeypatch,
        tmp_path,
        "1:0:d=2018071312:TMP:1000 mb:6 hour fcst:\n"
        "2:1000:d=2018071312:TMP:500 mb:6 hour fcst:\n"
        "3:2000:d=2018071312:TMP:50 mb:6 hour fcst:\n",
    )
    var = get_var("pressure_level/temperature")
    job = make_job(template_ds, [var])

    refs = job.file_refs(coord("prs", [var]), file_size=3000)

    # pressure_level runs 1000 mb down to 50 mb in 39 single-level chunks.
    assert [job.chunk_key(ref.out_loc, ref.data_var) for ref in refs] == [
        (0, 6, 0, 0, 0),
        (0, 6, 0, 0, 20),
        (0, 6, 0, 0, 38),
    ]
    assert [(ref.offset, ref.length) for ref in refs] == [
        (0, 1000),
        (1000, 1000),
        (2000, 1000),
    ]


def test_model_level_chunk_indices(
    template_ds: xr.DataTree, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_index(
        monkeypatch,
        tmp_path,
        "1:0:d=2018071312:TMP:1 hybrid level:6 hour fcst:\n"
        "2:1000:d=2018071312:TMP:50 hybrid level:6 hour fcst:\n",
    )
    var = get_var("model_level/temperature")
    job = make_job(template_ds, [var])

    refs = job.file_refs(coord("nat", [var]), file_size=2000)

    assert [job.chunk_key(ref.out_loc, ref.data_var) for ref in refs] == [
        (0, 6, 0, 0, 0),
        (0, 6, 0, 0, 49),
    ]


@pytest.mark.parametrize(
    ("path", "dim"),
    [
        ("pressure_level/temperature", "pressure_level"),
        ("model_level/temperature", "model_level"),
    ],
)
def test_level_labels_are_the_template_coordinate_values(
    template_ds: xr.DataTree, path: str, dim: str
) -> None:
    var = get_var(path)
    job = make_job(template_ds, [var])
    index = template_ds[path].to_dataset().get_index(dim)

    labels = [
        label
        for entries in job._message_lookup([var], lead_hours=6).values()
        for _, level_label in entries
        for label in level_label.values()
    ]
    assert sorted(labels) == sorted(index)
    assert np.asarray(labels).dtype == index.dtype == np.int64


@pytest.mark.parametrize(
    ("path", "dim", "expected"),
    [
        ("pressure_level/temperature", "pressure_level", ["1000 mb", "975 mb"]),
        (
            "model_level/temperature",
            "model_level",
            ["1 hybrid level", "2 hybrid level"],
        ),
    ],
)
def test_grib_index_level_renders_hrrr_levels(
    template_ds: xr.DataTree, path: str, dim: str, expected: list[str]
) -> None:
    """The idx level strings the lookup matches HRRR messages on."""
    level_format = get_var(path).internal_attrs.grib_index_level
    index = template_ds[path].to_dataset().get_index(dim)
    assert [level_format.format(level=level) for level in index[:2]] == expected


# --- A consumer with its own DataVar type and a float64 vertical coordinate ---


class OtherInternalAttrs(NoaaInternalAttrs):
    pass


class OtherDataVar(DataVar[OtherInternalAttrs]):
    pass


class OtherSourceFileCoord(NoaaVirtualSourceFileCoord[OtherDataVar]):
    init_time: Timestamp

    def get_url(self) -> str:
        return "s3://example-nodd-bucket/other.grib2"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"init_time": self.init_time, "lead_time": self.lead_time}


class OtherVirtualRegionJob(NoaaVirtualRegionJob[OtherDataVar, OtherSourceFileCoord]):
    source_location_prefix: ClassVar[str] = "s3://example-nodd-bucket/"
    source_bucket_region: ClassVar[str] = "us-east-1"
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("6h")


def other_pressure_var() -> OtherDataVar:
    hrrr_var = get_var("pressure_level/temperature")
    internal_attrs = hrrr_var.internal_attrs
    return OtherDataVar(
        name=hrrr_var.name,
        group=hrrr_var.group,
        encoding=hrrr_var.encoding,
        attrs=hrrr_var.attrs,
        internal_attrs=OtherInternalAttrs(
            **{
                field: getattr(internal_attrs, field)
                for field in NoaaInternalAttrs.model_fields
            }
        ),
    )


@pytest.fixture(scope="module")
def float_level_template_ds(template_ds: xr.DataTree) -> xr.DataTree:
    """The HRRR template with float64 pressure levels, one of them fractional."""
    levels = (
        template_ds["pressure_level"]
        .to_dataset()
        .get_index("pressure_level")
        .astype("float64")
        .to_numpy()
        .copy()
    )
    levels[-1] = 0.01
    tree = template_ds.copy()
    tree["pressure_level"] = xr.DataTree(
        tree["pressure_level"].to_dataset().assign_coords(pressure_level=levels)
    )
    return tree


def test_float_levels_render_and_resolve_for_a_non_hrrr_data_var(
    float_level_template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_index(
        monkeypatch,
        tmp_path,
        "1:0:d=2018071312:TMP:1000 mb:6 hour fcst:\n"
        "2:1000:d=2018071312:TMP:0.01 mb:6 hour fcst:\n",
    )
    var = other_pressure_var()
    source_file_coord = OtherSourceFileCoord(
        init_time=INIT_TIME, lead_time=LEAD_TIME, data_vars=[var]
    )
    # A coord parameterized on its own DataVar type holds the caller's objects rather
    # than pydantic-revalidated copies of them.
    assert source_file_coord.data_vars[0] is var

    job = OtherVirtualRegionJob(
        tmp_store=Path("unused-tmp.zarr"),
        template_ds=float_level_template_ds,
        data_vars=[var],
        append_dim="init_time",
        region=slice(0, 1),
        reformat_job_name="test",
    )
    refs = job.file_refs(source_file_coord, file_size=2000)

    # "{level:g} mb" renders 1000.0 as "1000 mb", matching how the source index spells it.
    assert [ref.out_loc["pressure_level"] for ref in refs] == [1000.0, 0.01]
    assert [job.chunk_key(ref.out_loc, ref.data_var) for ref in refs] == [
        (0, 6, 0, 0, 0),
        (0, 6, 0, 0, 38),
    ]
