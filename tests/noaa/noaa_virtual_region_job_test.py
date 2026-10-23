"""The shared NOAA GRIB-index ref builder: byte ranges, message matching, and the
chunk indices the production HRRR virtual datasets resolve."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from obstore.exceptions import GenericError

from reformatters.common import template_utils
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
from tests.common import virtual_region_job_test as virtual_fixtures
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
    # A source may publish the same field twice, with distinct byte ranges.
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


@pytest.mark.parametrize(
    "message", ["2:1000:d=2018071312", "2:1000:d=2018071312::surface:6 hour fcst:"]
)
def test_malformed_message_retains_healthy_refs_and_exact_boundaries(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    message: str,
) -> None:
    fake_index(
        monkeypatch,
        tmp_path,
        "1:0:d=2018071312:TMP:2 m above ground:6 hour fcst:\n"
        f"{message}\n"
        "3:2000:d=2018071312:RH:2 m above ground:6 hour fcst:\n",
    )
    variables = [get_var("temperature_2m"), get_var("relative_humidity_2m")]
    refs = make_job(template_ds, variables).file_refs(coord("sfc", variables), 3000)
    assert [(r.data_var.name, r.offset, r.length) for r in refs] == [
        ("temperature_2m", 0, 1000),
        ("relative_humidity_2m", 2000, 1000),
    ]
    assert "malformed GRIB index message" in caplog.text
    assert "GRIB messages absent" not in caplog.text


@pytest.mark.parametrize("offset", ["unreadable", ""])
def test_unreadable_offset_never_extends_the_preceding_ref(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    offset: str,
) -> None:
    fake_index(
        monkeypatch,
        tmp_path,
        "1:0:d=2018071312:TMP:2 m above ground:6 hour fcst:\n"
        f"2:{offset}:d=2018071312:RH:2 m above ground:6 hour fcst:\n"
        "3:2000:d=2018071312:REFC:entire atmosphere:6 hour fcst:\n",
    )
    variables = [get_var("temperature_2m")]
    assert (
        make_job(template_ds, variables).file_refs(coord("sfc", variables), 3000) == []
    )
    assert "message boundaries cannot be inferred safely" in caplog.text


@pytest.mark.parametrize("offset", [-1000, 4000])
def test_isolated_out_of_range_row_recovers_preceding_message_length(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    offset: int,
) -> None:
    fake_index(
        monkeypatch,
        tmp_path,
        "1:0:d=2018071312:TMP:2 m above ground:6 hour fcst:\n"
        f"2:{offset}:d=2018071312:REFC:entire atmosphere:6 hour fcst:\n"
        "3:2000:d=2018071312:RH:2 m above ground:6 hour fcst:\n",
    )
    monkeypatch.setattr(
        shared_region_job_module,
        "s3_read_bytes",
        lambda *args, **kwargs: grib_section_0(1000),
    )
    variables = [get_var("temperature_2m"), get_var("relative_humidity_2m")]
    refs = make_job(template_ds, variables).file_refs(coord("sfc", variables), 3000)
    assert [(r.data_var.name, r.offset, r.length) for r in refs] == [
        ("temperature_2m", 0, 1000),
        ("relative_humidity_2m", 2000, 1000),
    ]
    assert f"skipping index row at offset {offset}" in caplog.text


def test_completeness_reports_each_missing_vertical_cell(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_index(monkeypatch, tmp_path, "1:0:d=2018071312:TMP:1000 mb:6 hour fcst:\n")
    variables = [get_var("pressure_level/temperature")]
    refs = make_job(template_ds, variables).file_refs(coord("prs", variables), 1000)
    assert len(refs) == 1
    assert "pressure_level/temperature" in caplog.text
    assert "('pressure_level', 975)" in caplog.text
    assert "('pressure_level', 1000)" not in caplog.text
    assert caplog.records[-1].levelname == "ERROR"


def _noaa_index_for_failure(
    failure: str, window: str, block_bytes: int
) -> tuple[str, int]:
    middle = f"2:{block_bytes}:d=2024010100:BAD:2 m above ground:{window}:"
    if failure == "missing_message":
        middle = middle.replace(":BAD:", ":UNKNOWN:")
    elif failure == "malformed_message":
        middle = f"2:{block_bytes}:unreadable message"
    elif failure == "malformed_offset":
        middle = middle.replace(f":{block_bytes}:", ":unreadable:")
    elif failure == "out_of_range":
        middle = middle.replace(f":{block_bytes}:", ":99999:")
    blocks = 4 if failure == "recovery_read_error" else 3
    if failure == "recovery_read_error":
        middle += f"\n3:99999:d=2024010100:UNKNOWN:surface:{window}:"
    return (
        (
            f"1:0:d=2024010100:TMP:2 m above ground:{window}:\n"
            f"{middle}\n"
            f"4:{(blocks - 1) * block_bytes}:d=2024010100:RH:2 m above ground:{window}:\n"
        ),
        blocks,
    )


@pytest.mark.slow
@pytest.mark.parametrize("processing_mode", ["backfill", "update"])
@pytest.mark.parametrize(
    "failure",
    [
        "missing_message",
        "malformed_message",
        "malformed_offset",
        "out_of_range",
        "stale_tail",
        "tail_read_error",
        "download_error",
        "recovery_read_error",
    ],
)
def test_noaa_source_failures_commit_healthy_siblings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    processing_mode: Literal["backfill", "update"],
    failure: str,
) -> None:
    dataset = virtual_fixtures._make_dataset(tmp_path)
    tree = virtual_fixtures._create_template_ds(2)
    for name in ("bad", "other"):
        tree[name] = tree["temperature_2m"].copy(deep=True)
    template_utils.write_metadata(tree, dataset.store_factory)
    repo = virtual_fixtures._primary_repo(dataset.store_factory)
    block_bytes = virtual_fixtures.BLOCK_NBYTES
    prototype = virtual_fixtures.VirtualTestDataVar(name="temperature_2m")
    variables = [
        OtherDataVar(
            name=name,
            encoding=prototype.encoding,
            attrs=prototype.attrs,
            internal_attrs=OtherInternalAttrs(
                grib_element=element,
                grib_description=name,
                grib_index_level="2 m above ground",
                index_position=0,
                keep_mantissa_bits="no-rounding",
            ),
        )
        for name, element in (
            ("temperature_2m", "TMP"),
            ("bad", "BAD"),
            ("other", "RH"),
        )
    ]

    class LocalCoord(OtherSourceFileCoord):
        source_path: Path

        def get_url(self) -> str:
            return self.source_path.as_uri()

    class LocalJob(OtherVirtualRegionJob):
        def generate_source_file_coords(
            self,
            processing_region_ds: xr.Dataset,
            data_var_group: Sequence[OtherDataVar],
        ) -> Sequence[OtherSourceFileCoord]:
            return [
                LocalCoord(
                    init_time=init,
                    lead_time=lead,
                    data_vars=data_var_group,
                    source_path=tmp_path / f"source-{init.value}-{lead.value}.bin",
                )
                for init in pd.to_datetime(processing_region_ds.init_time.values)
                for lead in pd.to_timedelta(processing_region_ds.lead_time.values)
            ]

        def discover_available(
            self, pending: list[OtherSourceFileCoord]
        ) -> list[tuple[OtherSourceFileCoord, int]]:
            return [
                (coord, Path(coord.get_url().removeprefix("file://")).stat().st_size)
                for coord in pending
            ]

        def grib_message_length_at(
            self, coord: OtherSourceFileCoord, offset: int, file_size: int
        ) -> int | None:
            if (
                failure == "stale_tail"
                and coord.init_time == virtual_fixtures.APPEND_DIM_START
                and coord.lead_time == pd.Timedelta(0)
                and offset == 2 * block_bytes
            ):
                return None
            if (
                failure == "tail_read_error"
                and coord.init_time == virtual_fixtures.APPEND_DIM_START
                and coord.lead_time == pd.Timedelta(0)
                and offset == 2 * block_bytes
            ):
                raise OSError("unreadable source tail header")
            if (
                failure == "recovery_read_error"
                and coord.init_time == virtual_fixtures.APPEND_DIM_START
                and coord.lead_time == pd.Timedelta(0)
                and offset == block_bytes
            ):
                raise OSError("unreadable source header")
            assert offset + block_bytes <= file_size
            return block_bytes

    job = LocalJob(
        tmp_store=tmp_path / "unused",
        template_ds=tree,
        data_vars=variables,
        append_dim="init_time",
        region=slice(0, 2),
        reformat_job_name="test",
        processing_mode=processing_mode,
    )
    coords = job.source_file_coords()
    indices = {}
    for i, coord in enumerate(coords):
        assert isinstance(coord, LocalCoord)
        block = virtual_fixtures._block_values(i // 2, i % 2)
        window = "anl" if coord.lead_time == pd.Timedelta(0) else "6 hour fcst"
        index, blocks = _noaa_index_for_failure(
            failure if i == 0 else "", window, block_bytes
        )
        coord.source_path.write_bytes(block.tobytes() * blocks)
        indices[coord.get_index_url()] = index

    def download(url: str, dataset_id: str, **kwargs: object) -> Path:
        if failure == "download_error" and url == coords[0].get_index_url():
            raise OSError("unreadable source index")
        path = tmp_path / url.rsplit("/", 1)[-1]
        path.write_text(indices[url])
        return path

    monkeypatch.setattr(shared_region_job_module, "s3_download_to_disk", download)
    branch = "main"
    if processing_mode == "backfill":
        branch = "backfill"
        repo.create_branch(branch, repo.lookup_branch("main"))
    job.process_virtual(repo, [], branch, coords)
    assert virtual_fixtures._snapshot_count(repo, branch) > 1
    result = xr.open_zarr(repo.readonly_session(branch).store, decode_timedelta=True)
    whole_file_skipped = failure in {
        "malformed_offset",
        "stale_tail",
        "tail_read_error",
        "download_error",
    }
    for name in ("temperature_2m", "bad", "other"):
        if whole_file_skipped or name == "bad":
            assert np.isnan(result[name].isel(init_time=0, lead_time=0)).all()
        else:
            np.testing.assert_array_equal(
                result[name].isel(init_time=0, lead_time=0),
                virtual_fixtures._block_values(0, 0),
            )
        for init_idx, lead_idx in ((0, 1), (1, 0), (1, 1)):
            np.testing.assert_array_equal(
                result[name].isel(init_time=init_idx, lead_time=lead_idx),
                virtual_fixtures._block_values(init_idx, lead_idx),
            )
    assert any(record.levelname == "ERROR" for record in caplog.records)
    if failure == "recovery_read_error":
        assert "failed to read its length" in caplog.text


@pytest.mark.parametrize(
    ("error", "fatal"),
    [
        (OSError, False),
        (shared_region_job_module.HTTPError, False),
        (GenericError, False),
        (AssertionError, True),
        (ValueError, True),
    ],
)
def test_length_recovery_isolates_source_read_errors_and_preserves_invariants(
    template_ds: xr.DataTree,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: type[Exception],
    fatal: bool,
) -> None:
    variables = [get_var("temperature_2m")]
    job = make_job(template_ds, variables)

    def read_length(
        self: NoaaHrrrForecast48HourVirtualRegionJob,
        source: NoaaHrrrForecastVirtualSourceFileCoord,
        offset: int,
        file_size: int,
    ) -> int:
        if offset == 0:
            raise error("unreadable source header")
        return 1000

    monkeypatch.setattr(type(job), "grib_message_length_at", read_length)
    if fatal:
        with pytest.raises(error, match="unreadable source header"):
            job._message_lengths(coord("sfc", variables), [0, 9999, 2000], 3000)
    else:
        assert job._message_lengths(coord("sfc", variables), [0, 9999, 2000], 3000) == {
            2000: 1000
        }
        assert "failed to read its length" in caplog.text
        assert caplog.records[0].levelname == "ERROR"
