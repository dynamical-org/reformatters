from urllib.parse import urlparse

import numpy as np
import obstore
import pandas as pd
import pytest

from reformatters.common.config_models import DataVarAttrs, Encoding
from reformatters.noaa.gefs import utils
from reformatters.noaa.gefs.analysis.region_job import GefsAnalysisSourceFileCoord
from reformatters.noaa.gefs.analysis.template_config import GefsAnalysisTemplateConfig
from reformatters.noaa.gefs.gefs_config_models import (
    GEFSDataVar,
    GefsEnsembleSourceFileCoord,
    GEFSInternalAttrs,
)
from reformatters.noaa.gefs.utils import _index_data_vars, gefs_published_coords

_IN_PRODUCTION_INIT = pd.Timestamp.now().floor("D")
_SETTLED_INIT = pd.Timestamp("2024-06-01T00:00")


@pytest.fixture
def data_var() -> GEFSDataVar:
    return GEFSDataVar(
        name="temperature_2m",
        encoding=Encoding(
            dtype="float32", fill_value=np.nan, chunks=(1, 8, 4), shards=(1, 8, 4)
        ),
        attrs=DataVarAttrs(
            long_name="2 metre temperature",
            short_name="t2m",
            units="C",
            step_type="instant",
        ),
        internal_attrs=GEFSInternalAttrs(
            grib_element="TMP",
            grib_description='2[m] HTGL="Specified height level above ground"',
            grib_index_level="2 m above ground",
            gefs_file_type="a",
            index_position=10,
            keep_mantissa_bits=10,
        ),
    )


def _coord(
    data_var: GEFSDataVar, lead_hours: int, member: int, init_time: pd.Timestamp
) -> GefsEnsembleSourceFileCoord:
    return GefsEnsembleSourceFileCoord(
        init_time=init_time,
        lead_time=pd.Timedelta(hours=lead_hours),
        ensemble_member=member,
        data_vars=[data_var],
    )


def _key(coord: GefsEnsembleSourceFileCoord) -> str:
    return urlparse(coord.get_url()).path.removeprefix("/")


def _fake_listing(
    monkeypatch: pytest.MonkeyPatch, listed: dict[str, int]
) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake(store: obstore.store.ObjectStore, prefixes: list[str]) -> dict[str, int]:
        calls.append(prefixes)
        return listed

    monkeypatch.setattr(utils, "listed_keys_by_prefix", fake)
    return calls


def test_settled_cycle_returns_everything_without_listing(
    data_var: GEFSDataVar, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fake_listing(monkeypatch, {})
    coords = [_coord(data_var, hours, 0, _SETTLED_INIT) for hours in (0, 384, 840)]

    assert gefs_published_coords(coords) == coords
    assert calls == []


def test_drops_lead_times_past_the_published_frontier(
    data_var: GEFSDataVar, monkeypatch: pytest.MonkeyPatch
) -> None:
    coords = [
        _coord(data_var, hours, 0, _IN_PRODUCTION_INIT)
        for hours in (0, 240, 384, 390, 840)
    ]
    published_through_384 = {
        _key(coord): 9000
        for coord in coords
        if coord.lead_time <= pd.Timedelta(hours=384)
    }
    calls = _fake_listing(monkeypatch, published_through_384)

    kept = gefs_published_coords(coords)

    assert [coord.lead_time for coord in kept] == [
        pd.Timedelta(hours=hours) for hours in (0, 240, 384)
    ]
    assert len(calls) == 1


def test_keeps_an_unlisted_member_at_the_frontier(
    data_var: GEFSDataVar, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A member whose f384 has reached NOMADS but not yet the S3 mirror must still be
    # attempted, so gefs_download_file's fallback can reach it.
    mirrored = _coord(data_var, 384, 0, _IN_PRODUCTION_INIT)
    lagging = _coord(data_var, 384, 30, _IN_PRODUCTION_INIT)
    unpublished = _coord(data_var, 390, 30, _IN_PRODUCTION_INIT)
    _fake_listing(monkeypatch, {_key(mirrored): 9000})

    kept = gefs_published_coords([mirrored, lagging, unpublished])

    assert kept == [mirrored, lagging]


def test_drops_everything_when_the_cycle_has_not_started(
    data_var: GEFSDataVar, monkeypatch: pytest.MonkeyPatch
) -> None:
    coords = [_coord(data_var, hours, 0, _IN_PRODUCTION_INIT) for hours in (0, 3)]
    _fake_listing(monkeypatch, {})

    assert gefs_published_coords(coords) == []


def test_each_source_directory_gets_its_own_frontier(
    data_var: GEFSDataVar, monkeypatch: pytest.MonkeyPatch
) -> None:
    # s files stop at lead time 240h, so an s+a variable's coords span two directories
    # whose frontiers are independent.
    s_and_a_var = data_var.model_copy(
        update={
            "internal_attrs": data_var.internal_attrs.model_copy(
                update={"gefs_file_type": "s+a"}
            )
        }
    )
    s_file = _coord(s_and_a_var, 240, 0, _IN_PRODUCTION_INIT)
    a_file = _coord(s_and_a_var, 246, 0, _IN_PRODUCTION_INIT)
    late_a_file = _coord(s_and_a_var, 390, 0, _IN_PRODUCTION_INIT)
    assert "pgrb2sp25" in s_file.get_url()
    assert "pgrb2ap5" in a_file.get_url()

    calls = _fake_listing(monkeypatch, {_key(s_file): 100, _key(a_file): 200})

    assert gefs_published_coords([s_file, a_file, late_a_file]) == [s_file, a_file]
    assert len(calls[0]) == 2


def _mean_sea_level_pressure() -> GEFSDataVar:
    return next(
        var
        for var in GefsAnalysisTemplateConfig().data_vars
        if var.name == "pressure_reduced_to_mean_sea_level"
    )


def test_index_data_vars_renames_reforecast_elements() -> None:
    """The v12 reforecast index labels mean sea level pressure PRES, not PRMSL."""
    var = _mean_sea_level_pressure()
    assert var.internal_attrs.grib_element == "PRMSL"

    reforecast_coord = GefsAnalysisSourceFileCoord(
        init_time=pd.Timestamp("2000-06-01T00:00"),
        lead_time=pd.Timedelta("3h"),
        data_vars=[var],
    )
    operational_coord = GefsAnalysisSourceFileCoord(
        init_time=pd.Timestamp("2024-01-15T12:00"),
        lead_time=pd.Timedelta("3h"),
        data_vars=[var],
    )

    (reforecast_var,) = _index_data_vars(reforecast_coord)
    (operational_var,) = _index_data_vars(operational_coord)

    assert reforecast_var.internal_attrs.grib_element == "PRES"
    assert operational_var.internal_attrs.grib_element == "PRMSL"
    # Only the element name changes; the rest of the variable is untouched.
    assert reforecast_var.internal_attrs.grib_index_level == "mean sea level"
    assert reforecast_var.attrs == var.attrs
