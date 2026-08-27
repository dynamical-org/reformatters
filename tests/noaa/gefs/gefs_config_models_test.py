"""Unit tests for GEFS file type branching and URL generation."""

import numpy as np
import pandas as pd

from reformatters.common.config_models import DataVarAttrs, Encoding
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_B22_TRANSITION_DATE,
    GEFS_CURRENT_ARCHIVE_START,
    GEFS_REFORECAST_END,
    GEFS_REFORECAST_START,
    GEFS_S_FILE_MAX,
    GEFSDataVar,
    GefsEnsembleSourceFileCoord,
    GEFSFileType,
    GEFSInternalAttrs,
)


def _make_gefs_var(gefs_file_type: GEFSFileType) -> GEFSDataVar:
    return GEFSDataVar(
        name="test_var",
        encoding=Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=(1,),
            shards=None,
        ),
        attrs=DataVarAttrs(
            units="K",
            long_name="Test variable",
            short_name="test",
            step_type="instant",
        ),
        internal_attrs=GEFSInternalAttrs(
            keep_mantissa_bits=10,
            grib_element="TMP",
            grib_description="2 m temperature",
            grib_index_level="2 m above ground",
            index_position=1,
            gefs_file_type=gefs_file_type,
        ),
    )


def _make_ensemble_coord(
    init_time: pd.Timestamp,
    lead_time: pd.Timedelta,
    gefs_file_type: GEFSFileType,
    ensemble_member: int = 0,
) -> GefsEnsembleSourceFileCoord:
    return GefsEnsembleSourceFileCoord(
        init_time=init_time,
        lead_time=lead_time,
        data_vars=[_make_gefs_var(gefs_file_type)],
        ensemble_member=ensemble_member,
    )


# ---------------------------------------------------------------------------
# gefs_file_type property — current archive (>= GEFS_CURRENT_ARCHIVE_START)
# ---------------------------------------------------------------------------


def test_file_type_current_archive_s_plus_a_short_lead() -> None:
    coord = _make_ensemble_coord(
        GEFS_CURRENT_ARCHIVE_START, GEFS_S_FILE_MAX, "s+a", ensemble_member=1
    )
    assert coord.gefs_file_type == "s"


def test_file_type_current_archive_s_plus_a_long_lead() -> None:
    coord = _make_ensemble_coord(
        GEFS_CURRENT_ARCHIVE_START,
        GEFS_S_FILE_MAX + pd.Timedelta(hours=6),
        "s+a",
        ensemble_member=1,
    )
    assert coord.gefs_file_type == "a"


def test_file_type_current_archive_s_plus_b_short_lead() -> None:
    coord = _make_ensemble_coord(
        GEFS_CURRENT_ARCHIVE_START, pd.Timedelta(hours=6), "s+b", ensemble_member=1
    )
    assert coord.gefs_file_type == "s"


def test_file_type_current_archive_s_plus_b_long_lead() -> None:
    coord = _make_ensemble_coord(
        GEFS_CURRENT_ARCHIVE_START,
        GEFS_S_FILE_MAX + pd.Timedelta(hours=6),
        "s+b",
        ensemble_member=1,
    )
    assert coord.gefs_file_type == "b"


def test_file_type_current_archive_plain_a() -> None:
    coord = _make_ensemble_coord(
        GEFS_CURRENT_ARCHIVE_START, pd.Timedelta(hours=24), "a", ensemble_member=1
    )
    assert coord.gefs_file_type == "a"


def test_file_type_current_archive_plain_b() -> None:
    coord = _make_ensemble_coord(
        GEFS_CURRENT_ARCHIVE_START, pd.Timedelta(hours=24), "b", ensemble_member=1
    )
    assert coord.gefs_file_type == "b"


# ---------------------------------------------------------------------------
# s+b-b22 branching (transition date dependent)
# ---------------------------------------------------------------------------


def test_file_type_s_plus_b_minus_b22_after_transition_short_lead() -> None:
    coord = _make_ensemble_coord(
        GEFS_B22_TRANSITION_DATE,
        pd.Timedelta(hours=6),
        "s+b-b22",
        ensemble_member=1,
    )
    assert coord.gefs_file_type == "s"


def test_file_type_s_plus_b_minus_b22_after_transition_long_lead() -> None:
    coord = _make_ensemble_coord(
        GEFS_B22_TRANSITION_DATE,
        GEFS_S_FILE_MAX + pd.Timedelta(hours=6),
        "s+b-b22",
        ensemble_member=1,
    )
    assert coord.gefs_file_type == "b"


def test_file_type_s_plus_b_minus_b22_before_transition() -> None:
    # Before GEFS_B22_TRANSITION_DATE, s+b-b22 always uses "b"
    before_transition = GEFS_B22_TRANSITION_DATE - pd.Timedelta(hours=6)
    # But before_transition must still be >= GEFS_CURRENT_ARCHIVE_START
    assert before_transition >= GEFS_CURRENT_ARCHIVE_START
    coord = _make_ensemble_coord(
        before_transition,
        pd.Timedelta(hours=6),  # short lead - but still "b" before transition
        "s+b-b22",
        ensemble_member=1,
    )
    assert coord.gefs_file_type == "b"


# ---------------------------------------------------------------------------
# gefs_file_type — intermediate period (GEFS_REFORECAST_END <= t < GEFS_CURRENT_ARCHIVE_START)
# ---------------------------------------------------------------------------


def test_file_type_intermediate_s_plus_a_maps_to_a() -> None:
    mid = GEFS_REFORECAST_END + pd.Timedelta(days=30)
    assert mid < GEFS_CURRENT_ARCHIVE_START
    coord = _make_ensemble_coord(mid, pd.Timedelta(hours=6), "s+a", ensemble_member=1)
    assert coord.gefs_file_type == "a"


def test_file_type_intermediate_s_plus_b_maps_to_b() -> None:
    mid = GEFS_REFORECAST_END + pd.Timedelta(days=30)
    coord = _make_ensemble_coord(mid, pd.Timedelta(hours=6), "s+b", ensemble_member=1)
    assert coord.gefs_file_type == "b"


def test_file_type_intermediate_b_maps_to_b() -> None:
    mid = GEFS_REFORECAST_END + pd.Timedelta(days=30)
    coord = _make_ensemble_coord(mid, pd.Timedelta(hours=6), "b", ensemble_member=1)
    assert coord.gefs_file_type == "b"


# ---------------------------------------------------------------------------
# gefs_file_type — reforecast period (GEFS_REFORECAST_START <= t < GEFS_REFORECAST_END)
# ---------------------------------------------------------------------------


def test_file_type_reforecast_period() -> None:
    reforecast_time = GEFS_REFORECAST_START + pd.Timedelta(days=100)
    assert reforecast_time < GEFS_REFORECAST_END
    coord = _make_ensemble_coord(
        reforecast_time, pd.Timedelta(hours=24), "s+a", ensemble_member=1
    )
    assert coord.gefs_file_type == "reforecast"


# ---------------------------------------------------------------------------
# get_url — current archive
# ---------------------------------------------------------------------------


def test_get_url_current_archive_s_file_ensemble() -> None:
    coord = _make_ensemble_coord(
        pd.Timestamp("2023-01-01T00"),
        pd.Timedelta(hours=6),
        "s+a",
        ensemble_member=1,
    )
    url = coord.get_url()
    assert "gefs.20230101/00/atmos/pgrb2s" in url
    assert "gep01" in url
    assert "0p25" in url
    assert "f006" in url


def test_get_url_current_archive_control_member() -> None:
    coord = _make_ensemble_coord(
        pd.Timestamp("2023-06-15T12"),
        pd.Timedelta(hours=3),
        "s+a",
        ensemble_member=0,
    )
    url = coord.get_url()
    assert "gec00" in url
    assert "12/atmos" in url


def test_get_url_current_archive_long_lead_a_file() -> None:
    coord = _make_ensemble_coord(
        pd.Timestamp("2023-01-01T00"),
        GEFS_S_FILE_MAX + pd.Timedelta(hours=6),
        "s+a",
        ensemble_member=1,
    )
    url = coord.get_url()
    # Long lead time → falls back to "a" file at 0.5 deg resolution
    assert "pgrb2a" in url
    assert "0p50" in url


# ---------------------------------------------------------------------------
# get_url — intermediate period (pre-current-archive)
# ---------------------------------------------------------------------------


def test_get_url_intermediate_period() -> None:
    mid = GEFS_REFORECAST_END + pd.Timedelta(days=30)
    assert mid < GEFS_CURRENT_ARCHIVE_START

    coord = _make_ensemble_coord(mid, pd.Timedelta(hours=6), "a", ensemble_member=1)
    url = coord.get_url()
    # Intermediate period uses the older URL format (no /atmos/ sub-path)
    assert "atmos" not in url
    assert "pgrb2af" in url


# ---------------------------------------------------------------------------
# get_fallback_url and get_index_url
# ---------------------------------------------------------------------------


def test_get_fallback_url_replaces_host() -> None:
    coord = _make_ensemble_coord(
        pd.Timestamp("2023-01-01T00"),
        pd.Timedelta(hours=6),
        "s+a",
        ensemble_member=1,
    )
    primary_url = coord.get_url()
    fallback_url = coord.get_fallback_url()
    assert coord.primary_base_url in primary_url
    assert coord.fallback_base_url in fallback_url
    # The filename at the end of the URL should be the same
    assert primary_url.split("/")[-1] == fallback_url.split("/")[-1]


def test_get_index_url_appends_idx() -> None:
    coord = _make_ensemble_coord(
        pd.Timestamp("2023-01-01T00"),
        pd.Timedelta(hours=6),
        "s+a",
        ensemble_member=1,
    )
    assert coord.get_index_url().endswith(".idx")
    assert coord.get_index_url(fallback=True).endswith(".idx")
