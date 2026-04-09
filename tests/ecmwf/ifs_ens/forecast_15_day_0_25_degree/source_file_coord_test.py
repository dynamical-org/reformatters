from typing import Literal

import numpy as np
import pandas as pd

from reformatters.common.config_models import DataVarAttrs, Encoding
from reformatters.ecmwf.ecmwf_config_models import (
    EcmwfDataVar,
    EcmwfInternalAttrs,
    MarsSourceOverrides,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.source_file_coord import (
    DYNAMICAL_MARS_GRIB_BASE_URL,
    MarsSourceFileCoord,
    OpenDataSourceFileCoord,
)


def _make_var(
    name: str = "temperature_2m",
    grib_index_param: str = "2t",
    grib_index_level_type: Literal["sfc", "pl"] = "sfc",
    grib_index_level_value: float = float("nan"),
    grib_comment: str = "Temperature [C]",
    grib_description: str = '2[m] HTGL="Specified height level above ground"',
    mars: MarsSourceOverrides | None = None,
    open_data_date_available: pd.Timestamp | None = None,
    grib_index_param_lead_time_overrides: tuple[
        tuple[pd.Timedelta, pd.Timedelta, str], ...
    ] = (),
) -> EcmwfDataVar:
    return EcmwfDataVar(
        name=name,
        encoding=Encoding(dtype="float32", fill_value=np.nan, chunks=(1,), shards=None),
        attrs=DataVarAttrs(
            units="K",
            long_name="Test",
            short_name=grib_index_param,
            step_type="instant",
        ),
        internal_attrs=EcmwfInternalAttrs(
            keep_mantissa_bits=10,
            grib_index_param=grib_index_param,
            grib_index_level_type=grib_index_level_type,
            grib_index_level_value=grib_index_level_value,
            grib_element=grib_index_param,
            grib_comment=grib_comment,
            grib_description=grib_description,
            mars=mars,
            open_data_date_available=open_data_date_available,
            grib_index_param_lead_time_overrides=grib_index_param_lead_time_overrides,
        ),
    )


# ---------------------------------------------------------------------------
# OpenDataSourceFileCoord
# ---------------------------------------------------------------------------


def test_open_data_get_url_with_ifs_directory() -> None:
    coord = OpenDataSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01"),
        lead_time=pd.Timedelta("0h"),
        data_var_group=[],
        ensemble_member=0,
    )
    assert (
        coord.get_url()
        == "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/20250101/00z/ifs/0p25/enfo/20250101000000-0h-enfo-ef.grib2"
    )


def test_open_data_get_url_without_ifs_directory() -> None:
    coord = OpenDataSourceFileCoord(
        init_time=pd.Timestamp("2024-02-28"),
        lead_time=pd.Timedelta("0h"),
        data_var_group=[],
        ensemble_member=0,
    )
    assert (
        coord.get_url()
        == "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/20240228/00z/0p25/enfo/20240228000000-0h-enfo-ef.grib2"
    )


def test_open_data_index_step_is_none() -> None:
    coord = OpenDataSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01"),
        lead_time=pd.Timedelta("3h"),
        data_var_group=[],
        ensemble_member=0,
    )
    assert coord.index_step is None


def test_open_data_validate_grib_comment_unit_only_is_false() -> None:
    coord = OpenDataSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01"),
        lead_time=pd.Timedelta("3h"),
        data_var_group=[],
        ensemble_member=0,
    )
    assert coord.validate_grib_comment_unit_only is False


def test_open_data_out_loc() -> None:
    coord = OpenDataSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01"),
        lead_time=pd.Timedelta("6h"),
        data_var_group=[],
        ensemble_member=3,
    )
    assert coord.out_loc() == {
        "init_time": pd.Timestamp("2025-01-01"),
        "lead_time": pd.Timedelta("6h"),
        "ensemble_member": 3,
    }


def test_open_data_resolve_data_vars_no_overrides() -> None:
    var = _make_var()
    coord = OpenDataSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01"),
        lead_time=pd.Timedelta("3h"),
        data_var_group=[var],
        ensemble_member=0,
    ).resolve_data_vars()
    assert coord.data_var_group[0].internal_attrs.grib_index_param == "2t"


def test_open_data_resolve_data_vars_applies_lead_time_override() -> None:
    var = _make_var(
        name="wind_gust_10m",
        grib_index_param="10fg",
        grib_index_param_lead_time_overrides=(
            (pd.Timedelta("93h"), pd.Timedelta("144h"), "10fg3"),
        ),
    )
    coord_no_override = OpenDataSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01"),
        lead_time=pd.Timedelta("3h"),
        data_var_group=[var],
        ensemble_member=0,
    ).resolve_data_vars()
    assert coord_no_override.data_var_group[0].internal_attrs.grib_index_param == "10fg"

    coord_with_override = OpenDataSourceFileCoord(
        init_time=pd.Timestamp("2025-01-01"),
        lead_time=pd.Timedelta("96h"),
        data_var_group=[var],
        ensemble_member=0,
    ).resolve_data_vars()
    assert (
        coord_with_override.data_var_group[0].internal_attrs.grib_index_param == "10fg3"
    )


# ---------------------------------------------------------------------------
# MarsSourceFileCoord
# ---------------------------------------------------------------------------


def test_mars_get_url() -> None:
    coord = MarsSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01"),
        lead_time=pd.Timedelta("3h"),
        ensemble_member=0,
        data_var_group=[],
        request_type="cf_sfc",
    )
    assert coord.get_url() == f"{DYNAMICAL_MARS_GRIB_BASE_URL}/2024-01-01/cf_sfc.grib"
    assert (
        coord.get_index_url()
        == f"{DYNAMICAL_MARS_GRIB_BASE_URL}/2024-01-01/cf_sfc.grib.idx"
    )


def test_mars_index_step() -> None:
    coord = MarsSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01"),
        lead_time=pd.Timedelta("6h"),
        ensemble_member=0,
        data_var_group=[],
        request_type="cf_sfc",
    )
    assert coord.index_step == 6


def test_mars_validate_grib_comment_unit_only_is_true() -> None:
    coord = MarsSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01"),
        lead_time=pd.Timedelta("3h"),
        ensemble_member=0,
        data_var_group=[],
        request_type="cf_sfc",
    )
    assert coord.validate_grib_comment_unit_only is True


def test_mars_out_loc() -> None:
    coord = MarsSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01"),
        lead_time=pd.Timedelta("3h"),
        ensemble_member=5,
        data_var_group=[],
        request_type="pf_sfc_0",
    )
    assert coord.out_loc() == {
        "init_time": pd.Timestamp("2024-01-01"),
        "lead_time": pd.Timedelta("3h"),
        "ensemble_member": 5,
    }


def test_mars_get_request_type() -> None:
    assert MarsSourceFileCoord.get_request_type("sfc", 0) == "cf_sfc"
    assert MarsSourceFileCoord.get_request_type("sfc", 1) == "pf_sfc_0"
    assert MarsSourceFileCoord.get_request_type("sfc", 25) == "pf_sfc_0"
    assert MarsSourceFileCoord.get_request_type("sfc", 26) == "pf_sfc_1"
    assert MarsSourceFileCoord.get_request_type("pl", 0) == "cf_pl"
    assert MarsSourceFileCoord.get_request_type("pl", 1) == "pf_pl"


def test_mars_resolve_data_vars_no_mars_overrides() -> None:
    var = _make_var()
    coord = MarsSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01"),
        lead_time=pd.Timedelta("3h"),
        ensemble_member=0,
        data_var_group=[var],
        request_type="cf_sfc",
    ).resolve_data_vars()
    assert coord.data_var_group[0].internal_attrs.grib_index_param == "2t"
    assert coord.data_var_group[0].internal_attrs.grib_comment == "Temperature [C]"


def test_mars_resolve_data_vars_clears_open_data_date_available() -> None:
    """open_data_date_available tracks open data availability; MARS has all configured vars."""
    var = _make_var(open_data_date_available=pd.Timestamp("2024-11-13"))
    coord = MarsSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01"),
        lead_time=pd.Timedelta("3h"),
        ensemble_member=0,
        data_var_group=[var],
        request_type="cf_sfc",
    ).resolve_data_vars()
    assert coord.data_var_group[0].internal_attrs.open_data_date_available is None


def test_mars_resolve_data_vars_merges_overrides() -> None:
    var = _make_var(
        name="geopotential_height_500hpa",
        grib_index_param="gh",
        grib_index_level_type="pl",
        grib_index_level_value=500.0,
        grib_comment="Geopotential height [gpm]",
        mars=MarsSourceOverrides(
            grib_index_param="z",
            grib_comment="Geopotential (at the surface = orography) [m^2/s^2]",
            scale_factor=1 / 9.80665,
        ),
    )
    coord = MarsSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01"),
        lead_time=pd.Timedelta("3h"),
        ensemble_member=0,
        data_var_group=[var],
        request_type="cf_pl",
    ).resolve_data_vars()

    resolved = coord.data_var_group[0]
    assert resolved.internal_attrs.grib_index_param == "z"
    assert resolved.internal_attrs.grib_comment == (
        "Geopotential (at the surface = orography) [m^2/s^2]"
    )
    # scale_factor is merged into internal_attrs
    assert resolved.internal_attrs.scale_factor == 1 / 9.80665
    # mars is cleared after merge
    assert resolved.internal_attrs.mars is None


def test_mars_resolve_data_vars_preserves_unset_fields() -> None:
    var = _make_var(
        grib_comment="Temperature [C]",
        grib_description='2[m] HTGL="Specified height level above ground"',
        mars=MarsSourceOverrides(grib_index_param="2t"),
    )
    coord = MarsSourceFileCoord(
        init_time=pd.Timestamp("2024-01-01"),
        lead_time=pd.Timedelta("3h"),
        ensemble_member=0,
        data_var_group=[var],
        request_type="cf_sfc",
    ).resolve_data_vars()

    resolved = coord.data_var_group[0]
    # Only grib_index_param was overridden; others preserved
    assert resolved.internal_attrs.grib_comment == "Temperature [C]"
    assert (
        resolved.internal_attrs.grib_description
        == '2[m] HTGL="Specified height level above ground"'
    )
