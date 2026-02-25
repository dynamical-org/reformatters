import pandas as pd

from reformatters.ecmwf.aifs.forecast.region_job import (
    AIFS_SINGLE_PATH_CHANGE_DATE,
    EcmwfAifsForecastSourceFileCoord,
)
from reformatters.ecmwf.aifs.forecast.template_config import (
    EcmwfAifsForecastTemplateConfig,
)


def test_source_file_coord_url_before_path_change() -> None:
    config = EcmwfAifsForecastTemplateConfig()
    coord = EcmwfAifsForecastSourceFileCoord(
        init_time=pd.Timestamp("2024-07-01T12:00"),
        lead_time=pd.Timedelta("6h"),
        data_var_group=list(config.data_vars[:1]),
    )
    url = coord.get_url()
    assert "/aifs/0p25/oper/" in url
    assert "20240701120000-6h-oper-fc.grib2" in url

    idx_url = coord.get_index_url()
    assert idx_url.endswith(".index")


def test_source_file_coord_url_after_path_change() -> None:
    config = EcmwfAifsForecastTemplateConfig()
    coord = EcmwfAifsForecastSourceFileCoord(
        init_time=AIFS_SINGLE_PATH_CHANGE_DATE,
        lead_time=pd.Timedelta("12h"),
        data_var_group=list(config.data_vars[:1]),
    )
    url = coord.get_url()
    assert "/aifs-single/0p25/oper/" in url
    assert "20250226000000-12h-oper-fc.grib2" in url


def test_source_file_coord_out_loc() -> None:
    config = EcmwfAifsForecastTemplateConfig()
    init_time = pd.Timestamp("2024-04-01T00:00")
    lead_time = pd.Timedelta("6h")
    coord = EcmwfAifsForecastSourceFileCoord(
        init_time=init_time,
        lead_time=lead_time,
        data_var_group=list(config.data_vars[:1]),
    )
    out = coord.out_loc()
    assert out["init_time"] == init_time
    assert out["lead_time"] == lead_time
