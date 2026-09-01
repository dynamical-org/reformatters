import pandas as pd

from reformatters.noaa.gefs.analysis.region_job import GefsAnalysisSourceFileCoord
from reformatters.noaa.gefs.analysis.template_config import GefsAnalysisTemplateConfig
from reformatters.noaa.gefs.gefs_config_models import NoaaGefsDataVar
from reformatters.noaa.gefs.utils import _index_data_vars


def _mean_sea_level_pressure() -> NoaaGefsDataVar:
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
