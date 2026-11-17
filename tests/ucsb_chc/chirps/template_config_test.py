import numpy as np

from reformatters.ucsb_chc.chirps.analysis_final.template_config import (
    UcsbChcChirpsAnalysisFinalTemplateConfig,
)
from reformatters.ucsb_chc.chirps.analysis_preliminary.template_config import (
    UcsbChcChirpsAnalysisPreliminaryTemplateConfig,
)
from reformatters.ucsb_chc.chirps.template_config import (
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    SOURCE_FILL_VALUE,
)


def test_dataset_ids_and_names() -> None:
    final = UcsbChcChirpsAnalysisFinalTemplateConfig()
    preliminary = UcsbChcChirpsAnalysisPreliminaryTemplateConfig()
    assert final.dataset_attributes.dataset_id == "ucsb-chc-chirps-analysis-final"
    assert final.dataset_attributes.name == "UCSB CHC CHIRPS analysis, final"
    assert (
        preliminary.dataset_attributes.dataset_id
        == "ucsb-chc-chirps-analysis-preliminary"
    )
    assert (
        preliminary.dataset_attributes.name == "UCSB CHC CHIRPS analysis, preliminary"
    )


def test_append_dim_starts() -> None:
    final = UcsbChcChirpsAnalysisFinalTemplateConfig()
    preliminary = UcsbChcChirpsAnalysisPreliminaryTemplateConfig()
    # The time coordinate labels each daily total at the start of the day it covers.
    assert str(final.append_dim_start) == "1981-01-01 00:00:00"
    assert str(preliminary.append_dim_start) == "2025-01-01 00:00:00"
    assert final.append_dim_frequency == preliminary.append_dim_frequency
    assert str(final.append_dim_frequency) == "1 days 00:00:00"

    first_time = final.dimension_coordinates()["time"][0]
    assert str(first_time) == "1981-01-01 00:00:00"


def test_grid_orientation_and_endpoints() -> None:
    coords = UcsbChcChirpsAnalysisFinalTemplateConfig().dimension_coordinates()
    lat = coords["latitude"]
    lon = coords["longitude"]
    assert lat.shape == (GRID_LAT_SIZE,)
    assert lon.shape == (GRID_LON_SIZE,)
    # Pixel centers of the 60N-60S, 180W-180E grid, latitude descending to match
    # the source files' north -> south row order.
    assert lat[0] == 59.975
    assert lat[-1] == -59.975
    assert np.all(np.diff(lat) < 0)
    assert lon[0] == -179.975
    assert lon[-1] == 179.975
    assert np.all(np.diff(lon) > 0)


def test_precipitation_surface_metadata() -> None:
    (precip,) = UcsbChcChirpsAnalysisFinalTemplateConfig().data_vars
    assert precip.name == "precipitation_surface"
    assert precip.attrs.short_name == "prate"
    assert precip.attrs.long_name == "Precipitation rate"
    assert precip.attrs.standard_name == "precipitation_flux"
    assert precip.attrs.units == "kg m-2 s-1"
    assert precip.attrs.step_type == "avg"
    assert precip.attrs.comment == (
        "Average precipitation rate over the 24 hours starting at the time coordinate. "
        "Units equivalent to mm/s. NaN over oceans."
    )
    assert precip.internal_attrs.keep_mantissa_bits == 8
    assert precip.internal_attrs.source_fill_value == SOURCE_FILL_VALUE
    assert np.isnan(precip.encoding.fill_value)
