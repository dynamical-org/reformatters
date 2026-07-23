import numpy as np

from reformatters.nasa.imerg.analysis_early.template_config import (
    NasaImergAnalysisEarlyTemplateConfig,
)
from reformatters.nasa.imerg.analysis_late.template_config import (
    NasaImergAnalysisLateTemplateConfig,
)


def test_dataset_ids_and_names() -> None:
    early = NasaImergAnalysisEarlyTemplateConfig()
    late = NasaImergAnalysisLateTemplateConfig()
    assert early.dataset_attributes.dataset_id == "nasa-imerg-analysis-early"
    assert early.dataset_attributes.name == "NASA IMERG analysis, early"
    assert late.dataset_attributes.dataset_id == "nasa-imerg-analysis-late"
    assert late.dataset_attributes.name == "NASA IMERG analysis, late"


def test_grid_orientation_and_size() -> None:
    coords = NasaImergAnalysisEarlyTemplateConfig().dimension_coordinates()
    lat = coords["latitude"]
    lon = coords["longitude"]
    assert lat.shape == (1800,)
    assert lon.shape == (3600,)
    # Latitude descends (north first) to match the repo convention.
    assert lat[0] == 89.95
    assert lat[-1] == -89.95
    assert np.all(np.diff(lat) < 0)
    # Longitude ascends across pixel centers.
    assert lon[0] == -179.95
    assert lon[-1] == 179.95


def test_materialized_data_vars() -> None:
    config = NasaImergAnalysisEarlyTemplateConfig()
    by_name = {v.name: v for v in config.data_vars}
    assert set(by_name) == {
        "precipitation_surface",
        "precipitation_quality_index_surface",
    }

    precip = by_name["precipitation_surface"]
    assert precip.attrs.units == "kg m-2 s-1"
    assert precip.attrs.standard_name == "precipitation_flux"
    assert precip.attrs.short_name == "prate"
    assert precip.attrs.step_type == "avg"
    assert precip.internal_attrs.keep_mantissa_bits == 7
    assert precip.internal_attrs.h5_path == "//Grid/precipitation"
    assert precip.internal_attrs.source_scale == 1.0 / 3600.0

    qi = by_name["precipitation_quality_index_surface"]
    assert qi.attrs.standard_name is None
    assert qi.internal_attrs.h5_path == "//Grid/precipitationQualityIndex"
    assert qi.internal_attrs.source_scale == 1.0
