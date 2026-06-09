import numpy as np
import pandas as pd

from reformatters.nasa.imerg.analysis_early_v7.template_config import (
    NasaImergAnalysisEarlyV7TemplateConfig,
)


def test_template_config_attrs() -> None:
    config = NasaImergAnalysisEarlyV7TemplateConfig()
    assert config.dims == ("time", "latitude", "longitude")
    assert config.append_dim == "time"
    assert config.append_dim_start == pd.Timestamp("1998-01-01T00:00")
    assert config.append_dim_frequency == pd.Timedelta("30min")
    assert config.dataset_attributes.dataset_id == "nasa-imerg-analysis-early-v7"
    assert config.dataset_attributes.spatial_resolution == "0.1 degrees (~10km)"


def test_dimension_coordinates() -> None:
    config = NasaImergAnalysisEarlyV7TemplateConfig()
    dim_coords = config.dimension_coordinates()

    lat = dim_coords["latitude"]
    lon = dim_coords["longitude"]
    assert len(lat) == 1800
    assert len(lon) == 3600
    # Latitude descending (north -> south), longitude ascending; pixel centers.
    assert np.isclose(lat[0], 89.95)
    assert np.isclose(lat[-1], -89.95)
    assert np.isclose(lon[0], -179.95)
    assert np.isclose(lon[-1], 179.95)
    assert np.allclose(np.diff(lat), -0.1)
    assert np.allclose(np.diff(lon), 0.1)

    assert dim_coords["time"][0] == pd.Timestamp("1998-01-01T00:00")


def test_coordinate_configs() -> None:
    config = NasaImergAnalysisEarlyV7TemplateConfig()
    coord_names = {coord.name for coord in config.coords}
    assert coord_names == {"time", "latitude", "longitude", "spatial_ref"}


def test_data_vars() -> None:
    config = NasaImergAnalysisEarlyV7TemplateConfig()
    by_name = {v.name: v for v in config.data_vars}
    assert set(by_name) == {
        "precipitation_surface",
        "probability_of_liquid_precipitation_surface",
        "precipitation_quality_index_surface",
    }

    for var in config.data_vars:
        assert var.encoding.dtype == "float32"
        assert var.encoding.chunks == (1440, 45, 45)
        assert var.encoding.shards == (1440, 450, 450)

    precip = by_name["precipitation_surface"]
    assert precip.attrs.standard_name == "precipitation_flux"
    assert precip.attrs.units == "kg m-2 s-1"
    assert precip.attrs.step_type == "avg"
    assert precip.internal_attrs.h5_path == "//Grid/precipitation"
    assert precip.internal_attrs.source_fill_value == -9999.9
    # mm/hr -> kg m-2 s-1
    assert precip.internal_attrs.units_scale_factor == 1 / 3600

    prob = by_name["probability_of_liquid_precipitation_surface"]
    assert prob.attrs.units == "percent"
    assert prob.attrs.standard_name is None
    assert prob.internal_attrs.h5_path == "//Grid/probabilityLiquidPrecipitation"
    assert prob.internal_attrs.source_fill_value == -9999
    assert prob.internal_attrs.units_scale_factor == 1.0

    quality = by_name["precipitation_quality_index_surface"]
    assert quality.attrs.units == "1"
    assert quality.attrs.standard_name is None
    assert quality.internal_attrs.h5_path == "//Grid/precipitationQualityIndex"
