import numpy as np

from reformatters.common.config_models import ROOT
from reformatters.google.weathernext2.forecast_historical_virtual.template_config import (
    GoogleWeathernext2ForecastHistoricalVirtualTemplateConfig,
)
from reformatters.google.weathernext2.forecast_operational_virtual.template_config import (
    GoogleWeathernext2ForecastOperationalVirtualTemplateConfig,
)
from reformatters.google.weathernext2.forecast_virtual.template_config import (
    GoogleWeathernext2DataVar,
    GoogleWeathernext2ForecastVirtualTemplateConfig,
)

HISTORICAL = GoogleWeathernext2ForecastHistoricalVirtualTemplateConfig()
OPERATIONAL = GoogleWeathernext2ForecastOperationalVirtualTemplateConfig()


def _var(
    config: GoogleWeathernext2ForecastVirtualTemplateConfig, path: str
) -> GoogleWeathernext2DataVar:
    return next(var for var in config.data_vars if var.path == path)


def test_products_have_distinct_ids_and_archive_extents() -> None:
    assert HISTORICAL.dataset_id == "google-weathernext2-forecast-historical-virtual"
    assert OPERATIONAL.dataset_id == "google-weathernext2-forecast-operational-virtual"
    assert HISTORICAL.append_dim_start.year == 2022
    assert OPERATIONAL.append_dim_start.year == 2025
    historical_inits = HISTORICAL.append_dim_coordinates(
        np.datetime64("2026-01-01T00:00")
    )
    assert len(historical_inits) == 4384
    assert historical_inits[-1].isoformat() == "2024-12-31T18:00:00"


def test_each_product_contains_all_variables_and_required_attribution() -> None:
    for config in (HISTORICAL, OPERATIONAL):
        assert len(config.data_vars) == 14
        assert sum(var.group is ROOT for var in config.data_vars) == 8
        assert sum(var.group == "pressure_level" for var in config.data_vars) == 6
        assert config.dataset_attributes.license == "CC-BY-4.0"
        assert config.dataset_attributes.attribution == (
            "Google requires this attribution: © 2025 DeepMind Technologies Limited's "
            "machine learning models used to create the experimental data made available "
            "at https://developers.google.com/earth-engine/datasets/catalog/"
            "projects_gcp-public-data-weathernext_assets_weathernext_2_0_0 under CC BY "
            "4.0 licence terms. This data is intended for experimental modelling only "
            "and is not intended, validated, or approved for real world use. Use of the "
            "third-party materials referred to in the Acknowledgements section may be "
            "governed by separate terms and conditions or license provisions. Your use "
            "of the third-party materials is subject to any such terms and you should "
            "check that you can comply with any applicable restrictions or terms and "
            "conditions before use."
        )


def test_historical_coordinate_statistics_are_fixed() -> None:
    coords = {coord.name: coord for coord in HISTORICAL.coords}
    assert coords["init_time"].attrs.statistics_approximate is not None
    assert coords["init_time"].attrs.statistics_approximate.max == "2024-12-31T18:00:00"
    assert coords["valid_time"].attrs.statistics_approximate is not None
    assert (
        coords["valid_time"].attrs.statistics_approximate.max == "2025-01-15T18:00:00"
    )


def test_historical_encoding_preserves_native_annual_chunks() -> None:
    root = _var(HISTORICAL, "temperature_2m")
    pressure = _var(HISTORICAL, "pressure_level/temperature")

    assert root.encoding.chunks == (1, 4, 1, 721, 1440)
    assert pressure.encoding.chunks == (1, 4, 1, 721, 1440, 13)
    assert root.encoding.filters is not None
    assert [codec["name"] for codec in root.encoding.filters] == ["scale_offset"]
    assert pressure.encoding.filters is not None
    assert [codec["name"] for codec in pressure.encoding.filters] == [
        "scale_offset",
        "transpose",
    ]
    assert pressure.encoding.filters[-1] == {
        "name": "transpose",
        "configuration": {"order": (0, 1, 2, 5, 3, 4)},
    }


def test_operational_encoding_preserves_native_per_init_chunks() -> None:
    root = _var(OPERATIONAL, "temperature_2m")
    pressure = _var(OPERATIONAL, "pressure_level/temperature")

    assert root.encoding.chunks == (1, 1, 1, 721, 1440)
    assert pressure.encoding.chunks == (1, 1, 1, 721, 1440, 1)
    assert root.encoding.filters is not None
    assert [codec["name"] for codec in root.encoding.filters] == ["scale_offset"]
    assert pressure.encoding.filters is not None
    assert [codec["name"] for codec in pressure.encoding.filters] == ["scale_offset"]


def test_every_data_var_decodes_native_blosc_lz4_shuffle() -> None:
    for config in (HISTORICAL, OPERATIONAL):
        for var in config.data_vars:
            assert var.encoding.serializer == {
                "name": "bytes",
                "configuration": {"endian": "little"},
            }
            assert var.encoding.compressors is not None
            assert len(var.encoding.compressors) == 1
            compressor = var.encoding.compressors[0]
            assert compressor["name"] == "blosc"
            assert compressor["configuration"] == {
                "typesize": 4,
                "cname": "lz4",
                "clevel": 5,
                "shuffle": "shuffle",
                "blocksize": 0,
            }
            assert var.internal_attrs.keep_mantissa_bits == "no-rounding"


def test_read_time_unit_conversions_use_standard_filters() -> None:
    for config in (HISTORICAL, OPERATIONAL):
        temperature = _var(config, "temperature_2m")
        precipitation = _var(config, "total_precipitation_surface")
        geopotential = _var(config, "pressure_level/geopotential_height")
        assert temperature.encoding.filters is not None
        assert temperature.encoding.filters[0]["configuration"]["offset"] == -273.15
        assert precipitation.encoding.filters is not None
        assert precipitation.encoding.filters[0]["configuration"]["scale"] == 0.001
        assert geopotential.encoding.filters is not None
        assert geopotential.encoding.filters[0]["configuration"]["scale"] == 9.80665
        assert all(
            codec["name"] in {"scale_offset", "transpose"}
            for var in (temperature, precipitation, geopotential)
            for codec in var.encoding.filters or ()
        )


def test_coordinate_values_match_native_spatial_grid_and_level_order() -> None:
    for config in (HISTORICAL, OPERATIONAL):
        coords = config.dimension_coordinates()
        np.testing.assert_array_equal(coords["latitude"], np.arange(-90, 90.25, 0.25))
        np.testing.assert_array_equal(coords["longitude"], np.arange(0, 360, 0.25))
        np.testing.assert_array_equal(
            coords["pressure_level"],
            np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]),
        )
