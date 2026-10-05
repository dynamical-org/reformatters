import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import template_utils
from reformatters.common.config_models import ROOT
from reformatters.common.region_job import CoordinateValue
from reformatters.common.types import Dim
from reformatters.noaa.noaa_grib_index import grib_index_window_str
from reformatters.noaa.refs.forecast_virtual.region_job import (
    NoaaRefsRegionJob,
)
from reformatters.noaa.refs.forecast_virtual.template_config import (
    NoaaRefsForecastVirtualTemplateConfig,
)


class TinyProductsConfig(NoaaRefsForecastVirtualTemplateConfig):
    def _spatial_info(
        self,
    ) -> tuple[
        tuple[int, int], tuple[float, float, float, float], tuple[float, float], str
    ]:
        _, _, resolution, crs = super()._spatial_info()
        return (2, 3), (0.0, 0.0, 9000.0, 6000.0), resolution, crs


def test_complete_schema_dimensions_and_encoding() -> None:
    config = NoaaRefsForecastVirtualTemplateConfig()
    variables = config.data_vars
    assert len(variables) == len({v.path for v in variables}) == 360
    assert sum(v.has_statistic for v in variables) == 82
    assert {
        families: sum(
            v.has_statistic and v.internal_attrs.source_families == families
            for v in variables
        )
        for families in (("mean", "sprd"), ("mean",), ("sprd",))
    } == {("mean", "sprd"): 64, ("mean",): 13, ("sprd",): 5}
    assert (
        sum(
            v.internal_attrs.source_families[0] in ("pmmn", "lpmm", "avrg")
            for v in variables
        )
        == 13
    )
    assert (
        sum(
            v.internal_attrs.source_families[0] in ("prob", "eas", "ffri")
            for v in variables
        )
        == 252
    )
    assert all(
        v.has_statistic for v in variables if "mean" in v.internal_attrs.source_families
    )
    assert all(v.group is ROOT for v in variables)
    assert set(config.dimension_coordinates()) == set(config.all_dims)
    assert config.groups == (ROOT,)
    assert "ensemble_member" not in config.all_dims
    config._assert_valid_structure()
    for var in variables:
        assert isinstance(var.encoding.chunks, tuple)
        assert len(var.encoding.chunks) == len(config.data_var_dims(var))
        assert ("statistic" in config.data_var_dims(var)) == var.has_statistic
        assert np.isnan(var.encoding.fill_value)


def test_offset_fields_have_mean_only_slices_and_kelvin_deviations() -> None:
    config = NoaaRefsForecastVirtualTemplateConfig()
    variables = {v.name: v for v in config.data_vars}
    offset_names = {
        v.name
        for v in variables.values()
        if v.has_statistic
        and any(
            codec.get("name") == "scale_offset"
            and codec.get("configuration", {}).get("offset", 0) != 0
            for codec in v.encoding.filters or ()
        )
    }
    assert offset_names == {
        "temperature_2m",
        "temperature_surface",
        "temperature_250hpa",
        "temperature_500hpa",
        "temperature_700hpa",
        "temperature_850hpa",
        "temperature_925hpa",
        "dew_point_temperature_2m",
        "dew_point_temperature_500hpa",
        "dew_point_temperature_700hpa",
        "dew_point_temperature_850hpa",
        "dew_point_temperature_925hpa",
        "soil_temperature_0m",
    }
    differences = {
        v.name
        for v in variables.values()
        if not v.has_statistic and v.internal_attrs.source_families == ("sprd",)
    }
    assert differences == {f"{name}_standard_deviation" for name in offset_names}
    for name in offset_names:
        statistic = variables[name]
        deviation = variables[f"{name}_standard_deviation"]
        assert statistic.internal_attrs.source_families == ("mean",)
        assert statistic.has_statistic
        assert not deviation.has_statistic
        assert statistic.attrs.units == "degree_Celsius"
        assert deviation.attrs.units == "K"
        assert deviation.attrs.standard_name is None
        assert deviation.attrs.short_name == deviation.name
        assert (
            deviation.attrs.long_name
            == f"Standard deviation of {statistic.attrs.long_name.lower()}"
        )
        assert not deviation.encoding.filters
        assert f"not populated; read {deviation.name}" in (
            statistic.attrs.comment or ""
        )
        assert config.data_var_dims(statistic) == (
            "init_time",
            "statistic",
            "lead_time",
            "y",
            "x",
        )
        assert config.data_var_dims(deviation) == ("init_time", "lead_time", "y", "x")
    soil_comment = variables["soil_temperature_0m"].attrs.comment or ""
    assert "raw zero before Celsius conversion" in soil_comment
    assert "use a land mask" in soil_comment


def test_single_isobaric_names_use_hpa_without_renaming_layers() -> None:
    config = NoaaRefsForecastVirtualTemplateConfig()
    single_levels = []
    layers = []
    for var in config.data_vars:
        level = var.internal_attrs.grib_index_level
        if re.fullmatch(r"\d+ mb", level):
            pressure = level.removesuffix(" mb")
            assert re.search(rf"_{pressure}hpa(?:_|$)", var.name)
            assert not re.search(r"\dmb(?:_|$)", var.name)
            single_levels.append(var)
        elif re.search(r"\dmb(?:_|$)", var.name):
            assert "hpa" not in var.name
            assert "mb" in level
            layers.append(var)
    assert len(single_levels) == 91
    assert layers
    names = {v.name for v in config.data_vars}
    assert {
        "geopotential_height_500hpa",
        "geopotential_height_850hpa",
        "geopotential_height_925hpa",
        "temperature_850hpa",
        "temperature_925hpa",
    } <= names


@pytest.mark.parametrize(
    ("element", "category"),
    [
        ("CRAIN", "rain"),
        ("CSNOW", "snow"),
        ("CICEP", "ice_pellets"),
        ("CFRZR", "freezing_rain"),
    ],
)
def test_categorical_probabilities_describe_member_indicators(
    element: str, category: str
) -> None:
    config = NoaaRefsForecastVirtualTemplateConfig()
    variables = {v.name: v for v in config.data_vars}
    name = f"probability_of_categorical_{category}_surface"
    var = variables[name]
    label = category.replace("_", " ")
    assert var.attrs.long_name == f"Probability of categorical {label} (surface)"
    assert var.attrs.comment == f"Percentage of members diagnosing {label}."
    assert var.attrs.units == "percent"
    assert var.internal_attrs.source_families == ("prob",)
    assert var.internal_attrs.grib_element == element
    assert var.internal_attrs.grib_index_selectors == ("prob >=1 <0",)
    assert "statistic" not in config.data_var_dims(var)
    assert f"probability_of_{category}_fraction_surface_at_least_1" not in variables
    assert variables[f"{category}_fraction_surface"].has_statistic


def test_probability_level_display_keeps_exact_source_level() -> None:
    source_level = "entire atmosphere (considered as a single layer)"
    variables = [
        v
        for v in NoaaRefsForecastVirtualTemplateConfig().data_vars
        if not v.has_statistic
        and v.internal_attrs.source_families == ("prob",)
        and v.internal_attrs.grib_index_level == source_level
    ]
    assert len(variables) == 13
    for var in variables:
        assert "(entire atmosphere)" in var.attrs.long_name
        assert "(entire atmosphere)" in (var.attrs.comment or "")
        assert "(considered as a single layer)" not in var.attrs.long_name


@pytest.mark.parametrize(
    ("element", "count", "text"),
    [
        (
            "CAT",
            27,
            "0 at most 4; 1 above 4 through 8; 2 above 8 through 12; 3 above 12",
        ),
        (
            "FLGHT",
            4,
            "1 = LIFR (low instrument flight rules), 2 = IFR (instrument flight rules), 3 = MVFR (marginal visual flight rules), and 4 = VFR (visual flight rules)",
        ),
    ],
)
def test_probability_category_codes_describe_thresholds_without_output_flags(
    element: str, count: int, text: str
) -> None:
    variables = [
        v
        for v in NoaaRefsForecastVirtualTemplateConfig().data_vars
        if v.internal_attrs.grib_element == element
    ]
    assert len(variables) == count
    for var in variables:
        assert var.attrs.units == "percent"
        assert text in (var.attrs.comment or "")
        assert var.attrs.flag_values is None
        assert var.attrs.flag_meanings is None
        assert var.internal_attrs.grib_index_selectors is not None
        assert var.internal_attrs.grib_index_selectors[0].startswith("prob >=")
        if element == "CAT":
            assert "Higher categories indicate stronger diagnosed turbulence" in (
                var.attrs.comment or ""
            )
        else:
            assert "Lower categories indicate greater restriction" in (
                var.attrs.comment or ""
            )


def test_mixed_axes_and_utf8_labels_roundtrip_and_resolve_chunk_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = TinyProductsConfig()
    path = tmp_path / "templates/latest.zarr"
    monkeypatch.setattr(TinyProductsConfig, "template_path", lambda self: path)
    config.update_template()
    metadata = json.loads((path / "statistic/zarr.json").read_text())
    assert metadata["data_type"] == "string"
    assert metadata["codecs"][0]["name"] == "vlen-utf8"
    end = config.append_dim_start + pd.Timedelta("12h")
    tree = config.get_template(end)
    ds = tree.to_dataset()
    assert ds.statistic.values.tolist() == ["mean", "standard_deviation"]
    assert ds.temperature_2m.dims == ("init_time", "statistic", "lead_time", "y", "x")
    sparse_name = "probability_matched_mean_total_precipitation_1_hour_surface"
    assert ds[sparse_name].dims == ("init_time", "lead_time", "y", "x")
    assert ds.temperature_500hpa.dims == ds.temperature_2m.dims
    assert ds.init_time.size == 2
    job = NoaaRefsRegionJob(
        tmp_store=tmp_path,
        template_ds=tree,
        data_vars=config.data_vars,
        append_dim="init_time",
        region=slice(0, 2),
        reformat_job_name="test",
        processing_mode="backfill",
    )
    vars_by_name = {v.name: v for v in config.data_vars}
    loc: dict[Dim, CoordinateValue] = {
        "init_time": config.append_dim_start,
        "lead_time": pd.Timedelta("1h"),
    }
    assert job._resolve_chunk_keys(
        [
            ({**loc, "statistic": "mean"}, vars_by_name["temperature_2m"]),
            (
                {**loc, "statistic": "standard_deviation"},
                vars_by_name["temperature_2m"],
            ),
            (loc, vars_by_name[sparse_name]),
        ]
    ) == [(0, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 0, 0)]
    roundtrip_path = tmp_path / "roundtrip.zarr"
    template_utils.write_metadata(tree, roundtrip_path)
    with xr.open_datatree(
        roundtrip_path, engine="zarr", decode_timedelta=True
    ) as reopened:
        assert reopened.statistic.values.tolist() == ["mean", "standard_deviation"]
        assert reopened.temperature_2m.sel(statistic="standard_deviation").dims == (
            "init_time",
            "lead_time",
            "y",
            "x",
        )
        assert reopened[sparse_name].dims == ("init_time", "lead_time", "y", "x")
        assert reopened.init_time.size == 2


@pytest.mark.parametrize(
    ("dims", "message"),
    [
        (("statistic", "lead_time", "y", "x"), "append dimension"),
        (("init_time", "lead_time", "statistic", "y", "x"), "ordered subset"),
        (("init_time", "statistic", "statistic", "lead_time", "y", "x"), "unique"),
        (("init_time", "statistic", "lead_time", "latitude", "x"), "group dimensions"),
    ],
)
def test_per_variable_dimension_hook_rejects_invalid_subsets(
    dims: tuple[Dim, ...], message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = NoaaRefsForecastVirtualTemplateConfig()
    original = config.data_var_dims
    monkeypatch.setattr(
        type(config),
        "data_var_dims",
        lambda self, var: dims if var.name == "temperature_2m" else original(var),
    )
    with pytest.raises(AssertionError, match=message):
        config._assert_valid_structure()


def test_product_caps_and_vector_spread_have_no_inherited_fill() -> None:
    config = NoaaRefsForecastVirtualTemplateConfig()
    variables = {v.name: v for v in config.data_vars}
    precipitation = variables["total_precipitation_6_hour_surface"]
    probability = variables[
        "neighborhood_probability_of_total_precipitation_6_hour_surface_above_12p7"
    ]
    assert precipitation.attrs.units == "kg m-2"
    assert probability.attrs.units == "percent"
    assert "12.7 kg m-2" in (probability.attrs.comment or "")
    assert "preceding 6 hours of accumulation" in (probability.attrs.comment or "")
    assert not probability.encoding.filters
    for name in (
        "visibility_surface",
        "geopotential_height_cloud_ceiling",
        "geopotential_height_cloud_base",
    ):
        var = variables[name]
        assert np.isnan(var.encoding.fill_value)
        assert var.internal_attrs.source_fill_value is None
        assert "NaN" not in (var.attrs.comment or "")
    for name in ("wind_speed_10m", "vertical_wind_shear_0_609m"):
        assert "vector spread" in (variables[name].attrs.comment or "")
    ffri = [
        v for v in config.data_vars if v.internal_attrs.source_families == ("ffri",)
    ]
    assert len(ffri) == 17
    assert all((v.attrs.comment or "").endswith("Mask values < -0.1.") for v in ffri)


def test_statistic_comments_describe_layers_signs_indicators_and_terrain() -> None:
    variables = {v.name: v for v in NoaaRefsForecastVirtualTemplateConfig().data_vars}
    for layer in ("180_0mb", "90_0mb"):
        for quantity in (
            "convective_available_potential_energy",
            "convective_inhibition",
        ):
            comment = variables[f"{quantity}_{layer}"].attrs.comment or ""
            assert (
                "pressure differences from the surface, not isobaric levels" in comment
            )
            if quantity == "convective_inhibition":
                assert "The mean is negative" in comment
    assert "The mean is negative" in (
        variables["convective_inhibition_surface"].attrs.comment or ""
    )
    for component, axis, direction in (("u", "x", "eastward"), ("v", "y", "northward")):
        for level in (
            "10m",
            "250hpa",
            "500hpa",
            "550hpa",
            "700hpa",
            "850hpa",
            "925hpa",
        ):
            comment = variables[f"wind_{component}_{level}"].attrs.comment or ""
            assert f"grid's {axis} dimension, not {direction} velocity" in comment
    for category in ("freezing_rain", "ice_pellets", "rain", "snow"):
        comment = variables[f"{category}_fraction_surface"].attrs.comment or ""
        assert f"members diagnosing {category.replace('_', ' ')}" in comment
        assert "total weight of contributing members" in comment
        assert "weighted population standard deviation of the same 0/1" in comment
    assert "differences in terrain height between member grids" in (
        variables["geopotential_height_surface"].attrs.comment or ""
    )
    assert not variables[
        "probability_matched_mean_geopotential_height_surface"
    ].attrs.comment


@pytest.mark.parametrize(
    ("name", "reader_text", "source_threshold"),
    [
        (
            "neighborhood_probability_of_total_precipitation_6_hour_surface_above_12p7",
            "6 hour total precipitation (surface) above 12.7 kg m-2",
            "prob >12.7",
        ),
        (
            "ensemble_agreement_scale_probability_of_total_precipitation_6_hour_surface_above_12p7",
            "6 hour total precipitation (surface) above 12.7 kg m-2",
            "prob >12.7",
        ),
        (
            "probability_of_clear_air_turbulence_175hpa_at_least_3",
            "clear air turbulence (175 mb) at least 3",
            "prob >=3 <0",
        ),
        (
            "probability_of_clear_air_turbulence_175hpa_at_least_1_below_2",
            "clear air turbulence (175 mb) at least 1 and below 2",
            "prob >=1 <2",
        ),
        (
            "probability_of_wind_speed_10m_at_least_9_and_relative_humidity_2m_at_most_20",
            "10 m wind speed at least 9 m s-1 and 2 m relative humidity at most 20 percent",
            "prob >=9 <20",
        ),
        (
            "neighborhood_probability_of_precipitation_exceeding_6_hour_flash_flood_guidance_6_hour_surface",
            "6 hour precipitation above the 6 hour flash flood guidance depth",
            "prob >6",
        ),
        (
            "neighborhood_probability_of_precipitation_exceeding_10_year_average_recurrence_interval_6_hour_surface",
            "6 hour precipitation above the 10 year average recurrence interval depth",
            "prob >10",
        ),
        (
            "probability_of_convective_inhibition_90_0mb_below_minus_50",
            "convective inhibition (90-0 mb above ground) below -50 J kg-1",
            "prob <-50",
        ),
    ],
)
def test_probability_reader_text_preserves_exact_source_selectors(
    name: str, reader_text: str, source_threshold: str
) -> None:
    var = next(
        v for v in NoaaRefsForecastVirtualTemplateConfig().data_vars if v.name == name
    )
    assert reader_text in var.attrs.long_name
    assert reader_text in (var.attrs.comment or "")
    assert var.internal_attrs.grib_index_selectors is not None
    assert var.internal_attrs.grib_index_selectors[0] == source_threshold
    if var.internal_attrs.grib_element == "CAT":
        assert var.attrs.long_name.endswith(reader_text)
        assert "<0" not in (var.attrs.comment or "")
    if var.internal_attrs.grib_element == "CIN":
        assert "Convective inhibition is negative" in (var.attrs.comment or "")
        assert "The mean is negative" not in (var.attrs.comment or "")
        assert "pressure differences from the surface" in (var.attrs.comment or "")
    if var.internal_attrs.source_families == ("ffri",):
        assert "kg m-2" not in var.attrs.long_name
        assert "kg m-2" not in (var.attrs.comment or "")
        assert (var.attrs.comment or "").endswith("Mask values < -0.1.")


def test_cloud_soil_and_radar_comments_keep_raw_values_and_statistic_scope() -> None:
    variables = {v.name: v for v in NoaaRefsForecastVirtualTemplateConfig().data_vars}
    cloud = variables["geopotential_height_cloud_base"]
    comment = cloud.attrs.comment or ""
    for text in (
        "When none contribute",
        "In some source messages",
        "multiples of 8192 m",
        "near 3616 m",
        "cannot be distinguished from true heights",
    ):
        assert text in comment
    assert "Mask" not in comment
    for name in ("soil_temperature_0m", "volumetric_soil_moisture_0m"):
        comment = variables[name].attrs.comment or ""
        assert "Over water, absent member soil values" in comment
        assert "These cells are not marked; use a land mask" in comment
        assert "Mask" not in comment
    for name in (
        "composite_reflectivity",
        "derived_radar_reflectivity_1000m",
        "hourly_maximum_radar_reflectivity_1000m",
        "echo_top",
    ):
        statistic = variables[name]
        sparse = variables[f"probability_matched_mean_{name}"]
        assert "Standard deviation includes members' no-echo markers" in (
            statistic.attrs.comment or ""
        )
        assert "The probability-matched mean retains" not in (
            statistic.attrs.comment or ""
        )
        assert "Standard deviation includes" not in (sparse.attrs.comment or "")
        if name == "echo_top":
            assert (sparse.attrs.comment or "").endswith("Mask values < 0.")
            assert "Mask" not in (statistic.attrs.comment or "")
        elif name == "hourly_maximum_radar_reflectivity_1000m":
            assert "0 dBZ can mean no echo or a valid reflectivity" in (
                sparse.attrs.comment or ""
            )
        else:
            assert "-20, -10 or 0 dBZ" in (sparse.attrs.comment or "")
            assert "other negative dBZ values can be valid" in (
                sparse.attrs.comment or ""
            )
        assert np.isnan(statistic.encoding.fill_value)
        assert np.isnan(sparse.encoding.fill_value)
        assert statistic.internal_attrs.source_fill_value is None
        assert sparse.internal_attrs.source_fill_value is None
    for var in (
        cloud,
        variables["soil_temperature_0m"],
        variables["volumetric_soil_moisture_0m"],
    ):
        assert np.isnan(var.encoding.fill_value)
        assert var.internal_attrs.source_fill_value is None


@pytest.mark.parametrize("element", ["MXUPHL", "MAXREF", "MAXUVV"])
def test_hourly_maximum_probabilities_keep_instantaneous_source_selection(
    element: str,
) -> None:
    variables = [
        v
        for v in NoaaRefsForecastVirtualTemplateConfig().data_vars
        if v.internal_attrs.grib_element == element
        and v.internal_attrs.source_families == ("prob",)
    ]
    assert variables
    for var in variables:
        assert var.attrs.step_type == "max"
        assert "Maximum value over the preceding hour" in (var.attrs.comment or "")
        assert var.internal_attrs.grib_index_step_type == "instant"
        assert grib_index_window_str(var, 6) == "6 hour fcst"
