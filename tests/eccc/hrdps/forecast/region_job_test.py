import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.download import FALLBACK_EXCEPTIONS
from reformatters.eccc.hrdps.forecast.region_job import (
    EcccHrdpsForecastRegionJob,
    EcccHrdpsForecastSourceFileCoord,
)
from reformatters.eccc.hrdps.forecast.template_config import (
    EcccHrdpsForecastTemplateConfig,
)

TEMPLATE_CONFIG = EcccHrdpsForecastTemplateConfig()
DATA_VARS_BY_NAME = {var.name: var for var in TEMPLATE_CONFIG.data_vars}


def make_coord(
    variable_name: str = "temperature_2m",
    init_time: str = "2026-07-09T12:00",
    lead_time: str = "6h",
) -> EcccHrdpsForecastSourceFileCoord:
    return EcccHrdpsForecastSourceFileCoord(
        init_time=pd.Timestamp(init_time),
        lead_time=pd.Timedelta(lead_time),
        data_var=DATA_VARS_BY_NAME[variable_name],
    )


def test_source_file_coord_urls() -> None:
    coord = make_coord("precipitation_surface", lead_time="48h")
    file_name = "20260709T12Z_MSC_HRDPS_APCP-Accum1h_Sfc_RLatLon0.0225_PT048H.grib2"

    assert coord.get_url() == (
        "https://s3-us-west-2.amazonaws.com/us-west-2.opendata.source.coop/"
        f"dynamical/eccc-hrdps-grib/20260709/12/048/{file_name}"
    )
    assert coord.get_datamart_url() == (
        "https://dd.weather.gc.ca/20260709/WXO-DD/model_hrdps/continental/2.5km/"
        f"12/048/{file_name}"
    )


@pytest.mark.parametrize(
    ("variable_name", "expected_field_and_level"),
    [
        ("temperature_2m", "TMP_AGL-2m"),
        ("wind_speed_80m", "WIND_AGL-80m"),
        ("wind_direction_10m", "WDIR_AGL-10m"),
        ("pressure_reduced_to_mean_sea_level", "PRMSL_MSL"),
        ("categorical_precipitation_type_surface", "PTYPE_Sfc"),
        ("snow_water_equivalent_surface", "SDWE_Sfc"),
        ("downward_long_wave_radiation_flux_surface", "DLWRF_Sfc"),
    ],
)
def test_every_field_and_level_reaches_the_file_name(
    variable_name: str, expected_field_and_level: str
) -> None:
    coord = make_coord(variable_name, lead_time="1h")
    assert coord.get_url().endswith(
        f"20260709T12Z_MSC_HRDPS_{expected_field_and_level}_RLatLon0.0225_PT001H.grib2"
    )


def test_generate_source_file_coords_skips_lead_time_0_when_unavailable() -> None:
    template_ds = TEMPLATE_CONFIG.get_template(pd.Timestamp("2026-07-09T12:00"))
    region_job = EcccHrdpsForecastRegionJob(
        tmp_store=Path("/tmp/not-used"),  # noqa: S108
        template_ds=template_ds,
        data_vars=TEMPLATE_CONFIG.data_vars,
        append_dim=TEMPLATE_CONFIG.append_dim,
        region=slice(0, 2),
        reformat_job_name="test",
    )
    processing_region_ds, _output_region_ds = region_job._get_region_datasets()

    instant_coords = region_job.generate_source_file_coords(
        processing_region_ds, [DATA_VARS_BY_NAME["temperature_2m"]]
    )
    assert len(instant_coords) == 2 * 49
    assert min(c.lead_time for c in instant_coords) == pd.Timedelta("0h")

    # The source publishes no lead time 0 file for accumulations, and its lead time 0
    # total cloud cover field carries no information.
    for variable_name in (
        "precipitation_surface",
        "categorical_precipitation_type_surface",
        "downward_short_wave_radiation_flux_surface",
        "total_cloud_cover_atmosphere",
    ):
        coords = region_job.generate_source_file_coords(
            processing_region_ds, [DATA_VARS_BY_NAME[variable_name]]
        )
        assert len(coords) == 2 * 48
        assert min(c.lead_time for c in coords) == pd.Timedelta("1h")


@pytest.mark.parametrize(
    ("init_time", "expected_first_source"),
    [
        # The Datamart has the newest runs hours before our archive copy does,
        # and holds only the last ~30 days.
        ("2026-08-24T00:00", "dd.weather.gc.ca"),
        ("2026-07-09T00:00", "opendata.source.coop"),
    ],
)
def test_download_file_source_order(
    init_time: str,
    expected_first_source: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2026-08-24T12:00")),
    )
    requested_urls: list[str] = []

    def fake_download(url: str, dataset_id: str) -> Path:
        requested_urls.append(url)
        if len(requested_urls) == 1:
            raise FALLBACK_EXCEPTIONS[0](url)
        return Path(url)

    monkeypatch.setattr(
        "reformatters.eccc.hrdps.forecast.region_job.http_download_to_disk",
        fake_download,
    )

    region_job = EcccHrdpsForecastRegionJob.model_construct(
        tmp_store=Path("/tmp/not-used"),  # noqa: S108
        template_ds=TEMPLATE_CONFIG.get_template(pd.Timestamp("2026-07-09T06:00")),
        data_vars=TEMPLATE_CONFIG.data_vars,
        append_dim=TEMPLATE_CONFIG.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )
    region_job.download_file(make_coord(init_time=init_time))

    assert expected_first_source in requested_urls[0]
    assert expected_first_source not in requested_urls[1]


def make_data_array(variable_name: str, values: list[float]) -> xr.DataArray:
    data_var = DATA_VARS_BY_NAME[variable_name]
    return xr.DataArray(
        np.array(values, dtype=np.float32).reshape(len(values), 1, 1),
        dims=("lead_time", "y", "x"),
        coords={"lead_time": pd.timedelta_range("0h", periods=len(values), freq="1h")},
        attrs={"units": data_var.attrs.units},
    )


def apply_transformations(variable_name: str, values: list[float]) -> list[float]:
    region_job = EcccHrdpsForecastRegionJob.model_construct(
        tmp_store=Path("/tmp/not-used"),  # noqa: S108
        template_ds=TEMPLATE_CONFIG.get_template(pd.Timestamp("2026-07-09T06:00")),
        data_vars=TEMPLATE_CONFIG.data_vars,
        append_dim=TEMPLATE_CONFIG.append_dim,
        region=slice(0, 1),
        reformat_job_name="test",
    )
    data_array = make_data_array(variable_name, values)
    region_job.apply_data_transformations(data_array, DATA_VARS_BY_NAME[variable_name])
    return [float(v) for v in data_array.values.ravel()]


def test_hourly_accumulation_becomes_a_mean_rate() -> None:
    # Every lead time resets, so each hourly bucket becomes its own mean rate.
    assert apply_transformations("precipitation_surface", [np.nan, 1800.0, 900.0]) == [
        pytest.approx(np.nan, nan_ok=True),
        0.5,
        0.25,
    ]


def test_run_total_is_differenced_into_a_mean_rate() -> None:
    # A run total accumulating 100 W m-2 then 200 W m-2 over consecutive hours.
    assert apply_transformations(
        "downward_short_wave_radiation_flux_surface", [0.0, 360_000.0, 1_080_000.0]
    ) == [pytest.approx(np.nan, nan_ok=True), 100.0, 200.0]


def test_snow_water_equivalent_converts_to_metres() -> None:
    assert apply_transformations(
        "snow_water_equivalent_surface", [1000.0, 500.0, 250.0]
    ) == [1.0, 0.5, 0.25]


# A run total rising 1000 W m-2 an hour, dipping 10 W m-2 once. The dip is a negative
# step rate above the invalid threshold, so it clamps to 0: one of ten values, 10%.
RUN_TOTAL_WITH_ONE_CLAMPED_STEP = [
    0.0,
    3_600_000.0,
    7_200_000.0,
    10_800_000.0,
    14_400_000.0,
    14_364_000.0,
    21_600_000.0,
    25_200_000.0,
    28_800_000.0,
    32_400_000.0,
]


def deaccumulation_errors(
    variable_name: str, values: list[float], caplog: pytest.LogCaptureFixture
) -> list[str]:
    with caplog.at_level(logging.ERROR):
        transformed = apply_transformations(variable_name, values)
    assert transformed[5] == 0.0  # the dip clamped rather than going negative
    return [record.message for record in caplog.records]


def test_short_wave_allows_the_clamping_dark_hours_produce(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Short wave is flat overnight, where precision jitter clamps far more than the 5%
    # default allows, so its allowance is raised and 10% must pass quietly.
    assert (
        deaccumulation_errors(
            "downward_short_wave_radiation_flux_surface",
            RUN_TOTAL_WITH_ONE_CLAMPED_STEP,
            caplog,
        )
        == []
    )


def test_long_wave_keeps_the_default_clamp_allowance(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Long wave accumulates day and night, so the same 10% is unexpected and reported.
    assert any(
        "Error deaccumulating downward_long_wave_radiation_flux_surface" in message
        for message in deaccumulation_errors(
            "downward_long_wave_radiation_flux_surface",
            RUN_TOTAL_WITH_ONE_CLAMPED_STEP,
            caplog,
        )
    )
