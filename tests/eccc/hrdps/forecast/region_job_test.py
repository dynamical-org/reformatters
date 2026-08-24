from pathlib import Path

import pandas as pd
import pytest

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
