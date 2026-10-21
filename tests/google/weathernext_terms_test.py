import hashlib
import json

import pytest

from reformatters.common.config import Config, Env
from reformatters.common.storage import StoreFactory
from reformatters.google.weathernext2.forecast_historical_virtual.template_config import (
    GoogleWeathernext2ForecastHistoricalVirtualTemplateConfig,
)
from reformatters.google.weathernext2.forecast_operational_virtual.dynamical_dataset import (
    GoogleWeathernext2ForecastOperationalVirtualDataset,
)
from reformatters.google.weathernext2.forecast_operational_virtual.template_config import (
    GoogleWeathernext2ForecastOperationalVirtualTemplateConfig,
)
from reformatters.google.weathernext2.storage import (
    Weathernext2IcechunkDatasetStorageConfig,
)
from reformatters.google.weathernext_terms import (
    COPYRIGHT,
    LEGALLY_BINDING_NOTICE,
    REAL_TIME_ATTRIBUTION,
    TERMS_SHA256,
    TERMS_URL,
    notice_revision,
    render_notice_files,
    repo_notice_files,
)


def test_terms_constants_and_pdf_are_pinned() -> None:
    assert (
        TERMS_URL
        == "https://storage.googleapis.com/weathernext-public/terms-of-use.pdf"
    )
    assert (
        TERMS_SHA256
        == "5fe155ace6bb11737f15bc735913c5739d31d27374269a8cbfa5f338c4d40654"
    )
    assert COPYRIGHT == "Copyright 2024-6 Google LLC"
    assert (
        LEGALLY_BINDING_NOTICE
        == "By using this information, you agree to the Terms of Use found at https://storage.googleapis.com/weathernext-public/terms-of-use.pdf"
    )
    assert (
        REAL_TIME_ATTRIBUTION
        == "© 2024-6 Google LLC, whose machine learning models were used to create the experimental data made available under the following licence terms https://storage.googleapis.com/weathernext-public/terms-of-use.pdf. This data is intended for experimental modelling only and is not intended, validated, or approved for real world use."
    )
    files = render_notice_files(
        GoogleWeathernext2ForecastOperationalVirtualTemplateConfig().dataset_attributes,
        product="Google WeatherNext 2",
    )
    assert hashlib.sha256(files["TERMS_OF_USE.pdf"]).hexdigest() == TERMS_SHA256


def test_notice_rendering_and_revision() -> None:
    attributes = (
        GoogleWeathernext2ForecastOperationalVirtualTemplateConfig().dataset_attributes
    )
    files = render_notice_files(attributes, product="Google WeatherNext 2")
    assert {name: hashlib.sha256(data).hexdigest() for name, data in files.items()} == {
        "TERMS_OF_USE.pdf": "5fe155ace6bb11737f15bc735913c5739d31d27374269a8cbfa5f338c4d40654",
        "LEGALLY_BINDING_TERMS_OF_USE.txt": "35a0deb0a5a3c4436a7fffe6e7253baa4c43ec41804a449600e44b233d1b443d",
        "NOTICE.txt": "93129d6352e91f6a0210d277eb3d51a30ff5211b74d41b0589a910e795374970",
    }
    assert set(files) == {
        "TERMS_OF_USE.pdf",
        "LEGALLY_BINDING_TERMS_OF_USE.txt",
        "NOTICE.txt",
    }
    assert files["LEGALLY_BINDING_TERMS_OF_USE.txt"] == (
        LEGALLY_BINDING_NOTICE + "\n"
    ).encode("utf-8")
    expected_paragraphs = [
        COPYRIGHT,
        "Google WeatherNext 2 operational forecast, virtual (google-weathernext2-forecast-operational-virtual) is published by dynamical.org from Google WeatherNext 2 experimental forecast data.",
        "This data is licensed under the GDM Real-Time Weather Forecasting Experimental Data Terms of Use and CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), as set out in those Terms. TERMS_OF_USE.pdf beside this file is a copy of the Terms, and LEGALLY_BINDING_TERMS_OF_USE.txt carries Google's notice.",
        LEGALLY_BINDING_NOTICE,
        f"Modifications: {attributes.modifications}",
        f"Attribution: {attributes.attribution}",
    ]
    assert files["NOTICE.txt"] == ("\n\n".join(expected_paragraphs) + "\n").encode(
        "utf-8"
    )
    expected_revision = hashlib.sha256(
        b"LEGALLY_BINDING_TERMS_OF_USE.txt\0"
        + files["LEGALLY_BINDING_TERMS_OF_USE.txt"]
        + b"\0NOTICE.txt\0"
        + files["NOTICE.txt"]
        + b"\0TERMS_OF_USE.pdf\0"
        + files["TERMS_OF_USE.pdf"]
        + b"\0"
    ).hexdigest()[:12]
    assert notice_revision(files) == expected_revision
    assert expected_revision == "3063f11e3c58"
    assert notice_revision(dict(reversed(list(files.items())))) == expected_revision
    assert notice_revision(files | {"NOTICE.txt": b"Changed\n"}) != expected_revision
    assert repo_notice_files(attributes, product="Google WeatherNext 2") == {
        f"notices/{expected_revision}/{name}": data for name, data in files.items()
    }


def test_notice_product_is_supplied_by_caller() -> None:
    attributes = (
        GoogleWeathernext2ForecastOperationalVirtualTemplateConfig().dataset_attributes
    )
    wn2 = render_notice_files(attributes, product="Google WeatherNext 2")
    wn3 = render_notice_files(attributes, product="Google WeatherNext 3")
    assert wn3 == wn2 | {
        "NOTICE.txt": wn2["NOTICE.txt"].replace(
            b"from Google WeatherNext 2 experimental forecast data.",
            b"from Google WeatherNext 3 experimental forecast data.",
            1,
        )
    }
    revision = notice_revision(wn3)
    assert repo_notice_files(attributes, product="Google WeatherNext 3") == {
        f"notices/{revision}/{name}": data for name, data in wn3.items()
    }


@pytest.mark.parametrize("env", [Env.dev, Env.test, Env.prod])
def test_operational_notice_urls_use_canonical_public_storage(
    monkeypatch: pytest.MonkeyPatch, env: Env
) -> None:
    monkeypatch.setattr(Config, "env", env)
    dataset = GoogleWeathernext2ForecastOperationalVirtualDataset(
        primary_storage_config=Weathernext2IcechunkDatasetStorageConfig()
    )
    attrs = dataset.template_config.dataset_attributes
    assert attrs.license == "other"
    assert attrs.copyright == COPYRIGHT
    assert attrs.terms_of_use == LEGALLY_BINDING_NOTICE
    assert attrs.notice_files == tuple(
        f"{dataset.store_factory.primary_url(public=True)}/{name}"
        for name in dataset.repo_files()
    )
    factory = StoreFactory(
        primary_storage_config=Weathernext2IcechunkDatasetStorageConfig(),
        dataset_id=attrs.dataset_id,
        template_config_version=attrs.dataset_version,
    )
    assert factory.primary_url(public=True) == (
        "https://google-weathernext2.r2.dynamical.org/"
        "google-weathernext2-forecast-operational-virtual/v0.1.0.icechunk"
    )


def test_historical_license_and_attribution_are_unchanged() -> None:
    historical = (
        GoogleWeathernext2ForecastHistoricalVirtualTemplateConfig().dataset_attributes
    )
    operational = (
        GoogleWeathernext2ForecastOperationalVirtualTemplateConfig().dataset_attributes
    )
    assert historical.license == "CC-BY-4.0"
    assert historical.copyright is None
    assert historical.terms_of_use is None
    assert historical.notice_files is None
    historic_attribution = historical.attribution.removeprefix(
        "Google requires this attribution: "
    )
    assert operational.attribution == (
        f"Real-Time Data & Forecasts: Google WeatherNext 2, {REAL_TIME_ATTRIBUTION} "
        "Historical Data: " + historic_attribution
    )
    for config in (
        GoogleWeathernext2ForecastHistoricalVirtualTemplateConfig(),
        GoogleWeathernext2ForecastOperationalVirtualTemplateConfig(),
    ):
        attrs = config.dataset_attributes
        assert attrs.modifications is not None
        assert "temperatures from kelvin to degrees Celsius" in attrs.modifications
        assert "dividing by 9.80665 m s-2" in attrs.modifications
        assert (
            "6-hour total precipitation from metres of water equivalent to kg m-2"
            in attrs.modifications
        )
        assert ("moves the pressure-level axis" in attrs.modifications) == (
            config.source_layout == "historical"
        )
        for group in ("", "pressure_level"):
            path = config.template_path() / group / "zarr.json"
            assert json.loads(path.read_text())["attributes"] == attrs.model_dump(
                mode="json", exclude_none=True
            )
