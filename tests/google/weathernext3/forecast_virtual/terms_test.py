import pytest

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.google.weathernext3.forecast_virtual.dynamical_dataset import (
    GoogleWeathernext3ForecastVirtualDataset,
)
from reformatters.google.weathernext_terms import (
    COPYRIGHT,
    LEGALLY_BINDING_NOTICE,
    REAL_TIME_ATTRIBUTION,
    notice_revision,
    render_notice_files,
)

DATASETS = [
    dataset
    for dataset in DYNAMICAL_DATASETS
    if isinstance(dataset, GoogleWeathernext3ForecastVirtualDataset)
]
REVISIONS = {
    "google-weathernext3-forecast-15-day-0-1-degree-virtual": "1fb4c26066b6",
    "google-weathernext3-forecast-48-hour-0-1-degree-virtual": "320b61855ccb",
    "google-weathernext3-forecast-15-day-0-05-degree-virtual": "f5a6054ea3e1",
    "google-weathernext3-forecast-48-hour-0-05-degree-virtual": "6c106f17e52f",
}


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda dataset: dataset.dataset_id)
def test_wn3_notices_match_the_product_grid_and_public_repository(
    dataset: GoogleWeathernext3ForecastVirtualDataset,
) -> None:
    config = dataset.template_config
    attributes = config.dataset_attributes
    assert attributes.license == "other"
    assert attributes.copyright == COPYRIGHT
    assert attributes.terms_of_use == LEGALLY_BINDING_NOTICE
    assert attributes.modifications is not None
    assert attributes.modifications.startswith("The publisher, dynamical.org, ")
    real_time, historical = attributes.attribution.split(" Historical Data: ")
    assert real_time == (
        f"Real-Time Data & Forecasts: Google WeatherNext 3, {REAL_TIME_ATTRIBUTION}"
    )
    expected_grid = "0p1deg" if config.grid_degrees == 0.1 else "0p05deg"
    assert historical.startswith("© 2026 DeepMind Technologies Limited's ")
    assert (
        "https://developers.google.com/earth-engine/datasets/catalog/"
        f"projects_gcp-public-data-weathernext_assets_weathernext_3_0_0_{expected_grid} "
        "under CC BY 4.0 licence terms."
    ) in historical
    assert historical.endswith(
        "Your use of the third-party materials is subject to any such terms and "
        "you should check that you can comply with any applicable restrictions "
        "or terms and conditions before use."
    )

    files = render_notice_files(attributes, product=config.product)
    revision = notice_revision(files)
    assert revision == REVISIONS[dataset.dataset_id]
    assert (
        b"from Google WeatherNext 3 experimental forecast data." in files["NOTICE.txt"]
    )
    assert dataset.repo_files() == {
        f"notices/{revision}/{name}": data for name, data in files.items()
    }
    public_repo = (
        f"https://google-weathernext3.r2.dynamical.org/{dataset.dataset_id}"
        f"/v{attributes.dataset_version}.icechunk"
    )
    assert dataset.store_factory.primary_url(public=True) == public_repo
    assert attributes.notice_files == tuple(
        f"{public_repo}/{path}" for path in dataset.repo_files()
    )
