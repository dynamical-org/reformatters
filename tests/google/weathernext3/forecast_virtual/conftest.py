import pandas as pd
import pytest

from reformatters.google.weathernext3.forecast_virtual.region_job import (
    GoogleWeathernext3ForecastVirtualRegionJob,
)


@pytest.fixture
def delayed_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        GoogleWeathernext3ForecastVirtualRegionJob,
        "publication_holdback",
        pd.Timedelta("1h"),
    )
