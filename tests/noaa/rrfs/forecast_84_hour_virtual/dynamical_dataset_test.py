from pathlib import Path

import pytest

from reformatters.noaa.rrfs.forecast_84_hour_virtual.dynamical_dataset import (
    NoaaRrfsForecast84HourVirtualDataset,
)
from tests.noaa.rrfs.integration_helpers import backfill_and_update_in_isolated_store


@pytest.mark.slow
def test_real_source_backfill_and_update_in_isolated_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backfill_and_update_in_isolated_store(
        NoaaRrfsForecast84HourVirtualDataset, tmp_path, monkeypatch
    )
