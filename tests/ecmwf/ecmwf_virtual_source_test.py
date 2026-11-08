import obstore.store
import pandas as pd
import pytest

from reformatters.ecmwf.aifs_single.forecast_virtual.region_job import (
    EcmwfAifsSingleForecastVirtualSourceFileCoord,
)
from reformatters.ecmwf.ecmwf_virtual_source import discover_ecmwf_available


@pytest.mark.parametrize(
    ("cutoff", "gcs_present", "s3_present", "expected_calls", "source"),
    [
        ("2025-03-01", True, True, ["gs://ecmwf-open-data/"], "gcs"),
        (
            "2025-03-01",
            False,
            True,
            ["gs://ecmwf-open-data/", "s3://ecmwf-forecasts/"],
            "s3",
        ),
        (
            "2025-03-01",
            False,
            False,
            ["gs://ecmwf-open-data/", "s3://ecmwf-forecasts/"],
            "gcs",
        ),
        ("2025-02-20", False, True, ["gs://ecmwf-open-data/"], "gcs"),
        ("2025-02-13", False, True, ["gs://ecmwf-open-data/"], "gcs"),
        (None, False, True, ["gs://ecmwf-open-data/"], "gcs"),
    ],
)
def test_explicit_caller_cutoff_preserves_aifs_and_ifs_policies(
    cutoff: str | None,
    gcs_present: bool,
    s3_present: bool,
    expected_calls: list[str],
    source: str,
) -> None:
    coord = EcmwfAifsSingleForecastVirtualSourceFileCoord(
        init_time=pd.Timestamp("2025-02-20"),
        lead_time=pd.Timedelta("24h"),
        data_vars=[],
    )
    calls = []

    def listing(
        pending: list[EcmwfAifsSingleForecastVirtualSourceFileCoord],
        *,
        store: obstore.store.ObjectStore,
        location_prefix: str,
        require_index: bool,
    ) -> list[tuple[EcmwfAifsSingleForecastVirtualSourceFileCoord, int]]:
        assert require_index
        calls.append(location_prefix)
        available = gcs_present if location_prefix.startswith("gs") else s3_present
        return [(p, 100) for p in pending] if available else []

    found = discover_ecmwf_available(
        [coord],
        s3_fallback_before=pd.Timestamp(cutoff) if cutoff else None,
        listing=listing,
    )
    assert calls == expected_calls
    assert coord.source == source
    if found:
        assert found[0][0] is coord
    else:
        assert coord.source == "gcs"
