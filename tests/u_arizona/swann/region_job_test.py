import pandas as pd
import pytest
import xarray as xr

from reformatters.u_arizona.swann.region_job import (
    SWANNRegionJob,
    SWANNSourceFileCoord,
)

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    "time,expected_water_year,expected_url",
    [
        (
            pd.Timestamp("2023-09-30"),  # End of water year
            2023,
            "https://climate.arizona.edu/data/UA_SWE/DailyData_4km/WY2023/UA_SWE_Depth_4km_v1_20230930_stable.nc",
        ),
        (
            pd.Timestamp("2023-10-01"),  # Start of water year
            2024,
            "https://climate.arizona.edu/data/UA_SWE/DailyData_4km/WY2024/UA_SWE_Depth_4km_v1_20231001_stable.nc",
        ),
    ],
)
def test_source_file_coord_url_generation(
    time: pd.Timestamp,
    expected_water_year: int,
    expected_url: str,
) -> None:
    """Test URL generation and water year calculation for different times."""
    coord = SWANNSourceFileCoord(time=time)
    assert coord.get_water_year() == expected_water_year
    assert coord.get_url() == expected_url


def test_source_file_coord_data_status() -> None:
    """Test data status handling and advancement."""
    time = pd.Timestamp("2023-10-01")
    coord = SWANNSourceFileCoord(time=time)

    # Test initial status
    assert coord.get_data_status() == "stable"
    assert (
        coord.get_url()
        == "https://climate.arizona.edu/data/UA_SWE/DailyData_4km/WY2024/UA_SWE_Depth_4km_v1_20231001_stable.nc"
    )

    # Test advancing status
    assert coord.advance_data_status() is True
    assert coord.get_data_status() == "provisional"
    assert (
        coord.get_url()
        == "https://climate.arizona.edu/data/UA_SWE/DailyData_4km/WY2024/UA_SWE_Depth_4km_v1_20231001_provisional.nc"
    )

    assert coord.advance_data_status() is True
    assert coord.get_data_status() == "early"
    assert (
        coord.get_url()
        == "https://climate.arizona.edu/data/UA_SWE/DailyData_4km/WY2024/UA_SWE_Depth_4km_v1_20231001_early.nc"
    )

    # Test final status advancement
    assert coord.advance_data_status() is False
    # Should still return the last attempted status URL
    assert (
        coord.get_url()
        == "https://climate.arizona.edu/data/UA_SWE/DailyData_4km/WY2024/UA_SWE_Depth_4km_v1_20231001_early.nc"
    )


def test_update_append_dim_start() -> None:
    """Test the update_append_dim_start method."""
    existing_ds = xr.Dataset(coords={"time": pd.date_range("2023-10-01", "2025-01-03")})
    expected_start = pd.Timestamp("2024-01-04")
    assert SWANNRegionJob._update_append_dim_start(existing_ds) == expected_start
