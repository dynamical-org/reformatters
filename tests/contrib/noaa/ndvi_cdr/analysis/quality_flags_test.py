"""
Tests for quality flag processing in NOAA NDVI CDR data.
"""

import numpy as np
import pandas as pd
import pytest

from reformatters.contrib.noaa.ndvi_cdr.ndvi_cdr.analysis.quality_flags import (
    AVHRR_BRDF_CORR_PROBLEM,
    AVHRR_CH1_INVALID,
    AVHRR_CLOUD_SHADOW,
    AVHRR_CLOUDY,
    AVHRR_WATER,
    FILL_VALUE,
    VIIRS_AEROSOL_QUALITY_OK,
    VIIRS_CLOUD_SHADOW,
    VIIRS_CLOUD_STATE_CLOUDY,
    VIIRS_COASTAL,
    VIIRS_INLAND_WATER,
    VIIRS_LAND_NO_DESERT,
    VIIRS_SEA_WATER,
    _get_avhrr_mask,
    _get_viirs_mask,
    apply_quality_mask,
)


@pytest.mark.parametrize(
    "qa_value,expected_mask_value",
    [
        (FILL_VALUE, True),
        (0, False),
        (AVHRR_CLOUDY, True),
        (AVHRR_CLOUD_SHADOW, True),
        (AVHRR_WATER, True),
        (AVHRR_CH1_INVALID, True),
        (AVHRR_BRDF_CORR_PROBLEM, True),
        (-32768, False),  # AVHRR_POLAR_FLAG as it appears in int16
    ],
    ids=[
        "fill-value",
        "clear",
        "cloudy",
        "cloud-shadow",
        "water",
        "channel-one-invalid",
        "brdf-correction-problem",
        "polar-flag",
    ],
)
def test_avhrr_quality_flags(qa_value: int, expected_mask_value: bool) -> None:
    """Test AVHRR quality flag processing."""
    qa_array = np.array([qa_value], dtype=np.int16)
    result = _get_avhrr_mask(qa_array)
    assert result[0] == expected_mask_value


@pytest.mark.parametrize(
    "qa_value,expected_mask_value",
    [
        (FILL_VALUE, True),
        (VIIRS_INLAND_WATER, True),
        (VIIRS_SEA_WATER, True),
        (VIIRS_COASTAL, True),
        (VIIRS_INLAND_WATER + VIIRS_AEROSOL_QUALITY_OK, True),
        (VIIRS_CLOUD_STATE_CLOUDY, True),
        (VIIRS_CLOUD_STATE_CLOUDY + VIIRS_AEROSOL_QUALITY_OK, True),
        (VIIRS_CLOUD_SHADOW, True),
        (VIIRS_CLOUD_SHADOW + VIIRS_AEROSOL_QUALITY_OK, True),
        (0, True),
        (VIIRS_AEROSOL_QUALITY_OK, False),
        (VIIRS_LAND_NO_DESERT, True),
        (VIIRS_LAND_NO_DESERT + VIIRS_AEROSOL_QUALITY_OK, False),
    ],
    ids=[
        "fill-value",
        "inland-water",
        "sea-water",
        "coastal",
        "water-with-aerosol-quality",
        "probably-cloudy",
        "cloudy-with-aerosol-quality",
        "cloud-shadow",
        "cloud-shadow-with-aerosol-quality",
        "clear-no-aerosol-quality",
        "clear-with-aerosol-quality",
        "land-no-desert-no-aerosol-quality",
        "land-no-desert-with-aerosol-quality",
    ],
)
def test_viirs_quality_flags(qa_value: int, expected_mask_value: bool) -> None:
    """Test VIIRS quality flag processing."""
    qa_array = np.array([qa_value], dtype=np.int16)

    result = _get_viirs_mask(qa_array)
    assert result[0] == expected_mask_value


def test_apply_quality_mask() -> None:
    """Test that correct algorithm is selected based on timestamp."""
    ndvi_array = np.array([0.5, 0.6, 0.7], dtype=np.float32)

    # Use QA values that mean different things in each system
    qa_array = np.array(
        [
            64,  # AVHRR: night flag (bad), VIIRS: aerosol_quality_ok (good)
            2,  # Cloudy in both systems (bad)
            72,  # AVHRR: water+night (bad), VIIRS: land_no_desert+aerosol (good)
        ],
        dtype=np.int16,
    )

    # AVHRR: 64=night (bad), 2=cloudy (bad), 72=water+night (bad)
    result_avhrr = apply_quality_mask(
        ndvi_array.copy(), qa_array, pd.Timestamp("2013-12-31")
    )
    assert not np.isnan(result_avhrr[0])  # night preserved
    assert np.isnan(result_avhrr[1])  # cloudy masked
    assert np.isnan(result_avhrr[2])  # water+night masked

    # VIIRS: 64=aerosol_quality_ok (good), 2=probably_cloudy (bad), 72=land_no_desert+aerosol (good)
    result_viirs = apply_quality_mask(
        ndvi_array.copy(), qa_array, pd.Timestamp("2014-01-01")
    )
    assert not np.isnan(result_viirs[0])  # aerosol quality preserved
    assert np.isnan(result_viirs[1])  # cloudy masked
    assert not np.isnan(result_viirs[2])  # land_no_desert+aerosol preserved
