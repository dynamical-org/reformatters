"""
Tests for quality flag processing in NOAA NDVI CDR data.
"""

import numpy as np
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
    get_avhrr_mask,
    get_viirs_mask,
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
        (
            AVHRR_BRDF_CORR_PROBLEM,
            False,
        ),  # Masking BRDF_CORR_PROBLEM  is too aggressive and we get very little data
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
    result = get_avhrr_mask(qa_array)
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

    result = get_viirs_mask(qa_array)
    assert result[0] == expected_mask_value
