"""
Quality flag processing for NOAA NDVI CDR data.

Implements quality filtering for AVHRR (1981-2013) and VIIRS (2014-present) instruments.
Quality flags are stored as int16 but interpreted as unsigned bit patterns.
Algorithm selection is timestamp-based with transition at 2014-01-01.

AVHRR gdalinfo output:
NoData Value=-32767
flag_masks={2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,-32768}
flag_meanings=cloudy cloud_shadow water sunglint dense_dark_vegetation night ch1_to_5_valid ch1_invalid ch2_invalid ch3_invalid ch4_invalid ch5_invalid rho3_invalid BRDF_corr_problem polar_flag

VIIRS gdalinfo output:
NoData Value=-32767
flag_masks={32768,32768,1024,1024,512,512,256,256,192,192,192,192,56,56,56,56,56,4,4,3,3,3,3}
flag_meanings=snow-ice_flag_snow-ice snow-ice_flag_no_snow-ice cloud_flag_cloud cloud_flag_no_cloud thin_cirrus_emissive_yes thin_cirrus_emissive_no thin_cirrus_reflective_yes thin_cirrus_reflective_no aerosol_quantity:_level_of_uncertainty_in_aerosol_correction_climatology aerosol_quantity:_level_of_uncertainty_in_aerosol_correction_low aerosol_quantity:_level_of_uncertainty_in_aerosol_correction_average aerosol_quantity:_level_of_uncertainty_in_aerosol_correction_high land-water_flag_land_and_desert land-water_flag_land_no_desert land-water_flag_inland_water land-water_flag_sea_water land-water_flag_coastal cloud_shadow_yes cloud_shadow_no cloud_state_confident_clear cloud_state_probably_clear cloud_state_probably_cloudy cloud_state_confident_cloudy
flag_values={32768,0,1024,0,512,0,256,0,0,64,128,192,0,8,16,24,40,4,0,0,1,2,3}

Docs (see section 5.3 in the AVHRR doc and section 4.3 in the VIIRS doc for QA flag details):
- AVHRR: https://www.ncei.noaa.gov/pub/data/sds/cdr/CDRs/Normalized_Difference_Vegetation_Index/AVHRR/AlgorithmDescriptionAVHRR_01B-20b.pdf
- VIIRS: https://www.ncei.noaa.gov/pub/data/sds/cdr/CDRs/Normalized_Difference_Vegetation_Index/VIIRS/AlgorithmDescriptionVIIRS_01B-20b.pdf
"""

import numpy as np

from reformatters.common.types import Array2D

from .template_config import QA_FILL_VALUE

# VIIRS quality flag bit positions
VIIRS_CLOUD_STATE_CLOUDY = 1 << 1  # Probably or confident cloudy
VIIRS_CLOUD_SHADOW = 1 << 2
VIIRS_LAND_WATER_BIT_3 = 1 << 3
VIIRS_LAND_WATER_BIT_4 = 1 << 4
VIIRS_LAND_WATER_BIT_5 = 1 << 5
VIIRS_SEA_WATER = (1 << 3) | (1 << 4)  # Bits 3+4
VIIRS_COASTAL = (1 << 3) | (1 << 5)  # Bits 3+5
VIIRS_AEROSOL_QUALITY_OK = 1 << 6

# AVHRR quality flag bit positions
AVHRR_CLOUDY = 1 << 1
AVHRR_CLOUD_SHADOW = 1 << 2
AVHRR_WATER = 1 << 3
AVHRR_SUNGLINT = 1 << 4
AVHRR_CH1_INVALID = 1 << 8
AVHRR_CH2_INVALID = 1 << 9
AVHRR_CH3_INVALID = 1 << 10
AVHRR_CH4_INVALID = 1 << 11
AVHRR_CH5_INVALID = 1 << 12
AVHRR_RHO3_INVALID = 1 << 13
AVHRR_BRDF_CORR_PROBLEM = 1 << 14
AVHRR_POLAR_FLAG = np.int16(1) << 15


def get_avhrr_mask(qa_array: Array2D[np.int16]) -> Array2D[np.bool_]:
    """
    Generate "bad" quality mask for AVHRR data that is true where values should be ignored.

    The flags we use and don't use here come from a practical balance of maintaining
    enough decent quality data while removing values that clearly disrupt the NDVI signal.

    Masks pixels with cloud, shadow, water, sunglint, or invalid channels.
    Polar flag is NOT considered bad quality.
    """
    # Identify fill values before conversion
    is_fill = qa_array == np.int16(QA_FILL_VALUE)

    # Build up int16 mask of bits that indicate a bad value
    bad_mask = np.array(
        AVHRR_CLOUDY
        | AVHRR_CLOUD_SHADOW
        | AVHRR_WATER
        | AVHRR_SUNGLINT
        | AVHRR_CH1_INVALID
        | AVHRR_CH2_INVALID
        | AVHRR_CH3_INVALID
        | AVHRR_CH4_INVALID
        | AVHRR_CH5_INVALID,
        dtype=np.int16,
    )

    # Bitwise AND the qa_array with the bad_mask. If any bit is set,
    # we should ignore the pixel's value.
    bad_quality = (qa_array & bad_mask).astype(bool)

    # Fill values are always bad quality
    return bad_quality | is_fill  # type: ignore[no-any-return]


def get_viirs_mask(qa_array: Array2D[np.int16]) -> Array2D[np.bool_]:
    """Generate bad quality mask for VIIRS data.

    Masks pixels with cloud, shadow, water, or insufficient aerosol quality.
    Requires aerosol quality bit to be set (1) for good quality.
    """
    # Identify fill values before conversion
    is_fill = qa_array == np.int16(QA_FILL_VALUE)

    bad_mask = np.array(
        VIIRS_CLOUD_STATE_CLOUDY
        | VIIRS_CLOUD_SHADOW
        | VIIRS_LAND_WATER_BIT_4
        | VIIRS_LAND_WATER_BIT_5
        # Aerosol quality (we actually want this one set, 1 means OK)
        | VIIRS_AEROSOL_QUALITY_OK,
        dtype=np.int16,
    )

    # The Aerosol bit is 1 if the quality is OK, all
    # other mask bits should be unset
    bad_quality = (qa_array & bad_mask) != VIIRS_AEROSOL_QUALITY_OK

    # Fill values are always bad quality
    return bad_quality | is_fill  # type: ignore[no-any-return]
