from reformatters.common.config_models import BaseInternalAttrs, DataVar


class EcccHrdpsInternalAttrs(BaseInternalAttrs):
    # Variable and level as they appear in the source filename, e.g. "TMP_AGL-2m"
    variable_name_in_filename: str
    grib_element: str
    # HRDPS accumulation GRIB_ELEMENTs carry the window length as a suffix (APCP01...APCP48)
    include_lead_time_suffix: bool = False
    # Multiply raw values by this factor after reading (e.g. 0.001 to convert kg m-2 to m)
    scale_factor: float | None = None
    deaccumulation_invalid_below_threshold_rate: float | None = None
    deaccumulation_expected_clamp_fraction: float | None = None


class EcccHrdpsDataVar(DataVar[EcccHrdpsInternalAttrs]):
    pass
