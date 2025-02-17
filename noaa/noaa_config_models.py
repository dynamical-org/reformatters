from common.config_models import FrozenBaseModel


class NOAAInternalAttrs(FrozenBaseModel):
    grib_element: str
    grib_description: str
    grib_index_level: str
    index_position: int
    include_lead_time_suffix: bool = False
