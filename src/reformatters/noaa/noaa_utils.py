from reformatters.common.config_models import DataVar, INTERNAL_ATTRS_co


def has_hour_0_values(data_var: DataVar[INTERNAL_ATTRS_co]) -> bool:
    return data_var.attrs.step_type == "instant"
