from reformatters.common.config_models import INTERNAL_ATTRS, DataVar


def has_hour_0_values(data_var: DataVar[INTERNAL_ATTRS]) -> bool:
    return data_var.attrs.step_type == "instant"
