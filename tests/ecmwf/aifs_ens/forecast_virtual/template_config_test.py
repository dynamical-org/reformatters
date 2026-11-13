from reformatters.ecmwf.aifs_single.forecast_virtual.template_config import (
    EcmwfAifsSingleForecastVirtualTemplateConfig,
)
from tests.ecmwf.aifs_ens.forecast_virtual.conftest import CONFIG


def test_aifs_siblings_do_not_drift_in_variable_meaning() -> None:
    sibling = {
        var.path: var
        for var in EcmwfAifsSingleForecastVirtualTemplateConfig().data_vars
    }
    assert {var.path for var in CONFIG.data_vars} == sibling.keys()
    for var in CONFIG.data_vars:
        assert var.encoding.filters == sibling[var.path].encoding.filters
        for attr in ("long_name", "short_name", "units", "standard_name", "step_type"):
            assert getattr(var.attrs, attr) == getattr(sibling[var.path].attrs, attr), (
                var.path,
                attr,
            )
