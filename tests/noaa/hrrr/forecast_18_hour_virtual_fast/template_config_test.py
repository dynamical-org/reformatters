import pandas as pd

from reformatters.noaa.hrrr.forecast_18_hour_virtual.template_config import (
    NoaaHrrrForecast18HourVirtualTemplateConfig,
)
from reformatters.noaa.hrrr.forecast_18_hour_virtual_fast.template_config import (
    NoaaHrrrForecast18HourVirtualFastTemplateConfig,
)

CONFIG = NoaaHrrrForecast18HourVirtualFastTemplateConfig()


def test_dataset_attributes() -> None:
    attrs = CONFIG.dataset_attributes
    assert attrs.dataset_id == "noaa-hrrr-forecast-18-hour-virtual-fast"
    assert attrs.dataset_version == "0.1.0"
    assert attrs.name == "NOAA HRRR forecast, 18 hour, fast, virtual"
    assert "NOMADS" in attrs.description
    sibling = NoaaHrrrForecast18HourVirtualTemplateConfig().dataset_attributes
    assert attrs.description.startswith(sibling.description)


def test_variables_and_structure_match_the_18_hour_dataset() -> None:
    sibling = NoaaHrrrForecast18HourVirtualTemplateConfig()
    assert len(CONFIG.data_vars) == 176
    assert list(CONFIG.data_vars) == list(sibling.data_vars)
    assert CONFIG.dims == sibling.dims
    assert CONFIG.append_dim_frequency == sibling.append_dim_frequency


def test_template_starts_at_the_mirror_start() -> None:
    assert CONFIG.append_dim_start == pd.Timestamp("2026-09-01T00:00")
    template = CONFIG.get_template(pd.Timestamp("2026-09-01T02:00"))
    init_times = template.to_dataset().get_index("init_time")
    assert list(init_times) == [
        pd.Timestamp("2026-09-01T00:00"),
        pd.Timestamp("2026-09-01T01:00"),
    ]
