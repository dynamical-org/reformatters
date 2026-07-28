from pydantic import Field

from reformatters.common.storage import (
    IcechunkVirtualConfig,
    manifest_append_dim_split,
)
from reformatters.noaa.hrrr.forecast_48_hour_virtual.dynamical_dataset import (
    NoaaHrrrForecast48HourVirtualDataset,
)
from reformatters.noaa.hrrr.forecast_virtual_region_job import (
    hrrr_virtual_chunk_containers,
)

from .template_config import NoaaHrrrForecast48HourVirtualFastTemplateConfig


class NoaaHrrrForecast48HourVirtualFastDataset(NoaaHrrrForecast48HourVirtualDataset):
    """NOAA HRRR 48-hour virtual forecast trimmed to the materialized dataset's
    variable set. Operational timing is inherited from the full virtual dataset so the
    two products' ingest latency differs only by variable set."""

    template_config: NoaaHrrrForecast48HourVirtualFastTemplateConfig = (
        NoaaHrrrForecast48HourVirtualFastTemplateConfig()
    )

    icechunk_virtual_config: IcechunkVirtualConfig = Field(
        default_factory=lambda: IcechunkVirtualConfig(
            containers=hrrr_virtual_chunk_containers(),
            # Root-only, so one split size: 600 inits x 49 refs at ~16.4 bytes/ref
            # is ~0.5 MiB of active manifest per array, matching the full virtual
            # dataset's root arrays. See "Manifest splitting" in docs/virtual_datasets.md.
            manifest_split=manifest_append_dim_split(split_size=600, dim="init_time"),
        )
    )
