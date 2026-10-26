from pathlib import Path

import icechunk
import pandas as pd
import pytest

from reformatters.common import template_utils, validation
from reformatters.common.storage import DatasetFormat, StorageConfig
from reformatters.google.weathernext2.forecast_operational_virtual.dynamical_dataset import (
    GoogleWeathernext2ForecastOperationalVirtualDataset,
)
from reformatters.google.weathernext2.forecast_operational_virtual.template_config import (
    GoogleWeathernext2ForecastOperationalVirtualTemplateConfig,
)
from reformatters.google.weathernext2.forecast_virtual.region_job import (
    GoogleWeathernext2ForecastOperationalVirtualRegionJob,
)

OPERATIONAL = GoogleWeathernext2ForecastOperationalVirtualTemplateConfig()


@pytest.mark.parametrize("older_complete", [False, True])
def test_completeness_allows_only_the_newest_init_to_be_partial(
    older_complete: bool,
) -> None:
    template = OPERATIONAL.get_template(pd.Timestamp("2025-01-01T18:00")).isel(
        lead_time=slice(0, 20), y=slice(0, 1), x=slice(0, 1)
    )
    var = next(var for var in OPERATIONAL.data_vars if var.path == "temperature_2m")
    job = GoogleWeathernext2ForecastOperationalVirtualRegionJob(
        tmp_store=Path("unused"),
        template_ds=template,
        data_vars=[var],
        append_dim="init_time",
        region=slice(0, 3),
        reformat_job_name="landing",
    )
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    session = repo.writable_session("main")
    template_utils.write_metadata(
        template, session.store, "w-", consolidated=False, skip_icechunk_commit=True
    )
    for init_index, leads in enumerate((20, 20 if older_complete else 10, 10)):
        for lead_index in range(leads):
            session.store.set_virtual_ref(
                f"{var.path}/c/{init_index}/0/{lead_index}/0/0",
                "https://example.com/source-chunk",
                offset=0,
                length=16,
                validate_container=False,
            )
    session.commit("landing")
    dataset = GoogleWeathernext2ForecastOperationalVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path="unused", format=DatasetFormat.ICECHUNK
        )
    )
    (completeness,) = [
        check
        for check in dataset.validators()
        if isinstance(check, validation.CheckVirtualManifestCompleteness)
    ]
    store = repo.readonly_session("main").store
    context = validation.ValidationContext(
        store=store,
        ds=validation.open_flattened_dataset(store, consolidated=False),
        append_dim="init_time",
        region_job=job,
    )
    result = completeness.check(context)
    assert result.passed is older_complete
    if not older_complete:
        assert "init_time=2025-01-01 06:00:00" in result.message
