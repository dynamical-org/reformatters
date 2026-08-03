from collections.abc import Sequence
from unittest.mock import Mock

import pandas as pd
import pytest

from reformatters.common.region_job import SourceFileResult, SourceFileStatus
from reformatters.eccc.hrdps.analysis.region_job import (
    EcccHrdpsAnalysisRegionJob,
    EcccHrdpsAnalysisSourceFileCoord,
)
from reformatters.eccc.hrdps.analysis.template_config import (
    EcccHrdpsAnalysisTemplateConfig,
)


@pytest.fixture
def template_config() -> EcccHrdpsAnalysisTemplateConfig:
    return EcccHrdpsAnalysisTemplateConfig()


def test_source_file_coord_out_loc(
    template_config: EcccHrdpsAnalysisTemplateConfig,
) -> None:
    coord = EcccHrdpsAnalysisSourceFileCoord(
        init_time=pd.Timestamp("2026-07-09T06:00"),
        lead_time=pd.Timedelta("2h"),
        data_var=template_config.data_vars[0],
    )
    assert coord.out_loc() == {"time": pd.Timestamp("2026-07-09T08:00")}


@pytest.mark.parametrize(
    ("region", "expected_processing_region"),
    [
        (slice(0, 100), slice(0, 100)),  # At start: no buffer possible, clips to 0
        (slice(1, 100), slice(0, 100)),  # At index 1: buffer clips to 0
        (slice(10, 100), slice(9, 100)),  # Mid-dataset: full buffer of 1
    ],
)
def test_get_processing_region(
    template_config: EcccHrdpsAnalysisTemplateConfig,
    region: slice,
    expected_processing_region: slice,
) -> None:
    region_job = EcccHrdpsAnalysisRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=Mock(),
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=region,
        reformat_job_name="test",
    )

    assert region_job.get_processing_region() == expected_processing_region


def _generate_coords(
    template_config: EcccHrdpsAnalysisTemplateConfig, var_name: str
) -> Sequence[EcccHrdpsAnalysisSourceFileCoord]:
    data_var = next(v for v in template_config.data_vars if v.name == var_name)
    # Times 2026-07-09T04:00 through 2026-07-09T08:00 span an init boundary at 06:00
    template_ds = template_config.get_template(pd.Timestamp("2026-07-09T09:00"))
    region_job = EcccHrdpsAnalysisRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=[data_var],
        append_dim=template_config.append_dim,
        region=slice(4, 9),
        reformat_job_name="test",
    )
    processing_region_ds, _ = region_job._get_region_datasets()
    coords = region_job.generate_source_file_coords(processing_region_ds, [data_var])
    return coords


def test_generate_source_file_coords_instant_var(
    template_config: EcccHrdpsAnalysisTemplateConfig,
) -> None:
    """Instant variables use the shortest lead time (0-5h) from the most recent init."""
    coords = _generate_coords(template_config, "temperature_2m")

    # 6 coords: 1 deaccumulation buffer step + 5 output steps
    assert [(c.init_time, c.lead_time) for c in coords] == [
        (pd.Timestamp("2026-07-09T00:00"), pd.Timedelta("3h")),
        (pd.Timestamp("2026-07-09T00:00"), pd.Timedelta("4h")),
        (pd.Timestamp("2026-07-09T00:00"), pd.Timedelta("5h")),
        (pd.Timestamp("2026-07-09T06:00"), pd.Timedelta("0h")),
        (pd.Timestamp("2026-07-09T06:00"), pd.Timedelta("1h")),
        (pd.Timestamp("2026-07-09T06:00"), pd.Timedelta("2h")),
    ]


def test_generate_source_file_coords_accumulated_var(
    template_config: EcccHrdpsAnalysisTemplateConfig,
) -> None:
    """Accumulated variables have no hour 0 file so use lead times 1-6h."""
    coords = _generate_coords(template_config, "precipitation_surface")

    assert [(c.init_time, c.lead_time) for c in coords] == [
        (pd.Timestamp("2026-07-09T00:00"), pd.Timedelta("3h")),
        (pd.Timestamp("2026-07-09T00:00"), pd.Timedelta("4h")),
        (pd.Timestamp("2026-07-09T00:00"), pd.Timedelta("5h")),
        (pd.Timestamp("2026-07-09T00:00"), pd.Timedelta("6h")),
        (pd.Timestamp("2026-07-09T06:00"), pd.Timedelta("1h")),
        (pd.Timestamp("2026-07-09T06:00"), pd.Timedelta("2h")),
    ]


def test_update_template_with_results_trims_last_hour(
    template_config: EcccHrdpsAnalysisTemplateConfig,
) -> None:
    template_ds = template_config.get_template(pd.Timestamp("2026-07-09T05:00"))

    region_job = EcccHrdpsAnalysisRegionJob.model_construct(
        tmp_store=Mock(),
        template_ds=template_ds,
        data_vars=template_config.data_vars[:1],
        append_dim=template_config.append_dim,
        region=slice(0, 5),
        reformat_job_name="test",
    )

    last_time = pd.Timestamp(template_ds.time.values[-1])
    process_results = {
        template_config.data_vars[0].name: [
            SourceFileResult(
                status=SourceFileStatus.Succeeded,
                out_loc={"time": last_time},
                url="https://test/",
            )
        ]
    }

    result_ds = region_job.update_template_with_results(process_results)

    assert len(result_ds.time) == len(template_ds.time) - 1
    assert result_ds.time[-1] == template_ds.time.values[-2]
