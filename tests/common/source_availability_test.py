from copy import deepcopy
from typing import Any

import pandas as pd
import pytest

from reformatters.common.source_availability import SourceAvailability, Summary


@pytest.fixture
def summary_data() -> dict[str, Any]:
    return {
        "products": [
            {
                "id": "hrrr",
                "recent_inits": [
                    {
                        "init_time": "2026-10-07T12:00:00Z",
                        "lead_groups": [
                            {
                                "max_lead": lead,
                                "status": "in_flight",
                                "facets": [
                                    {
                                        "name": "component:conus/sfc",
                                        "status": "complete",
                                    },
                                    {
                                        "name": "component:conus/nat",
                                        "status": "in_flight",
                                    },
                                ],
                            }
                            for lead in (0, 6, 18, 48)
                        ],
                    }
                ],
            }
        ],
    }


def test_component_readiness_does_not_wait_for_unrelated_files(
    summary_data: dict[str, Any],
) -> None:
    summary = Summary.model_validate(summary_data)
    surface = SourceAvailability(
        product_id="hrrr", lead_hours=48, components=("conus/sfc",)
    )
    assert surface.available_inits(summary) == {pd.Timestamp("2026-10-07T12:00")}
    assert not SourceAvailability(product_id="hrrr", lead_hours=48).available_inits(
        summary
    )


@pytest.mark.parametrize("missing_group", [0, 1, 2, 3])
def test_component_requires_every_lead_interval(
    summary_data: dict[str, Any],
    missing_group: int,
) -> None:
    groups = summary_data["products"][0]["recent_inits"][0]["lead_groups"]
    groups[missing_group]["facets"][0]["status"] = "in_flight"
    surface = SourceAvailability(
        product_id="hrrr", lead_hours=48, components=("conus/sfc",)
    )
    assert not surface.available_inits(Summary.model_validate(summary_data))


def test_absent_component_or_final_group_is_not_ready(
    summary_data: dict[str, Any],
) -> None:
    absent = SourceAvailability(
        product_id="hrrr", lead_hours=48, components=("missing",)
    )
    assert not absent.available_inits(Summary.model_validate(summary_data))
    absent_lead = SourceAvailability(product_id="hrrr", lead_hours=840)
    assert not absent_lead.available_inits(Summary.model_validate(summary_data))


def test_missing_source_product_names_the_dependency() -> None:
    availability = SourceAvailability(product_id="missing-source", lead_hours=48)
    with pytest.raises(ValueError, match=r"wxopticon.*missing-source"):
        availability.available_inits(Summary(products=[]))


def test_source_readiness_is_per_init_and_phase(summary_data: dict[str, Any]) -> None:
    run = summary_data["products"][0]["recent_inits"][0]
    for group in run["lead_groups"]:
        group["status"] = "complete"
    run["lead_groups"][-1]["status"] = "in_flight"
    previous = deepcopy(run)
    previous["init_time"] = "2026-10-07T06:00:00Z"
    previous["lead_groups"][-1]["status"] = "complete"
    summary_data["products"][0]["recent_inits"].append(previous)
    summary = Summary.model_validate(summary_data)
    assert SourceAvailability(product_id="hrrr", lead_hours=48).available_inits(
        summary
    ) == {pd.Timestamp("2026-10-07T06:00")}
    assert (
        len(
            SourceAvailability(product_id="hrrr", lead_hours=18).available_inits(
                summary
            )
        )
        == 2
    )
