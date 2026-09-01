import pytest

from tests.common import datasets_cf_compliance_test as compliance

UNRESOLVED_CLOUD_COVER_EXCEPTION = (
    "total_cloud_cover_atmosphere",
    "step_type",
    "ecmwf-ifs-ens-forecast-46-day-1-5-degree",
)


def _cloud_cover_metadata() -> dict[
    str, dict[str, dict[str, compliance.MetadataValue]]
]:
    return {
        "total_cloud_cover_atmosphere": {
            "ecmwf-ifs-ens-forecast-46-day-1-5-degree": {"step_type": "avg"},
            "conforming-dataset": {"step_type": "instant"},
        }
    }


def test_check_consistency_exception_narrows_remaining_values() -> None:
    assert compliance._check_consistency(_cloud_cover_metadata(), ["step_type"]) == []


def test_check_consistency_accepts_conforming_new_dataset() -> None:
    metadata = _cloud_cover_metadata()
    metadata["total_cloud_cover_atmosphere"]["new-sibling"] = {"step_type": "instant"}

    assert compliance._check_consistency(metadata, ["step_type"]) == []


def test_check_consistency_rejects_new_divergence() -> None:
    metadata = _cloud_cover_metadata()
    metadata["total_cloud_cover_atmosphere"]["new-divergence"] = {"step_type": "max"}

    conflicts = compliance._check_consistency(metadata, ["step_type"])

    assert len(conflicts) == 1
    assert "new-divergence" in conflicts[0]


def test_unresolved_cloud_cover_exception_is_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compliance,
        "CROSS_DATASET_CONSISTENCY_EXCEPTIONS",
        compliance.CROSS_DATASET_CONSISTENCY_EXCEPTIONS
        - {UNRESOLVED_CLOUD_COVER_EXCEPTION},
    )

    with pytest.raises(
        AssertionError, match="total_cloud_cover_atmosphere step_type conflict"
    ):
        compliance.test_metadata_consistency_across_datasets()
