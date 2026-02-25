from datetime import timedelta

import pytest

from reformatters.common.kubernetes import CronJob, ReformatCronJob, ValidationCronJob
from reformatters.common.staging import (
    _MAX_KUBERNETES_NAME_LENGTH,
    rename_cronjob_for_staging,
    staging_branch_name,
    staging_cronjob_name,
    staging_cronjob_names,
)


def _make_cronjob(name: str) -> CronJob:
    cls = ReformatCronJob if name.endswith("-update") else ValidationCronJob
    return cls(
        name=name,
        schedule="0 * * * *",
        image="test:latest",
        dataset_id="test-dataset",
        cpu="1",
        memory="1G",
        pod_active_deadline=timedelta(minutes=10),
    )


class TestStagingCronjobName:
    def test_basic(self) -> None:
        assert (
            staging_cronjob_name("noaa-gfs-forecast", "0.3.0", "update")
            == "stage-noaa-gfs-forecast-v0-3-0-update"
        )

    def test_dots_replaced_with_dashes(self) -> None:
        name = staging_cronjob_name("test", "1.2.3", "validate")
        assert "." not in name
        assert name == "stage-test-v1-2-3-validate"

    def test_trims_dataset_id_to_fit_kubernetes_limit(self) -> None:
        long_id = "a" * 60
        name = staging_cronjob_name(long_id, "0.3.0", "update")
        assert len(name) <= _MAX_KUBERNETES_NAME_LENGTH
        assert name.startswith("stage-")
        assert name.endswith("-v0-3-0-update")

    def test_returns_longest_name_under_limit(self) -> None:
        long_id = "a" * 60
        name = staging_cronjob_name(long_id, "0.3.0", "update")
        # Adding one more char to the dataset_id portion should exceed the limit
        parts_without_id = "stage--v0-3-0-update"
        trimmed_id_len = _MAX_KUBERNETES_NAME_LENGTH - len(parts_without_id)
        assert name == f"stage-{'a' * trimmed_id_len}-v0-3-0-update"

    def test_no_trimming_when_fits(self) -> None:
        name = staging_cronjob_name("short", "0.1.0", "update")
        assert name == "stage-short-v0-1-0-update"


class TestRenameCronjobForStaging:
    def test_renames_update_cronjob(self) -> None:
        cronjob = _make_cronjob("noaa-gfs-forecast-update")
        result = rename_cronjob_for_staging(cronjob, "noaa-gfs-forecast", "0.3.0")
        assert result.name == "stage-noaa-gfs-forecast-v0-3-0-update"

    def test_renames_validate_cronjob(self) -> None:
        cronjob = _make_cronjob("noaa-gfs-forecast-validate")
        result = rename_cronjob_for_staging(cronjob, "noaa-gfs-forecast", "0.3.0")
        assert result.name == "stage-noaa-gfs-forecast-v0-3-0-validate"

    def test_preserves_other_fields(self) -> None:
        cronjob = _make_cronjob("test-dataset-update")
        result = rename_cronjob_for_staging(cronjob, "test-dataset", "1.0.0")
        assert result.schedule == cronjob.schedule
        assert result.image == cronjob.image
        assert result.dataset_id == cronjob.dataset_id
        assert result.cpu == cronjob.cpu

    def test_asserts_on_unexpected_name_prefix(self) -> None:
        cronjob = _make_cronjob("wrong-prefix-update")
        with pytest.raises(AssertionError):
            rename_cronjob_for_staging(cronjob, "noaa-gfs-forecast", "0.3.0")

    def test_longest_real_dataset_id_fits(self) -> None:
        # ecmwf-ifs-ens-forecast-15-day-0-25-degree is currently the longest
        dataset_id = "ecmwf-ifs-ens-forecast-15-day-0-25-degree"
        cronjob = _make_cronjob(f"{dataset_id}-update")
        result = rename_cronjob_for_staging(cronjob, dataset_id, "0.3.0")
        assert len(result.name) <= _MAX_KUBERNETES_NAME_LENGTH


class TestStagingCronjobNames:
    def test_returns_update_and_validate(self) -> None:
        names = staging_cronjob_names("noaa-gfs-forecast", "0.3.0")
        assert names == [
            "stage-noaa-gfs-forecast-v0-3-0-update",
            "stage-noaa-gfs-forecast-v0-3-0-validate",
        ]


class TestStagingBranchName:
    def test_basic(self) -> None:
        assert (
            staging_branch_name("noaa-gfs-forecast", "0.3.0")
            == "stage/noaa-gfs-forecast/v0.3.0"
        )
