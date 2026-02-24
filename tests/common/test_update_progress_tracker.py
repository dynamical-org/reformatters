import json
from pathlib import Path

import numpy as np
import pytest

from reformatters.common.config_models import (
    BaseInternalAttrs,
    DataVar,
    DataVarAttrs,
    Encoding,
)
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.common.update_progress_tracker import (
    PROCESSED_VARIABLES_KEY,
    UpdateProgressTracker,
)


class _TestDataVar(DataVar[BaseInternalAttrs]):
    encoding: Encoding = Encoding(
        dtype="float32",
        fill_value=np.nan,
        chunks=(1,),
        shards=None,
    )
    attrs: DataVarAttrs = DataVarAttrs(
        units="K",
        long_name="Test variable",
        short_name="test",
        step_type="instant",
    )
    internal_attrs: BaseInternalAttrs = BaseInternalAttrs(keep_mantissa_bits=10)


def _make_var(name: str) -> _TestDataVar:
    return _TestDataVar(name=name)


@pytest.fixture
def store_factory(tmp_path: Path) -> StoreFactory:
    return StoreFactory(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path),
            format=DatasetFormat.ZARR3,
        ),
        dataset_id="test-dataset",
        template_config_version="v1",
    )


def _make_tracker(
    store_factory: StoreFactory,
    job_name: str = "job1",
    time_i: int = 0,
) -> UpdateProgressTracker:
    return UpdateProgressTracker(
        reformat_job_name=job_name,
        time_i_slice_start=time_i,
        store_factory=store_factory,
    )


def test_initial_state_empty_when_no_file(store_factory: StoreFactory) -> None:
    tracker = _make_tracker(store_factory)
    assert tracker.processed_variables == set()


def test_initial_state_loads_existing_progress_file(
    store_factory: StoreFactory,
) -> None:
    # Write a progress file first
    tracker = _make_tracker(store_factory)
    path = tracker._get_path()
    content = json.dumps({PROCESSED_VARIABLES_KEY: ["var_a", "var_b"]})
    tracker.fs.pipe(path, content.encode("utf-8"))

    # A second tracker with the same params should load the file
    tracker2 = _make_tracker(store_factory)
    assert tracker2.processed_variables == {"var_a", "var_b"}


def test_get_unprocessed_returns_unprocessed_vars(store_factory: StoreFactory) -> None:
    tracker = _make_tracker(store_factory)
    tracker.processed_variables = {"var_a"}

    all_vars = [_make_var("var_a"), _make_var("var_b"), _make_var("var_c")]
    unprocessed = tracker.get_unprocessed(all_vars)
    assert {v.name for v in unprocessed} == {"var_b", "var_c"}


def test_get_unprocessed_all_done_returns_first_var(
    store_factory: StoreFactory,
) -> None:
    tracker = _make_tracker(store_factory)
    tracker.processed_variables = {"var_a", "var_b"}

    all_vars = [_make_var("var_a"), _make_var("var_b")]
    result = tracker.get_unprocessed(all_vars)
    # When all are processed, return the first to ensure metadata is written
    assert len(result) == 1
    assert result[0].name == "var_a"


def test_record_completion_adds_to_processed_variables(
    store_factory: StoreFactory,
) -> None:
    tracker = _make_tracker(store_factory)

    tracker.record_completion("var_x")
    tracker.queue.join()  # wait for background thread

    assert "var_x" in tracker.processed_variables


def test_record_completion_persists_to_disk(store_factory: StoreFactory) -> None:
    tracker = _make_tracker(store_factory)

    tracker.record_completion("var_y")
    tracker.queue.join()

    content = tracker.fs.read_text(tracker._get_path(), encoding="utf-8")
    data = json.loads(content)
    assert "var_y" in data[PROCESSED_VARIABLES_KEY]


def test_record_completion_accumulates_multiple_vars(
    store_factory: StoreFactory,
) -> None:
    tracker = _make_tracker(store_factory)

    tracker.record_completion("var_a")
    tracker.record_completion("var_b")
    tracker.queue.join()

    assert tracker.processed_variables == {"var_a", "var_b"}


def test_close_deletes_progress_file(store_factory: StoreFactory) -> None:
    tracker = _make_tracker(store_factory)

    tracker.record_completion("var_z")
    tracker.queue.join()

    assert tracker.fs.exists(tracker._get_path())
    tracker.close()
    assert not tracker.fs.exists(tracker._get_path())


def test_close_does_not_raise_when_file_missing(store_factory: StoreFactory) -> None:
    tracker = _make_tracker(store_factory)
    # No file written — close should not raise
    tracker.close()


def test_get_path_contains_job_name_and_time_index(store_factory: StoreFactory) -> None:
    tracker = _make_tracker(store_factory, job_name="my-job", time_i=42)
    path = tracker._get_path()
    assert "my-job" in path
    assert "42" in path


def test_different_job_names_use_different_paths(store_factory: StoreFactory) -> None:
    t1 = _make_tracker(store_factory, job_name="job-a", time_i=0)
    t2 = _make_tracker(store_factory, job_name="job-b", time_i=0)
    assert t1._get_path() != t2._get_path()


def test_different_time_indices_use_different_paths(
    store_factory: StoreFactory,
) -> None:
    t1 = _make_tracker(store_factory, job_name="job", time_i=0)
    t2 = _make_tracker(store_factory, job_name="job", time_i=10)
    assert t1._get_path() != t2._get_path()
