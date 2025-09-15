import json
from pathlib import Path

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


def create_example_datavar(name: str = "test_var") -> DataVar[BaseInternalAttrs]:
    return DataVar(
        name=name,
        encoding=Encoding(
            dtype="float32",
            chunks=(5, 5),
            shards=None,
            fill_value=0.0,
        ),
        attrs=DataVarAttrs(
            long_name=f"Test variable {name}",
            short_name=name,
            units="unitless",
            step_type="instant",
        ),
        internal_attrs=BaseInternalAttrs(
            keep_mantissa_bits="no-rounding",
        ),
    )


@pytest.fixture
def store_factory() -> StoreFactory:
    return StoreFactory(
        primary_storage_config=StorageConfig(
            base_path="fake-prod-path",
            format=DatasetFormat.ZARR3,
        ),
        dataset_id="test",
        template_config_version="dev",
    )


def test_update_progress_tracker_initialization_with_existing_progress(
    store_factory: StoreFactory,
) -> None:
    """Test UpdateProgressTracker correctly loads existing processed variables"""
    tracker = UpdateProgressTracker("test-job", 0, store_factory)

    # Create existing progress file
    progress_file = (
        Path(tracker.update_progress_dir) / "_internal_update_progress_test-job_0.json"
    )
    existing_vars = ["var1", "var2", "var3"]
    tracker.fs.write_text(
        progress_file,
        json.dumps({"processed_variables": existing_vars}),
    )

    # Create new tracker - should load existing progress
    new_tracker = UpdateProgressTracker("test-job", 0, store_factory)
    assert new_tracker.processed_variables == set(existing_vars)
    new_tracker.close()


@pytest.mark.parametrize(
    "processed_vars,all_vars,expected_unprocessed",
    [
        # Test case 1: Some variables processed, some not
        ({"var1", "var3"}, ["var1", "var2", "var3", "var4"], ["var2", "var4"]),
        # Test case 2: All variables processed - should return first variable
        ({"var1", "var2", "var3", "var4"}, ["var1", "var2", "var3", "var4"], ["var1"]),
    ],
)
def test_get_unprocessed_with_datavar_objects_parametrized(
    processed_vars: set[str],
    all_vars: list[str],
    expected_unprocessed: list[str],
    store_factory: StoreFactory,
) -> None:
    """Test get_unprocessed with DataVar objects - normal case and edge case"""
    tracker = UpdateProgressTracker("test-job", 0, store_factory)

    # Set processed variables
    tracker.processed_variables = processed_vars

    # Create DataVar objects
    datavars = [create_example_datavar(var_name) for var_name in all_vars]

    # Get unprocessed variables
    unprocessed = tracker.get_unprocessed(datavars)

    # Verify we never get None and the results match expectations
    assert unprocessed is not None
    assert len(unprocessed) > 0  # Should never be empty
    assert [v.name for v in unprocessed] == expected_unprocessed

    tracker.close()


def test_record_completion_adds_to_processed_variables(
    store_factory: StoreFactory,
) -> None:
    """Test record_completion adds variables to processed set"""
    tracker = UpdateProgressTracker("test-job", 0, store_factory)

    initial_count = len(tracker.processed_variables)
    tracker.record_completion("new_var")

    # Wait for processing to complete
    tracker.queue.join()
    assert len(tracker.processed_variables) == initial_count + 1
    assert "new_var" in tracker.processed_variables
    tracker.close()


def test_close_deletes_progress_file(store_factory: StoreFactory) -> None:
    tracker = UpdateProgressTracker("test-job", 0, store_factory)
    assert not tracker.fs.exists(tracker._get_path())

    tracker.fs.write_text(
        tracker._get_path(), json.dumps({"processed_variables": ["existing_var"]})
    )
    assert tracker.fs.exists(tracker._get_path())

    tracker.close()
    assert not tracker.fs.exists(tracker._get_path())


def test_process_queue_multiple_items_written_separately(
    store_factory: StoreFactory,
) -> None:
    """Test that multiple items on the queue are processed and written with separate content"""
    tracker = UpdateProgressTracker("test-job", 0, store_factory)

    # Record two variables to the queue
    tracker.record_completion("var1")
    tracker.record_completion("var2")

    # Wait for both items to be processed
    tracker.queue.join()

    # Verify both variables are in the processed set
    assert "var1" in tracker.processed_variables
    assert "var2" in tracker.processed_variables
    assert len(tracker.processed_variables) == 2

    # Verify the progress file exists and contains both variables
    assert tracker.fs.exists(tracker._get_path())

    # Read the content and verify it contains both variables
    content = tracker.fs.read_text(tracker._get_path(), encoding="utf-8")
    progress_data = json.loads(content)

    assert PROCESSED_VARIABLES_KEY in progress_data
    processed_vars = set(progress_data[PROCESSED_VARIABLES_KEY])
    assert processed_vars == {"var1", "var2"}

    tracker.close()
