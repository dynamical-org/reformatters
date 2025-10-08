import multiprocessing
import os
import sys
from pathlib import Path

# Spawn new processes since fork isn't safe with threads
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # Already set, ignore
    pass


import pytest

# Make tests able to import from other files in tests/
sys.path.append(str(Path(__file__).parent.parent))

# This needs to run before any application imports to ensure that
# Config.env is set to test.
os.environ["DYNAMICAL_ENV"] = "test"

from reformatters.common import storage


def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    """
    Determine the number of parallel test workers.

    Disables xdist when a single test file is specified,
    otherwise uses all available CPUs.
    """
    file_or_dir = config.option.file_or_dir
    if (
        file_or_dir
        and len(file_or_dir) == 1
        and file_or_dir[0].split("::")[0].endswith(".py")
    ):
        return 0
    return None  # use default behavior (all CPUs)


@pytest.fixture(autouse=True)
def set_local_zarr_store_base_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    if (
        request.node.get_closest_marker("skip_set_local_zarr_store_base_path")
        is not None
    ):
        return
    monkeypatch.setattr(storage, "_LOCAL_ZARR_STORE_BASE_PATH", tmp_path)
