import os
import sys
from pathlib import Path

import pytest

# Make tests able to import from other files in tests/
sys.path.append(str(Path(__file__).parent.parent))

# This needs to run before any application imports to ensure that
# Config.env is set to test.
os.environ["DYNAMICAL_ENV"] = "test"

from reformatters.common import zarr as common_zarr_module


@pytest.fixture(autouse=True)
def set_local_zarr_store_base_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    if (
        request.node.get_closest_marker("skip_set_local_zarr_store_base_path")
        is not None
    ):
        return
    monkeypatch.setattr(common_zarr_module, "_LOCAL_ZARR_STORE_BASE_PATH", tmp_path)
