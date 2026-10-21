import json
from pathlib import Path
from typing import Any

import pytest

from reformatters.noaa.urma.analysis_virtual.region_job import (
    NoaaUrmaAnalysisVirtualRegionJob,
)


@pytest.fixture(params=sorted((Path(__file__).parent / "fixtures").glob("*.json")))
def source_fixture(request: pytest.FixtureRequest) -> dict[str, Any]:
    return json.loads(request.param.read_text())


@pytest.fixture(autouse=True)
def limit_source_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(NoaaUrmaAnalysisVirtualRegionJob, "download_concurrency", 4)
