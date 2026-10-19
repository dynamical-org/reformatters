import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(params=["20170502", "20261015"])
def source_fixture(request: pytest.FixtureRequest) -> dict[str, Any]:
    return json.loads(
        (Path(__file__).parent / "fixtures" / f"{request.param}.json").read_text()
    )
