import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(
    params=[
        "20171213t0000",
        "20171213t0015",
        "20200727t2345",
        "20200728t0000",
        "20200728t1245",
        "20261015t1445",
    ]
)
def source_fixture(request: pytest.FixtureRequest) -> dict[str, Any]:
    return json.loads(
        (Path(__file__).parent / "fixtures" / f"{request.param}.json").read_text()
    )
