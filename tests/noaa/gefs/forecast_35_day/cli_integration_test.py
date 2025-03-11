import json
import os
from pathlib import Path

from pytest import MonkeyPatch

from reformatters.noaa.gefs.forecast_35_day import cli, template


def test_update_template(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    latest_zarr_template_path = template.TEMPLATE_PATH
    template_path = tmp_path / "latest.zarr"
    monkeypatch.setattr(template, "TEMPLATE_PATH", template_path)

    cli.update_template()

    with open(os.path.join(template_path, "zarr.json")) as test_f:
        test_zarr_json = json.load(test_f)

    with open(os.path.join(latest_zarr_template_path, "zarr.json")) as latest_f:
        latest_zarr_json = json.load(latest_f)

    assert json.dumps(test_zarr_json) == json.dumps(latest_zarr_json)
