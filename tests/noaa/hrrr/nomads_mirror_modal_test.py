import json
import subprocess
import sys
from pathlib import Path

# pytest's `pythonpath` puts src/reformatters first, where the `google` package
# shadows the namespace modal imports, so the Modal module is exercised in a
# subprocess with a plain interpreter path.
SCRIPT = r"""
import json
import warnings
import pandas as pd
warnings.simplefilter("ignore")
from reformatters.noaa.hrrr import nomads_mirror_modal as m

report = {}

windows = {}
for now in ["2026-09-23T17:49Z", "2026-09-23T18:48:59Z", "2026-09-23T18:49Z"]:
    init_time, deadline = m.fire_window(pd.Timestamp(now))
    windows[now] = [init_time.isoformat(), deadline.isoformat()]
report["windows"] = windows

now = pd.Timestamp("2026-09-23T20:00Z")
report["pilot_arguments"] = list(
    m.pilot_arguments("2026-09-23T17:00", "pilot", "sfc,nat", "0,18", 10, now)
)
rejected = {}
for name, args in {
    "no_types": ("2026-09-23T17:00", "pilot", "", "0", 10),
    "bad_lead": ("2026-09-23T17:00", "pilot", "sfc", "19", 10),
    "no_minutes": ("2026-09-23T17:00", "pilot", "sfc", "0", 0),
    "too_long": ("2026-09-23T17:00", "pilot", "sfc", "0", 41),
    "still_publishing": ("2026-09-23T19:00", "pilot", "sfc", "0", 10),
}.items():
    try:
        m.pilot_arguments(*args, now)
        rejected[name] = False
    except AssertionError:
        rejected[name] = True
report["rejected"] = rejected

calls = []
class FakePilotCopy:
    def remote(self, *args):
        calls.append(list(args))
        return {"copied": [], "pending": []}
m.pilot_copy = FakePilotCopy()
m.pilot.info.raw_f("2026-01-01T00:00", "pilot", file_types="prs", leads="3", minutes=5)
report["remote_calls"] = calls

report["apps"] = {
    "app": [m.app.name, sorted(m.app.registered_functions)],
    "pilot_app": [m.pilot_app.name, sorted(m.pilot_app.registered_functions)],
}
print(json.dumps(report))
"""


def run_modal_module_checks() -> dict[str, object]:
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", SCRIPT],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        check=True,
    )
    result: dict[str, object] = json.loads(completed.stdout.splitlines()[-1])
    return result


def test_the_modal_app_module() -> None:
    report = run_modal_module_checks()

    assert report["windows"] == {
        "2026-09-23T17:49Z": ["2026-09-23T17:00:00+00:00", "2026-09-23T18:43:00+00:00"],
        # A run starting just before the next fire still belongs to the 17:49 fire,
        "2026-09-23T18:48:59Z": [
            "2026-09-23T17:00:00+00:00",
            "2026-09-23T18:43:00+00:00",
        ],
        # and one starting at or after it works on the latest fire's init.
        "2026-09-23T18:49Z": ["2026-09-23T18:00:00+00:00", "2026-09-23T19:43:00+00:00"],
    }
    assert report["pilot_arguments"] == [
        "2026-09-23T17:00",
        "pilot",
        ["sfc", "nat"],
        [0, 18],
        "2026-09-23T20:10:00+00:00",
    ]
    assert report["rejected"] == {
        "no_types": True,
        "bad_lead": True,
        "no_minutes": True,
        "too_long": True,
        "still_publishing": True,
    }
    (call,) = report["remote_calls"]  # ty: ignore[not-iterable]
    assert call[:4] == ["2026-01-01T00:00", "pilot", ["prs"], [3]]
    assert report["apps"] == {
        "app": ["noaa-hrrr-nomads-mirror", ["mirror_gribs"]],
        "pilot_app": ["noaa-hrrr-nomads-mirror-pilot", ["pilot_copy"]],
    }
