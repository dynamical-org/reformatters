import pandas as pd
from typer.testing import CliRunner

from reformatters.noaa.hrrr.nomads_mirror.hrrr_nomads_cache_mirror import (
    NoaaHrrrNomadsCacheMirror,
    mirror_window,
)


def test_cron_and_cli() -> None:
    resource = NoaaHrrrNomadsCacheMirror()
    (cron,) = resource.operational_kubernetes_resources("test-image")
    assert not cron.suspend
    assert len(cron.name) <= 52
    assert cron.command == ["mirror-gribs"]
    assert cron.command[0] in {
        (command.name or command.callback.__name__).replace("_", "-")  # ty: ignore[unresolved-attribute]
        for command in resource.get_cli().registered_commands
        if command.callback is not None
    }
    result = CliRunner().invoke(resource.get_cli(), ["--help"])
    assert result.exit_code == 0
    assert "mirror-gribs" in result.output
    now = pd.Timestamp("2026-09-08T19:47Z")
    assert cron.previous_fire_time(now) == pd.Timestamp("2026-09-08T19:45Z")
    assert mirror_window(now, cron, 49) == tuple(
        pd.Timestamp(value)
        for value in ["2026-09-08T19:00Z", "2026-09-08T19:49Z", "2026-09-08T20:43Z"]
    )
