from datetime import timedelta
from itertools import count
from pathlib import Path
from unittest.mock import Mock

import obstore
import obstore.store
import pandas as pd
import pytest
from typer.testing import CliRunner

from reformatters.common.kubernetes import CronJob
from reformatters.noaa.hrrr import nomads_mirror
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrFileType
from reformatters.noaa.hrrr.nomads_mirror import (
    NoaaHrrrNomadsMirror,
    mirror_init_time,
    mirror_key,
    mirror_window,
)
from reformatters.noaa.hrrr.region_job import NoaaHrrrSourceFileCoord

FIXTURE_GRIB = Path(__file__).parents[1] / "fixtures/hrrr.t19z.wrfsfcf00.first2.grib2"
FIXTURE_INDEX = FIXTURE_GRIB.with_name(FIXTURE_GRIB.name + ".idx")
INIT = pd.Timestamp("2026-09-07T19:00")
runner = CliRunner()


def coord(lead: int, file_type: NoaaHrrrFileType = "sfc") -> NoaaHrrrSourceFileCoord:
    return NoaaHrrrSourceFileCoord(
        init_time=INIT,
        lead_time=pd.Timedelta(hours=lead),
        domain="conus",
        file_type=file_type,
        data_vars=[],
    )


def test_mirror_key_is_the_nodd_key() -> None:
    assert mirror_key(coord(1)) == "hrrr.20260907/conus/hrrr.t19z.wrfsfcf01.grib2"


class FakeNomads:
    """A NOMADS directory whose contents the test publishes over time."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.listed: dict[str, bytes] = {}
        self.listings = 0
        self.fetched: list[str] = []

    def publish(self, name: str, content: bytes) -> None:
        self.listed[name] = content

    def list_directory(self, url: str) -> set[str]:
        assert url.endswith("/hrrr.20260907/conus/")
        self.listings += 1
        return set(self.listed)

    def fetch(self, url: str) -> Path:
        name = url.rsplit("/", 1)[-1]
        self.fetched.append(name)
        path = self.tmp_path / f"{next(_counter)}-{name}"
        path.write_bytes(self.listed[name])
        return path


_counter = count()


def run(
    nomads: FakeNomads,
    mirror: obstore.store.ObjectStore,
    *,
    polls: int,
    monkeypatch: pytest.MonkeyPatch,
    file_types: tuple[NoaaHrrrFileType, ...] = ("sfc",),
    lead_hours: tuple[int, ...] = (0, 1),
) -> nomads_mirror.MirrorResult:
    """Poll `polls` times; the fake clock reaches the deadline after that."""
    start = pd.Timestamp("2026-09-07T19:49Z")
    clock = count()
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda cls, *a, **k: start + pd.Timedelta(seconds=next(clock))),
    )
    monkeypatch.setattr(nomads_mirror.time, "sleep", lambda _s: None)
    return mirror_init_time(
        INIT,
        mirror,
        deadline=start + pd.Timedelta(seconds=polls),
        file_types=file_types,
        lead_hours=lead_hours,
        poll_interval=timedelta(0),
        list_directory=nomads.list_directory,
        fetch=nomads.fetch,
    )


def mirror_contents(mirror: obstore.store.ObjectStore) -> set[str]:
    return {meta["path"] for batch in obstore.list(mirror) for meta in batch}


def test_copies_each_file_as_nomads_publishes_it_data_before_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    grib, index = FIXTURE_GRIB.read_bytes(), FIXTURE_INDEX.read_bytes()
    # The index is listed before its data file, as NOMADS sometimes does.
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2.idx", index)
    result = run(nomads, mirror, polls=1, monkeypatch=monkeypatch)
    assert result.copied == []
    assert nomads.fetched == []

    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", grib)
    nomads.publish("hrrr.t19z.wrfsfcf01.grib2", grib)
    nomads.publish("hrrr.t19z.wrfsfcf01.grib2.idx", index)
    result = run(nomads, mirror, polls=2, monkeypatch=monkeypatch)

    f00, f01 = mirror_key(coord(0)), mirror_key(coord(1))
    assert result.copied == [f00, f00 + ".idx", f01, f01 + ".idx"]
    assert result.pending == []
    assert mirror_contents(mirror) == set(result.copied)
    # One listing per poll; each object fetched once.
    assert nomads.listings == 1 + 1
    assert sorted(nomads.fetched) == sorted(
        name.rsplit("/", 1)[-1] for name in result.copied
    )


def test_files_already_in_the_mirror_are_not_fetched_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    obstore.put(mirror, mirror_key(coord(0)), FIXTURE_GRIB.read_bytes())
    obstore.put(mirror, mirror_key(coord(0)) + ".idx", FIXTURE_INDEX.read_bytes())
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", FIXTURE_GRIB.read_bytes())
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2.idx", FIXTURE_INDEX.read_bytes())
    nomads.publish("hrrr.t19z.wrfsfcf01.grib2", FIXTURE_GRIB.read_bytes())

    result = run(nomads, mirror, polls=1, monkeypatch=monkeypatch)

    assert result.copied == [mirror_key(coord(1))]
    assert result.pending == [mirror_key(coord(1)) + ".idx"]
    assert nomads.fetched == ["hrrr.t19z.wrfsfcf01.grib2"]


def test_a_partially_written_data_file_is_retried_not_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    whole = FIXTURE_GRIB.read_bytes()
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", whole[: len(whole) // 2])
    result = run(nomads, mirror, polls=2, monkeypatch=monkeypatch, lead_hours=(0,))
    assert result.copied == []
    assert nomads.fetched == ["hrrr.t19z.wrfsfcf00.grib2"] * 2
    assert mirror_contents(mirror) == set()

    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", whole)
    result = run(nomads, mirror, polls=1, monkeypatch=monkeypatch, lead_hours=(0,))
    assert result.copied == [mirror_key(coord(0))]


def test_every_file_type_is_mirrored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    file_types: tuple[NoaaHrrrFileType, ...] = ("sfc", "prs", "nat")
    for file_type in file_types:
        nomads.publish(f"hrrr.t19z.wrf{file_type}f00.grib2", FIXTURE_GRIB.read_bytes())
        nomads.publish(
            f"hrrr.t19z.wrf{file_type}f00.grib2.idx", FIXTURE_INDEX.read_bytes()
        )
    result = run(
        nomads,
        mirror,
        polls=1,
        monkeypatch=monkeypatch,
        file_types=file_types,
        lead_hours=(0,),
    )
    assert sorted(result.copied) == sorted(
        mirror_key(coord(0, t)) + suffix for t in file_types for suffix in ("", ".idx")
    )


def test_is_whole_grib2_rejects_a_cut_inside_a_message(tmp_path: Path) -> None:
    whole = FIXTURE_GRIB.read_bytes()
    assert nomads_mirror.is_whole_grib2(FIXTURE_GRIB)
    cut = tmp_path / "cut.grib2"
    cut.write_bytes(whole[: len(whole) - 100])
    assert not nomads_mirror.is_whole_grib2(cut)
    not_grib = tmp_path / "x.grib2"
    not_grib.write_bytes(b"<html>rate limited</html>")
    assert not nomads_mirror.is_whole_grib2(not_grib)


def test_downloads_and_listings_go_through_the_nomads_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download = Mock(return_value=Path("download"))
    monkeypatch.setattr(nomads_mirror, "httpx_download_to_disk", download)
    assert nomads_mirror.download_from_nomads("url") == Path("download")
    download.assert_called_once_with(
        "url",
        "noaa-hrrr-nomads",
        rate_limiter=nomads_mirror.nomads_rate_limiter,
        retry_status_codes=nomads_mirror.NOMADS_RETRY_STATUS_CODES,
    )
    get_text = Mock(
        return_value='<a href="hrrr.t19z.wrfsfcf00.grib2">x</a> <a href="hrrr.t19z.wrfsfcf00.grib2.idx">y</a> <a href="hrrr.t19z.wrfsubhf00.grib2">z</a>'
    )
    monkeypatch.setattr(nomads_mirror, "httpx_get_text", get_text)
    assert nomads_mirror.list_nomads_directory("dir/") == {
        "hrrr.t19z.wrfsfcf00.grib2",
        "hrrr.t19z.wrfsfcf00.grib2.idx",
        "hrrr.t19z.wrfsubhf00.grib2",
    }
    get_text.assert_called_once_with(
        "dir/",
        rate_limiter=nomads_mirror.nomads_rate_limiter,
        retry_status_codes=nomads_mirror.NOMADS_RETRY_STATUS_CODES,
    )


# --- The operational surface ---


def test_operational_kubernetes_resources_is_one_hourly_mirror_cron() -> None:
    (cron_job,) = NoaaHrrrNomadsMirror().operational_kubernetes_resources("image")
    assert cron_job.name == "noaa-hrrr-nomads-mirror-gribs"
    assert len(cron_job.name) <= 52
    assert cron_job.schedule == "45 * * * *"
    assert cron_job.pod_active_deadline == timedelta(minutes=59)
    assert cron_job.command == ["mirror-gribs"]
    assert not cron_job.suspend


def test_cron_command_matches_a_registered_cli_command() -> None:
    mirror = NoaaHrrrNomadsMirror()
    command_names = {
        (command.name or command.callback.__name__).replace("_", "-")  # ty: ignore[unresolved-attribute]
        for command in mirror.get_cli().registered_commands
    }
    for cron_job in mirror.operational_kubernetes_resources("image"):
        assert cron_job.command[0] in command_names
    assert runner.invoke(mirror.get_cli(), ["--help"]).exit_code == 0


def test_mirror_window_is_anchored_to_the_scheduled_fire() -> None:
    (cron_job,) = NoaaHrrrNomadsMirror().operational_kubernetes_resources("image")
    assert isinstance(cron_job, CronJob)
    # A pod (re)started at 20:10 belongs to the 19:45 fire: init 19:00, poll from
    # 19:49, stop a minute before the 59-minute deadline.
    init_time, poll_start, deadline = mirror_window(
        pd.Timestamp("2026-09-08T20:10Z"), cron_job, poll_start_minutes=49
    )
    assert init_time == pd.Timestamp("2026-09-08T19:00Z")
    assert poll_start == pd.Timestamp("2026-09-08T19:49Z")
    assert deadline == pd.Timestamp("2026-09-08T20:43Z")
