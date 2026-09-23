import threading
from datetime import timedelta
from itertools import count
from pathlib import Path
from unittest.mock import Mock

import httpx
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


def test_mirror_keys_are_the_nodd_keys() -> None:
    assert coord(1).relative_path() == "hrrr.20260907/conus/hrrr.t19z.wrfsfcf01.grib2"


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


def _clock_ticks_per_listing(
    nomads: FakeNomads, monkeypatch: pytest.MonkeyPatch
) -> pd.Timestamp:
    """A fake clock that advances one second per NOMADS listing; returns its start."""
    start = pd.Timestamp("2026-09-07T19:49Z")
    listings_before = nomads.listings
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(
            lambda cls, *a, **k: (
                start + pd.Timedelta(seconds=nomads.listings - listings_before)
            )
        ),
    )
    monkeypatch.setattr(nomads_mirror.time, "sleep", lambda _s: None)
    return start


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
    start = _clock_ticks_per_listing(nomads, monkeypatch)
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

    f00, f01 = coord(0).relative_path(), coord(1).relative_path()
    assert sorted(result.copied) == [f00, f00 + ".idx", f01, f01 + ".idx"]
    # Within a pair the data file is copied before its index.
    assert result.copied.index(f00) < result.copied.index(f00 + ".idx")
    assert result.copied.index(f01) < result.copied.index(f01 + ".idx")
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
    obstore.put(mirror, coord(0).relative_path(), FIXTURE_GRIB.read_bytes())
    obstore.put(mirror, coord(0).relative_path() + ".idx", FIXTURE_INDEX.read_bytes())
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", FIXTURE_GRIB.read_bytes())
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2.idx", FIXTURE_INDEX.read_bytes())
    nomads.publish("hrrr.t19z.wrfsfcf01.grib2", FIXTURE_GRIB.read_bytes())

    result = run(nomads, mirror, polls=1, monkeypatch=monkeypatch)

    assert result.copied == [coord(1).relative_path()]
    assert result.pending == [coord(1).relative_path() + ".idx"]
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
    assert result.copied == [coord(0).relative_path()]


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
        coord(0, t).relative_path() + suffix
        for t in file_types
        for suffix in ("", ".idx")
    )


def test_a_data_file_cut_between_messages_is_replaced_when_its_index_disagrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    whole, index = FIXTURE_GRIB.read_bytes(), FIXTURE_INDEX.read_bytes()
    second_message_start = int(index.decode().splitlines()[1].split(":")[1])
    # NOMADS lists the file while it holds only its first message: whole GRIB2, but short.
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", whole[:second_message_start])
    result = run(nomads, mirror, polls=1, monkeypatch=monkeypatch, lead_hours=(0,))
    assert result.copied == [coord(0).relative_path()]

    # The index arrives naming both messages: the short copy is replaced first.
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", whole)
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2.idx", index)
    result = run(nomads, mirror, polls=1, monkeypatch=monkeypatch, lead_hours=(0,))
    assert result.copied == [coord(0).relative_path() + ".idx"]
    assert nomads.fetched == [
        "hrrr.t19z.wrfsfcf00.grib2",
        "hrrr.t19z.wrfsfcf00.grib2.idx",
        "hrrr.t19z.wrfsfcf00.grib2",
    ]
    assert obstore.get(mirror, coord(0).relative_path()).bytes().to_bytes() == whole


def test_an_index_disagreeing_with_a_file_mirrored_by_an_earlier_pod_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    whole, index = FIXTURE_GRIB.read_bytes(), FIXTURE_INDEX.read_bytes()
    second_message_start = int(index.decode().splitlines()[1].split(":")[1])
    obstore.put(mirror, coord(0).relative_path(), whole[:second_message_start])
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", whole[:second_message_start])
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2.idx", index)
    result = run(nomads, mirror, polls=1, monkeypatch=monkeypatch, lead_hours=(0,))
    # The index's last message starts past the mirrored file; NOMADS still serves
    # the short file, so nothing is exposed and the index stays pending.
    assert result.copied == []
    assert result.pending == [coord(0).relative_path() + ".idx"]


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
    get = Mock(
        return_value=Mock(
            text='<a href="hrrr.t19z.wrfsfcf00.grib2">x</a> <a href="hrrr.t19z.wrfsfcf00.grib2.idx">y</a> <a href="hrrr.t19z.wrfsubhf00.grib2">z</a>'
        )
    )
    monkeypatch.setattr(nomads_mirror, "httpx_get", get)
    assert nomads_mirror.list_nomads_directory("dir/") == {
        "hrrr.t19z.wrfsfcf00.grib2",
        "hrrr.t19z.wrfsfcf00.grib2.idx",
        "hrrr.t19z.wrfsubhf00.grib2",
    }
    get.assert_called_once_with(
        "dir/",
        rate_limiter=nomads_mirror.nomads_rate_limiter,
        retry_status_codes=nomads_mirror.NOMADS_RETRY_STATUS_CODES,
    )


# --- The operational surface ---


def test_mirror_store_is_signed_from_the_mounted_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[str] = []

    def load_secret(name: str) -> dict[str, object]:
        loaded.append(name)
        return {
            "endpoint_url": "https://account.r2.cloudflarestorage.com",
            "access_key_id": "key-id",
            "secret_access_key": "secret",
            "region": "auto",
            "force_path_style": True,
        }

    monkeypatch.setattr(nomads_mirror.kubernetes, "load_secret", load_secret)
    store = nomads_mirror.mirror_store()
    assert loaded == [nomads_mirror.MIRROR_SECRET_NAME]
    assert isinstance(store, obstore.store.S3Store)
    assert (
        store.config
        | {
            "bucket": "noaa-hrrr-nomads-mirror",
            "endpoint": "https://account.r2.cloudflarestorage.com",
            "access_key_id": "key-id",
            "secret_access_key": "secret",
            "region": "auto",
            "virtual_hosted_style_request": "false",
            "skip_signature": "false",
        }
        == store.config
    )


def test_operational_kubernetes_resources_is_one_hourly_mirror_cron() -> None:
    (cron_job,) = NoaaHrrrNomadsMirror().operational_kubernetes_resources("image")
    assert cron_job.name == "noaa-hrrr-nomads-mirror-gribs"
    assert len(cron_job.name) <= 52
    assert cron_job.schedule == "45 * * * *"
    assert cron_job.pod_active_deadline == timedelta(minutes=59)
    assert cron_job.command == ["mirror-gribs"]
    assert cron_job.secret_names == [nomads_mirror.MIRROR_SECRET_NAME]
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


def test_after_a_restart_the_mirrored_file_is_scanned_before_its_index_is_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    whole, index = FIXTURE_GRIB.read_bytes(), FIXTURE_INDEX.read_bytes()
    # An earlier pod mirrored one version of the file; NOMADS now serves another
    # whose index names an in-bounds but different second offset.
    obstore.put(mirror, coord(0).relative_path(), whole)
    lines = index.decode().splitlines()
    shifted = lines[1].split(":")
    shifted[1] = str(int(shifted[1]) - 1)
    shifted_index = (lines[0] + "\n" + ":".join(shifted) + "\n").encode()
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", whole)
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2.idx", shifted_index)
    result = run(nomads, mirror, polls=1, monkeypatch=monkeypatch, lead_hours=(0,))
    assert result.copied == []
    assert result.pending == [coord(0).relative_path() + ".idx"]
    # The data file was fetched again in case NOMADS had replaced it.
    assert nomads.fetched == [
        "hrrr.t19z.wrfsfcf00.grib2.idx",
        "hrrr.t19z.wrfsfcf00.grib2",
    ]

    nomads.publish("hrrr.t19z.wrfsfcf00.grib2.idx", index)
    result = run(nomads, mirror, polls=1, monkeypatch=monkeypatch, lead_hours=(0,))
    assert result.copied == [coord(0).relative_path() + ".idx"]


def test_earlier_init_times_are_caught_up_newest_first_in_one_poll_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    grib, index = FIXTURE_GRIB.read_bytes(), FIXTURE_INDEX.read_bytes()
    # 18Z is whole in the mirror, 17Z lacks its index, 16Z was never published.
    for name in ("hrrr.t18z.wrfsfcf00.grib2", "hrrr.t17z.wrfsfcf00.grib2"):
        obstore.put(mirror, f"hrrr.20260907/conus/{name}", grib)
        nomads.publish(name, grib)
    obstore.put(mirror, "hrrr.20260907/conus/hrrr.t18z.wrfsfcf00.grib2.idx", index)
    nomads.publish("hrrr.t17z.wrfsfcf00.grib2.idx", index)
    start = _clock_ticks_per_listing(nomads, monkeypatch)

    result = nomads_mirror.mirror_earlier_init_times(
        INIT,
        mirror,
        deadline=start + pd.Timedelta(seconds=10),
        init_times_back=3,
        file_types=("sfc",),
        lead_hours=(0,),
        list_directory=nomads.list_directory,
        fetch=nomads.fetch,
    )

    assert result.copied == ["hrrr.20260907/conus/hrrr.t17z.wrfsfcf00.grib2.idx"]
    assert result.pending == [
        "hrrr.20260907/conus/hrrr.t16z.wrfsfcf00.grib2",
        "hrrr.20260907/conus/hrrr.t16z.wrfsfcf00.grib2.idx",
    ]
    # No NOMADS request for the whole init, one listing for each of the others.
    assert nomads.listings == 2
    assert nomads.fetched == ["hrrr.t17z.wrfsfcf00.grib2.idx"]


def test_catch_up_stops_at_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    start = _clock_ticks_per_listing(nomads, monkeypatch)

    result = nomads_mirror.mirror_earlier_init_times(
        INIT,
        mirror,
        deadline=start + pd.Timedelta(seconds=1),
        init_times_back=3,
        file_types=("sfc",),
        lead_hours=(0,),
        list_directory=nomads.list_directory,
        fetch=nomads.fetch,
    )

    assert nomads.listings == 1
    assert "Deadline reached with 2 earlier inits not checked" in caplog.text
    assert result.pending == [
        "hrrr.20260907/conus/hrrr.t18z.wrfsfcf00.grib2",
        "hrrr.20260907/conus/hrrr.t18z.wrfsfcf00.grib2.idx",
    ]


def test_no_copy_starts_after_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", FIXTURE_GRIB.read_bytes())
    nomads.publish("hrrr.t19z.wrfsfcf01.grib2", FIXTURE_GRIB.read_bytes())
    start = pd.Timestamp("2026-09-07T19:49Z")
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(
            lambda cls, *a, **k: start + pd.Timedelta(minutes=len(nomads.fetched))
        ),
    )

    result = mirror_init_time(
        INIT,
        mirror,
        deadline=start + pd.Timedelta(seconds=30),
        file_types=("sfc",),
        lead_hours=(0, 1),
        poll_interval=timedelta(0),
        max_concurrent_copies=1,
        list_directory=nomads.list_directory,
        fetch=nomads.fetch,
    )

    assert result.copied == [coord(0).relative_path()]
    assert nomads.fetched == ["hrrr.t19z.wrfsfcf00.grib2"]


def test_a_partially_written_index_is_retried_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    index = FIXTURE_INDEX.read_bytes()
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", FIXTURE_GRIB.read_bytes())
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2.idx", index[: index.index(b":") + 3])
    result = run(nomads, mirror, polls=1, monkeypatch=monkeypatch, lead_hours=(0,))
    assert result.pending == [coord(0).relative_path() + ".idx"]

    nomads.publish("hrrr.t19z.wrfsfcf00.grib2.idx", index)
    result = run(nomads, mirror, polls=1, monkeypatch=monkeypatch, lead_hours=(0,))
    assert result.copied == [coord(0).relative_path() + ".idx"]


def test_data_files_copy_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", FIXTURE_GRIB.read_bytes())
    nomads.publish("hrrr.t19z.wrfsfcf01.grib2", FIXTURE_GRIB.read_bytes())
    both_downloading = threading.Barrier(2, timeout=5)
    fetch = nomads.fetch

    def fetch_once_both_started(url: str) -> Path:
        both_downloading.wait()
        return fetch(url)

    monkeypatch.setattr(nomads, "fetch", fetch_once_both_started)
    result = run(nomads, mirror, polls=1, monkeypatch=monkeypatch)
    assert sorted(result.copied) == [coord(0).relative_path(), coord(1).relative_path()]


def test_polling_continues_while_a_copy_is_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", FIXTURE_GRIB.read_bytes())
    f01_fetched = threading.Event()
    list_directory, fetch = nomads.list_directory, nomads.fetch

    def list_publishing_f01_later(url: str) -> set[str]:
        if nomads.listings == 1:
            nomads.publish("hrrr.t19z.wrfsfcf01.grib2", FIXTURE_GRIB.read_bytes())
        return list_directory(url)

    def fetch_f00_slowly(url: str) -> Path:
        if url.endswith("f01.grib2"):
            f01_fetched.set()
        else:
            # f01 is only listed by a poll made while this download is in flight.
            assert f01_fetched.wait(timeout=5)
        return fetch(url)

    monkeypatch.setattr(nomads, "list_directory", list_publishing_f01_later)
    monkeypatch.setattr(nomads, "fetch", fetch_f00_slowly)
    result = run(nomads, mirror, polls=10, monkeypatch=monkeypatch)
    assert sorted(result.copied) == [coord(0).relative_path(), coord(1).relative_path()]


def test_no_transfer_starts_after_the_deadline_within_a_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", FIXTURE_GRIB.read_bytes())
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2.idx", FIXTURE_INDEX.read_bytes())
    start = pd.Timestamp("2026-09-07T19:49Z")
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(
            lambda cls, *a, **k: start + pd.Timedelta(minutes=len(nomads.fetched))
        ),
    )

    result = mirror_init_time(
        INIT,
        mirror,
        deadline=start + pd.Timedelta(seconds=30),
        file_types=("sfc",),
        lead_hours=(0,),
        poll_interval=timedelta(0),
        list_directory=nomads.list_directory,
        fetch=nomads.fetch,
    )

    # The index fetch passed the deadline, so the data file was never fetched.
    assert nomads.fetched == ["hrrr.t19z.wrfsfcf00.grib2.idx"]
    assert result.copied == []
    assert mirror_contents(mirror) == set()


def test_a_failed_copy_admits_no_further_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", FIXTURE_GRIB.read_bytes())
    nomads.publish("hrrr.t19z.wrfsfcf01.grib2", FIXTURE_GRIB.read_bytes())
    fetched: list[str] = []

    def fetch_failing(url: str) -> Path:
        fetched.append(url)
        raise RuntimeError("R2 rejected the write")

    _clock_ticks_per_listing(nomads, monkeypatch)
    with pytest.raises(RuntimeError, match="R2 rejected"):
        mirror_init_time(
            INIT,
            mirror,
            deadline=pd.Timestamp("2026-09-07T19:49Z") + pd.Timedelta(seconds=10),
            file_types=("sfc",),
            lead_hours=(0, 1),
            poll_interval=timedelta(0),
            max_concurrent_copies=1,
            list_directory=nomads.list_directory,
            fetch=fetch_failing,
        )
    assert len(fetched) == 1


def test_waits_on_copies_without_spinning_once_all_are_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", FIXTURE_GRIB.read_bytes())
    _clock_ticks_per_listing(nomads, monkeypatch)
    timeouts: list[float | None] = []
    real_wait = nomads_mirror.wait

    def recording_wait(
        futures: object, timeout: float | None, return_when: str
    ) -> object:
        timeouts.append(timeout)
        return real_wait(futures, timeout=timeout, return_when=return_when)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(nomads_mirror, "wait", recording_wait)
    result = mirror_init_time(
        INIT,
        mirror,
        deadline=pd.Timestamp("2026-09-07T19:49Z") + pd.Timedelta(seconds=10),
        file_types=("sfc",),
        lead_hours=(0,),
        poll_interval=timedelta(0),
        list_directory=nomads.list_directory,
        fetch=nomads.fetch,
    )
    assert result.copied == [coord(0).relative_path()]
    assert timeouts == [None]


def test_one_listing_drains_more_pairs_than_the_pool_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    for lead in range(3):
        nomads.publish(f"hrrr.t19z.wrfsfcf{lead:02d}.grib2", FIXTURE_GRIB.read_bytes())
    start = _clock_ticks_per_listing(nomads, monkeypatch)
    result = mirror_init_time(
        INIT,
        mirror,
        deadline=start + pd.Timedelta(seconds=10),
        file_types=("sfc",),
        lead_hours=(0, 1, 2),
        poll_interval=timedelta(0),
        max_polls=1,
        max_concurrent_copies=1,
        list_directory=nomads.list_directory,
        fetch=nomads.fetch,
    )
    assert sorted(result.copied) == [coord(lead).relative_path() for lead in range(3)]
    assert nomads.listings == 1


def _unserved(status_code: int | None) -> Exception:
    """What a NOMADS fetch raises for a listed file it does not serve: a status error,
    or (None) a body cut short on every retry."""
    request = httpx.Request("GET", "https://nomads.ncep.noaa.gov/x")
    if status_code is None:
        return httpx.RemoteProtocolError("peer closed connection", request=request)
    return httpx.HTTPStatusError(
        f"{status_code}", request=request, response=httpx.Response(status_code)
    )


@pytest.mark.parametrize("status_code", [404, None])
@pytest.mark.parametrize(
    "unserved_name", ["hrrr.t19z.wrfsfcf00.grib2", "hrrr.t19z.wrfsfcf00.grib2.idx"]
)
def test_a_listed_file_nomads_does_not_serve_is_retried_next_poll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status_code: int | None,
    unserved_name: str,
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", FIXTURE_GRIB.read_bytes())
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2.idx", FIXTURE_INDEX.read_bytes())
    serve = nomads.fetch
    failures = iter([_unserved(status_code)])

    def fetch_once_unserved(url: str) -> Path:
        if url.endswith("/" + unserved_name) and (failure := next(failures, None)):
            raise failure
        return serve(url)

    monkeypatch.setattr(nomads, "fetch", fetch_once_unserved)
    result = run(nomads, mirror, polls=2, monkeypatch=monkeypatch, lead_hours=(0,))

    f00 = coord(0).relative_path()
    assert sorted(result.copied) == [f00, f00 + ".idx"]
    assert result.pending == []
    assert nomads.listings == 2


def test_a_nomads_error_other_than_not_served_still_fails_the_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nomads = FakeNomads(tmp_path)
    mirror = obstore.store.LocalStore(tmp_path / "mirror", mkdir=True)
    nomads.publish("hrrr.t19z.wrfsfcf00.grib2", FIXTURE_GRIB.read_bytes())

    def forbidden(_url: str) -> Path:
        raise _unserved(403)

    monkeypatch.setattr(nomads, "fetch", forbidden)
    with pytest.raises(httpx.HTTPStatusError, match="403"):
        run(nomads, mirror, polls=1, monkeypatch=monkeypatch, lead_hours=(0,))
