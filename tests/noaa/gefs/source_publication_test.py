"""Checks GEFS's publication schedule against the constants the update relies on.

The schedule test hits noaa-gefs-pds, so it is slow. It fails when NCEP's timing
moves far enough that GEFS_PRE_EXTENSION_MAX or GEFS_EXTENSION_REQUEST_MIN_AGE no
longer describe the source.
"""

import pandas as pd
import pytest
from obstore.store import S3Store

from reformatters.noaa.gefs.forecast_35_day.dynamical_dataset import (
    GefsForecast35DayDataset,
)
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_EXTENSION_REQUEST_MIN_AGE,
    GEFS_INIT_TIME_FREQUENCY,
    GEFS_PRE_EXTENSION_MAX,
)
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG

_MEMBERS = ["gec00"] + [f"gep{member:02d}" for member in range(1, 31)]
_A_AND_B_PREFIXES = ("pgrb2ap5", "pgrb2bp5")

# Downloads for a cycle run from the cron time until the job finishes, well inside an
# hour of it. The next day's run asks for that cycle again a day later.
_REQUEST_WINDOW = pd.Timedelta(hours=1)


def _update_cron_offset() -> pd.Timedelta:
    """How long after a 00z init the operational update starts."""
    schedule = next(
        job.schedule
        for job in GefsForecast35DayDataset(
            primary_storage_config=NOOP_STORAGE_CONFIG
        ).operational_kubernetes_resources("test-image-tag")
        if job.name.endswith("-update")
    )
    minute, hour = schedule.split()[:2]
    return pd.Timedelta(hours=int(hour), minutes=int(minute))


def test_extension_request_age_splits_the_updates_inits() -> None:
    """The request age must gate the newest init and clear the one before it."""
    newest_age = _update_cron_offset()
    previous_age = newest_age + pd.Timedelta(days=1)

    assert newest_age < GEFS_EXTENSION_REQUEST_MIN_AGE < previous_age, (
        f"GEFS_EXTENSION_REQUEST_MIN_AGE={GEFS_EXTENSION_REQUEST_MIN_AGE} must fall "
        f"between the newest init ({newest_age}) and the previous one "
        f"({previous_age}) at cron time, or the update either requests an unpublished "
        f"extension or never requests it at all"
    )
    assert GEFS_INIT_TIME_FREQUENCY <= pd.Timedelta(days=1)


def _published_files(init_time: pd.Timestamp) -> pd.DataFrame:
    """Every a and b file of `init_time`, as (member, lead_time, published_at)."""
    store = S3Store("noaa-gefs-pds", region="us-east-1", skip_signature=True)
    date, hour = init_time.strftime("%Y%m%d"), init_time.strftime("%H")
    rows = []
    for sub in _A_AND_B_PREFIXES:
        for batch in store.list(f"gefs.{date}/{hour}/atmos/{sub}/", chunk_size=1000):
            for meta in batch:
                name = meta["path"].split("/")[-1]
                if not name.endswith(".idx") or ".f" not in name:
                    continue
                member = name.split(".")[0]
                if member not in _MEMBERS:
                    continue
                rows.append(
                    (
                        member,
                        pd.Timedelta(hours=int(name.split(".f")[-1][:-4])),
                        pd.Timestamp(meta["last_modified"]).tz_convert(None),
                    )
                )
    return pd.DataFrame(rows, columns=["member", "lead_time", "published_at"])


@pytest.mark.slow
def test_gefs_publishes_on_the_schedule_the_update_assumes() -> None:
    # Two days back: settled well past the extension, and recent enough to reflect
    # NCEP's current behavior.
    init_time = (pd.Timestamp.now() - pd.Timedelta(days=2)).floor("D")
    files = _published_files(init_time)
    assert not files.empty, f"no files listed for {init_time}"

    cron_offset = _update_cron_offset()
    age = files.published_at - init_time
    pre_extension = files[files.lead_time <= GEFS_PRE_EXTENSION_MAX]
    extension = files[files.lead_time > GEFS_PRE_EXTENSION_MAX]

    assert set(pre_extension.member) == set(_MEMBERS)
    assert set(extension.member) == set(_MEMBERS)

    # GEFS_PRE_EXTENSION_MAX is where the first wave ends: everything up to it lands
    # in one stretch shortly after init, rather than trickling in for many hours.
    first_wave_done = age[pre_extension.index].max()
    assert first_wave_done <= cron_offset + pd.Timedelta(hours=2), (
        f"lead times through {GEFS_PRE_EXTENSION_MAX} were still arriving at "
        f"init+{first_wave_done}, long after the update requests them"
    )

    # The extension is whole before the next day's run asks for it.
    extension_done = age[extension.index].max()
    next_update = cron_offset + pd.Timedelta(days=1)
    assert extension_done <= next_update, (
        f"the extension was still arriving at init+{extension_done}, after the next "
        f"update requests it at init+{next_update}; data past "
        f"{GEFS_PRE_EXTENSION_MAX} would be missed"
    )

    # Skipping the extension on the newest init gives up little. If NCEP gets much
    # faster this fails, and the update could start requesting it.
    requested_by = cron_offset + _REQUEST_WINDOW
    published_early = (age[extension.index] <= requested_by).mean()
    assert published_early < 0.25, (
        f"{published_early:.1%} of the extension was already published by "
        f"init+{requested_by}; the update is now skipping real data"
    )
