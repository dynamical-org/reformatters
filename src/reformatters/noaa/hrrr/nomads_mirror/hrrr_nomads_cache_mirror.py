import time
from collections.abc import Sequence
from datetime import timedelta
from typing import Annotated

import pandas as pd
import typer

from reformatters.common.kubernetes import CronJob
from reformatters.common.logging import get_logger
from reformatters.common.operational import OperationalResources
from reformatters.noaa.hrrr.nomads_cache import (
    NOMADS_CACHE_SECRET_NAME,
    nomads_cache_store,
)
from reformatters.noaa.hrrr.nomads_mirror.mirror import mirror_init_time

log = get_logger(__name__)


def mirror_window(
    now: pd.Timestamp, cron_job: CronJob, poll_start_minutes: int
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    fire = cron_job.previous_fire_time(now)
    init_time = fire.floor("1h")
    return (
        init_time,
        init_time + timedelta(minutes=poll_start_minutes),
        fire + cron_job.pod_active_deadline - timedelta(minutes=1),
    )


class NoaaHrrrNomadsCacheMirror(OperationalResources):
    @property
    def dataset_id(self) -> str:
        return "noaa-hrrr-nomads-cache"

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        return [
            CronJob(
                command=["mirror-gribs"],
                workers_total=1,
                parallelism=1,
                name=f"{self.dataset_id}-mirror-gribs",
                schedule="45 * * * *",
                pod_active_deadline=timedelta(minutes=59),
                image=image_tag,
                dataset_id=self.dataset_id,
                cpu="1",
                memory="2G",
                ephemeral_storage="4G",
                secret_names=[NOMADS_CACHE_SECRET_NAME],
            )
        ]

    def mirror_gribs(
        self,
        reformat_job_name: Annotated[str, typer.Argument(envvar="JOB_NAME")],
        poll_start_minutes: int = 49,
    ) -> None:
        with self._monitor(
            CronJob, reformat_job_name, cron_job_name=f"{self.dataset_id}-mirror-gribs"
        ):
            now = pd.Timestamp.now("UTC")
            init_time, poll_start, deadline = mirror_window(
                now, self._operational_cron_job(CronJob), poll_start_minutes
            )
            wait = max(0, (poll_start - now).total_seconds())
            log.info(f"Waiting {wait:.2f} seconds to mirror {init_time}")
            time.sleep(wait)
            result = mirror_init_time(
                init_time, nomads_cache_store(write=True), deadline=deadline
            )
            log.info(f"Mirror result: {result}")

    def get_cli(self) -> typer.Typer:
        app = typer.Typer()
        app.command()(self.mirror_gribs)
        return app
