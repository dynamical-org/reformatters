import os
import time
from dataclasses import dataclass

import httpx
import pandas as pd
from kubernetes.client.exceptions import ApiException
from urllib3.exceptions import HTTPError as KubernetesHTTPError

from reformatters.common import kubernetes
from reformatters.common.config import Config
from reformatters.common.logging import get_logger
from reformatters.common.pydantic import FrozenBaseModel
from reformatters.common.source_availability import SourceAvailability, Summary
from reformatters.common.types import Timedelta, Timestamp

log = get_logger(__name__)


class MaterializedUpdateTrigger(FrozenBaseModel):
    source_cronjob: str
    target_cronjob: str
    availability: SourceAvailability
    init_offset: Timedelta = pd.Timedelta(0)
    min_init_age: Timedelta = pd.Timedelta(0)

    def poller(self, newest_init: Timestamp) -> TriggerPoller | None:
        if Config.is_prod and os.getenv("CRON_JOB_NAME") == self.source_cronjob:
            return TriggerPoller(self, newest_init - self.init_offset)
        return None


@dataclass
class TriggerPoller:
    trigger: MaterializedUpdateTrigger
    init_time: Timestamp
    next_check: float = 0
    submitted: bool = False

    def pending(self) -> bool:
        if self.submitted or time.monotonic() < self.next_check:
            return not self.submitted
        self.next_check = time.monotonic() + 60
        if pd.Timestamp.now() - self.init_time < self.trigger.min_init_age:
            return True
        try:
            available = self.trigger.availability.available_inits(Summary.fetch())
        except httpx.HTTPError, ValueError:
            log.exception(
                "Could not check source readiness for %s", self.trigger.target_cronjob
            )
            return True
        if self.init_time not in available:
            return True
        name = (
            f"{self.trigger.target_cronjob.removesuffix('-update')}"
            f"-t{self.init_time:%y%m%d%H}-{self.trigger.availability.lead_hours}h"
        )
        assert len(name) <= 52, "Triggered Job name exceeds indexed Job name limit"
        try:
            if not kubernetes.has_running_cronjob_job(self.trigger.target_cronjob):
                kubernetes.create_job_from_cronjob(self.trigger.target_cronjob, name)
                self.submitted = True
        except ApiException, KubernetesHTTPError:
            log.exception("Could not submit materialized trigger %s", name)
        return not self.submitted
