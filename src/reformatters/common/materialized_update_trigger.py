from reformatters.common import kubernetes
from reformatters.common.iterating import digest
from reformatters.common.logging import get_logger
from reformatters.common.pydantic import FrozenBaseModel
from reformatters.common.types import Timestamp

log = get_logger(__name__)


class MaterializedUpdateTrigger(FrozenBaseModel):
    target_cronjob: str
    init_hours: tuple[int, ...] = tuple(range(24))

    def job_name(self, unit: Timestamp) -> str:
        prefix = self.target_cronjob.removesuffix("-update")
        suffix = f"-t{unit:%Y%m%d%H%M}z"
        if len(prefix) + len(suffix) > 52:
            prefix = f"{prefix[: 52 - len(suffix) - 5].rstrip('-')}-{digest([self.target_cronjob], length=4)}"
        return prefix + suffix

    def dispatch(self, unit: Timestamp) -> None:
        if unit.hour in self.init_hours:
            name = self.job_name(unit)
            try:
                kubernetes.create_job_from_cronjob(
                    self.target_cronjob, name, kind="trigger"
                )
            except Exception:
                log.exception("Could not submit materialized trigger %s", name)
