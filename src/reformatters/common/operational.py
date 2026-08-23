from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import Protocol

import typer

from reformatters.common.iterating import item
from reformatters.common.kubernetes import CronJob
from reformatters.common.pydantic import FrozenBaseModel


class OperationalResources(FrozenBaseModel):
    """A unit of work that runs on a schedule in kubernetes and reports each run.

    Subclassed by `DynamicalDataset` and by the source archivers that feed a dataset
    but have no store of their own. `dataset_id` is the CLI path segment a cron pod
    invokes, so a subclass's typer app must be registered under it.
    """

    @property
    def dataset_id(self) -> str:
        raise NotImplementedError("Subclasses must implement dataset_id")

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        raise NotImplementedError(
            "Subclasses must implement operational_kubernetes_resources"
        )

    def get_cli(self) -> typer.Typer:
        """The typer app registered under `dataset_id`, holding this unit's commands."""
        raise NotImplementedError("Subclasses must implement get_cli")

    def _operational_cron_job(
        self, cron_type: type[CronJob], cron_job_name: str | None = None
    ) -> CronJob:
        """The single cron job of `cron_type` (and name, when given) this unit defines."""
        return item(
            cron_job
            for cron_job in self.operational_kubernetes_resources(
                "placeholder-image-tag"
            )
            if isinstance(cron_job, cron_type)
            and (cron_job_name is None or cron_job.name == cron_job_name)
        )

    @contextmanager
    def _monitor(
        self,
        cron_type: type[CronJob],
        reformat_job_name: str,
        cron_job_name: str | None = None,
        *,
        send_in_progress: bool = True,
        send_result: bool = True,
    ) -> Iterator[None]:
        # No registered monitors -> nothing to report to, and no need to require
        # operational_kubernetes_resources to be defined.
        if not _RUN_MONITORS:
            yield
            return

        cron_job = self._operational_cron_job(cron_type, cron_job_name)

        with ExitStack() as stack:
            for monitor in _RUN_MONITORS:
                stack.enter_context(
                    monitor(
                        cron_job,
                        reformat_job_name,
                        send_in_progress=send_in_progress,
                        send_result=send_result,
                    )
                )
            yield


class RunMonitor(Protocol):
    """Wraps a single operational cron run to report it to a monitoring service.

    The application registers monitors (see `register_run_monitor`);
    `OperationalResources._monitor` enters every registered one around each run. This
    keeps the operational classes agnostic of any specific monitoring service — a
    different deployment registers whatever it uses, or nothing.
    """

    def __call__(
        self,
        cron_job: CronJob,
        reformat_job_name: str,
        *,
        send_in_progress: bool,
        send_result: bool,
    ) -> AbstractContextManager[None]: ...


_RUN_MONITORS: list[RunMonitor] = []


def register_run_monitor(monitor: RunMonitor) -> None:
    """Register a monitor to wrap every operational cron run. With none registered,
    monitoring is a no-op."""
    _RUN_MONITORS.append(monitor)
