import contextlib
import faulthandler
import multiprocessing
import os
from collections.abc import Sequence
from typing import Any

# Spawn new processes since fork isn't safe with threads
with contextlib.suppress(RuntimeError):  # skip if already set
    multiprocessing.set_start_method("spawn", force=True)

import sentry_sdk
import typer
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.typer import TyperIntegration
from sentry_sdk.types import Hint, Log

from reformatters.common import deploy as deploy_module
from reformatters.common import monitoring
from reformatters.common.config import Config
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.initialize_new_integration import initialize_new_integration
from reformatters.common.operational import (
    OperationalResources,
    register_run_monitor,
)
from reformatters.deployments.aws_us_east_1 import (
    CLUSTER_DYNAMICAL_DATASETS as AWS_US_EAST_1_DYNAMICAL_DATASETS,
)
from reformatters.deployments.aws_us_east_1 import (
    CLUSTER_OPERATIONAL_ARCHIVERS as AWS_US_EAST_1_OPERATIONAL_ARCHIVERS,
)
from reformatters.deployments.aws_us_west_2 import (
    CLUSTER_DYNAMICAL_DATASETS as AWS_US_WEST_2_DYNAMICAL_DATASETS,
)
from reformatters.deployments.aws_us_west_2 import (
    CLUSTER_OPERATIONAL_ARCHIVERS as AWS_US_WEST_2_OPERATIONAL_ARCHIVERS,
)

faulthandler.enable()


# Registry of all DynamicalDatasets.
DYNAMICAL_DATASETS: Sequence[DynamicalDataset[Any, Any]] = [
    *AWS_US_WEST_2_DYNAMICAL_DATASETS,
    *AWS_US_EAST_1_DYNAMICAL_DATASETS,
]

register_run_monitor(monitoring.monitor_cron)

if Config.is_sentry_enabled:
    cron_job_name = os.getenv("CRON_JOB_NAME")
    job_name = os.getenv("JOB_NAME")
    pod_name = os.getenv("POD_NAME")

    def before_log(log: Log, _hint: Hint) -> Log | None:
        if cron_job_name:
            log["attributes"]["cron_job_name"] = cron_job_name
        if job_name:
            log["attributes"]["job_name"] = job_name
        if pod_name:
            log["attributes"]["pod_name"] = pod_name
        return log

    sentry_sdk.init(
        dsn=Config.sentry_dsn,
        environment=Config.env.value,
        project_root="src/",
        in_app_include=["reformatters"],
        default_integrations=True,
        include_local_variables=True,
        event_scrubber=monitoring.CredentialScrubber(),
        # Connection idles cause us to lose events after quiet periods
        keep_alive=True,
        # Truncate long values so Sentry doesn't reject the event
        max_value_length=16 * 1024,
        before_send_log=before_log,
        integrations=[
            LoggingIntegration(capture_sentry_logs=True),
            TyperIntegration(),
        ],
    )
    sentry_sdk.set_tag("env", Config.env.value)
    sentry_sdk.set_tag("cron_job_name", cron_job_name)
    sentry_sdk.set_tag("job_name", job_name)
    sentry_sdk.set_tag("pod_name", pod_name)


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.callback()
def startup(ctx: typer.Context) -> None:
    ctx.call_on_close(monitoring.log_peak_memory)
    monitoring.install_sigterm_logger()


app.command()(initialize_new_integration)


OPERATIONAL_ARCHIVERS: Sequence[OperationalResources] = [
    *AWS_US_WEST_2_OPERATIONAL_ARCHIVERS,
    *AWS_US_EAST_1_OPERATIONAL_ARCHIVERS,
]

assert len(
    {resource.dataset_id for resource in (*DYNAMICAL_DATASETS, *OPERATIONAL_ARCHIVERS)}
) == len(DYNAMICAL_DATASETS) + len(OPERATIONAL_ARCHIVERS), (
    "Dataset and archiver IDs must be unique across the merged registry"
)

for dataset in DYNAMICAL_DATASETS:
    app.add_typer(dataset.get_cli(), name=dataset.dataset_id)

for archiver in OPERATIONAL_ARCHIVERS:
    app.add_typer(archiver.get_cli(), name=archiver.dataset_id)

deploy_module.register_commands(app, DYNAMICAL_DATASETS)


if not __debug__:
    raise RuntimeError(
        "This project relies on assert statements. Do not run with python -O."
    )

if __name__ == "__main__":
    app()
