import logging
import os

from logtail import LogtailHandler

# Every reformatters logger is a child of this one (get_logger returns a child of
# the root named "reformatters.<module>"), so attaching here streams our records
# to Better Stack without the chatter other libraries log to the root.
REFORMATTERS_LOGGER_NAME = "reformatters"

# Kubernetes context injected as env vars by common/kubernetes.py, emitted as
# structured fields so Better Stack surfaces them as queryable tags. cron_job_name
# is not derivable from pod metadata, so it must travel with the log.
_CONTEXT_FIELD_ENV_VARS = {
    "cron_job_name": "CRON_JOB_NAME",
    "job_name": "JOB_NAME",
    "pod_name": "POD_NAME",
    "env": "DYNAMICAL_ENV",
}


class _ContextFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self._context = {
            field: value
            for field, env_var in _CONTEXT_FIELD_ENV_VARS.items()
            if (value := os.getenv(env_var)) is not None
        }

    def filter(self, record: logging.LogRecord) -> bool:
        for field, value in self._context.items():
            setattr(record, field, value)
        return True


def attach_logtail() -> None:
    """Stream reformatters logs to the Better Stack source when the source token
    and ingesting host are set. Idempotent; a no-op when they are absent (local
    runs) so the only way logs stop reaching Better Stack is unsetting the env."""
    token = os.getenv("BETTERSTACK_SOURCE_TOKEN")
    host = os.getenv("BETTERSTACK_INGESTING_HOST")
    if not (token and host):
        return

    logger = logging.getLogger(REFORMATTERS_LOGGER_NAME)
    if any(isinstance(handler, LogtailHandler) for handler in logger.handlers):
        return

    handler = LogtailHandler(source_token=token, host=f"https://{host}")
    handler.setLevel(logging.INFO)
    handler.addFilter(_ContextFilter())
    logger.addHandler(handler)
