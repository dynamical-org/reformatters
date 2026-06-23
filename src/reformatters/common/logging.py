import logging
import os
import time

# Configure logging to include UTC date and time
logging.basicConfig(
    format=f"%(asctime)s {logging.BASIC_FORMAT}",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.INFO,
)
# Ensure timestamps are in UTC
logging.Formatter.converter = time.gmtime
# Silence httpx per-request INFO logs; warnings and errors still surface
logging.getLogger("httpx").setLevel(logging.WARNING)

_root_logger = logging.getLogger()


def get_logger(name: str) -> logging.Logger:
    return _root_logger.getChild(name)


class _KubernetesContextFilter(logging.Filter):
    """Enrich every record with the k8s job context so logs are filterable in Better Stack."""

    def __init__(self) -> None:
        super().__init__()
        self._context = {
            key: value
            for key in ("cron_job_name", "job_name", "pod_name")
            if (value := os.getenv(key.upper())) is not None
        }

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in self._context.items():
            setattr(record, key, value)
        return True


def add_betterstack_log_handler(
    source_token: str, ingesting_host: str | None = None
) -> None:
    # Imported lazily so this widely-imported module doesn't pull in logtail unless enabled.
    from logtail import LogtailHandler  # noqa: PLC0415

    host_kwarg = {"host": ingesting_host} if ingesting_host is not None else {}
    handler = LogtailHandler(source_token=source_token, **host_kwarg)
    handler.addFilter(_KubernetesContextFilter())
    _root_logger.addHandler(handler)
