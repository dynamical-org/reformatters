import logging
import time

# Configure logging to include UTC date and time
logging.basicConfig(
    format=f"%(asctime)s {logging.BASIC_FORMAT}",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.INFO,
)
# Ensure timestamps are in UTC
logging.Formatter.converter = time.gmtime

_root_logger = logging.getLogger()


def get_logger(name: str) -> logging.Logger:
    return _root_logger.getChild(name)
