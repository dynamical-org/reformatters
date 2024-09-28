import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from common.config import Config


@contextmanager
def download_directory() -> Iterator[Path]:
    """Yields a temporary directory, except in development mode when a consistent cache path is yielded to avoid re-downloading files."""
    if Config.is_dev():
        yield Path("data/download/")
    else:
        with tempfile.TemporaryDirectory() as dir:
            yield Path(dir)
