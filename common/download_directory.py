import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from common.config import Config


@contextmanager
def cd_into_download_directory() -> Iterator[Path]:
    """
    Changes current working directory into and yields a temporary directory,
    except in development mode when a consistent cache path is yielded to avoid re-downloading files.

    The changing directory nonsense is to work around eccodes grib index files breaking unless
    the process' current working direcory is the same as the location of the grib files it's reading.
    """
    previous_directory = os.getcwd()
    if Config.is_dev():
        directory = Path("data/download/").absolute()
        directory.mkdir(parents=True, exist_ok=True)
        os.chdir(directory)
        yield directory
        os.chdir(previous_directory)
    else:
        with tempfile.TemporaryDirectory() as directory_str:
            directory = Path(directory_str)
            os.chdir(directory)
            yield directory
            os.chdir(previous_directory)
