import time
from typing import Any

import fsspec  # type: ignore
import zarr


def fsspec_apply(
    fs: fsspec.AbstractFileSystem,
    method: str,
    *args: object,
    max_attempts: int = 6,
    **kwargs: object,
) -> Any:
    """
    Apply a method to the filesystem with retry logic.

    This function handles both sync and async filesystems. The fsspec local filesystem is sync,
    but the fsspec store from zarr.storage.FsspecStore is async, so we need to handle both cases.
    (The AsyncFileSystem wrapper on LocalFilesystem raises NotImplementedError when _put is called
    so we can't just use that.)

    Args:
        fs: The filesystem to apply the method to
        method: Name of the method to call on the filesystem
        *args: Arguments to pass to the method
        max_attempts: Maximum number of attempts to make
        **kwargs: Keyword arguments to pass to the method
    """
    for attempt in range(max_attempts):
        try:
            if hasattr(fs, f"_{method}"):
                # Zarr's FsspecStore creates async fsspec filesystems, so use their sync method
                return zarr.core.sync.sync(
                    getattr(fs, f"_{method}")(*args, **kwargs), timeout=120
                )
            else:
                return getattr(fs, method)(*args, **kwargs)
        except Exception:
            if attempt == max_attempts - 1:  # Last attempt failed
                raise
            time.sleep(attempt)
            continue
