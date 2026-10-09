from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from reformatters.common.download import FALLBACK_EXCEPTIONS

type EcmwfOpenDataSource = Literal["s3", "gcs"]


def ecmwf_download_with_fallback(
    sources: Sequence[EcmwfOpenDataSource],
    download_one: Callable[[EcmwfOpenDataSource], Path],
) -> Path:
    """Try each source in order, falling back on missing-file / transient errors."""
    assert len(sources) > 0
    last_exc: Exception | None = None
    for source in sources:
        try:
            return download_one(source)
        except FALLBACK_EXCEPTIONS as e:
            last_exc = e
    assert last_exc is not None
    raise last_exc
