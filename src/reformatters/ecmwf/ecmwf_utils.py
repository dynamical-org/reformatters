import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from reformatters.common.download import FALLBACK_EXCEPTIONS
from reformatters.common.logging import get_logger

log = get_logger(__name__)

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
            detail = str(e)
            if isinstance(e, FileNotFoundError) and (
                match := re.match(
                    r"Object at location [^\n]+ not found: Error performing GET (https?://\S+)"
                    r" in \S+ - Server returned non-2xx status code: 404 Not Found:",
                    detail,
                )
            ):
                detail = f"{type(e).__name__}: 404 Not Found for {match[1]}"
            log.warning(
                f"ECMWF download from {source!r} failed, will fall back: {detail}"
            )
            last_exc = e
    assert last_exc is not None
    raise last_exc
