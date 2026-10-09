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
    *,
    on_recovered_gcs_404: Callable[[str], None] | None = None,
) -> Path:
    """Try each source in order, falling back on missing-file / transient errors."""
    assert len(sources) > 0
    last_exc: Exception | None = None
    deferred_warning: str | None = None
    try:
        for source in sources:
            try:
                path = download_one(source)
            except FALLBACK_EXCEPTIONS as e:
                if deferred_warning is not None:
                    log.warning(deferred_warning)
                    deferred_warning = None
                detail = str(e)
                is_404 = "404 Not Found" in detail
                if is_404:
                    url = re.search(r"https?://\S+", detail)
                    detail = f"{type(e).__name__}: 404 Not Found"
                    if url:
                        detail += f" for {url[0]}"
                warning = (
                    f"ECMWF download from {source!r} failed, will fall back: {detail}"
                )
                if (
                    on_recovered_gcs_404 is not None
                    and tuple(sources) == ("gcs", "s3")
                    and source == "gcs"
                    and is_404
                ):
                    deferred_warning = warning
                else:
                    log.warning(warning)
                last_exc = e
            else:
                if deferred_warning is not None:
                    assert on_recovered_gcs_404 is not None
                    on_recovered_gcs_404(deferred_warning)
                    deferred_warning = None
                return path
    finally:
        if deferred_warning is not None:
            log.warning(deferred_warning)
    assert last_exc is not None
    raise last_exc
