from collections.abc import Sequence
from pathlib import Path

from reformatters.common.download import RateLimiter, httpx_download_to_disk

# Documented NOMADS rate limit is 120 requests/minute
# When you get rate limited it's hard to get out, so set a little under.
nomads_rate_limiter = RateLimiter(max_per_minute=100)

# Retry on server errors, rate limits, and redirects (Akamai bot mitigation returns 302)
NOMADS_RETRY_STATUS_CODES = {302, 429, 500, 502, 503, 504}


def nomads_download_to_disk(
    url: str,
    dataset_id: str,
    byte_ranges: tuple[Sequence[int], Sequence[int]] | None = None,
    local_path_suffix: str = "",
    disk_cache: bool = False,
) -> Path:
    """`httpx_download_to_disk` through the shared NOMADS rate limiter and retry codes."""
    return httpx_download_to_disk(
        url,
        dataset_id,
        byte_ranges=byte_ranges,
        local_path_suffix=local_path_suffix,
        disk_cache=disk_cache,
        rate_limiter=nomads_rate_limiter,
        retry_status_codes=NOMADS_RETRY_STATUS_CODES,
    )
