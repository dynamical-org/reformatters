from reformatters.common.config_models import DataVar, INTERNAL_ATTRS_co
from reformatters.common.download import RateLimiter

# Documented NOMADS rate limit is 120 requests/minute
nomads_rate_limiter = RateLimiter(max_per_minute=150)

# Retry on server errors, rate limits, and redirects (Akamai bot mitigation returns 302)
NOMADS_RETRY_STATUS_CODES = {302, 429, 500, 502, 503, 504}


def has_hour_0_values(data_var: DataVar[INTERNAL_ATTRS_co]) -> bool:
    return data_var.attrs.step_type == "instant"
