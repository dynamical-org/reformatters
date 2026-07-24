import os
from enum import StrEnum

import pydantic


class Env(StrEnum):
    dev = "dev"
    prod = "prod"
    test = "test"


def _errors_dsn() -> str | None:
    # Prefer the Better Stack Errors DSN (Sentry-protocol compatible); fall back to
    # the Sentry DSN so unsetting BETTERSTACK_ERRORS_DSN reverts error tracking to Sentry.
    return os.getenv("BETTERSTACK_ERRORS_DSN") or os.getenv("DYNAMICAL_SENTRY_DSN")


class DynamicalConfig(pydantic.BaseModel):
    env: Env = Env(os.getenv("DYNAMICAL_ENV", "dev"))

    sentry_dsn: str | None = _errors_dsn()

    @property
    def is_test(self) -> bool:
        return self.env == Env.test

    @property
    def is_dev(self) -> bool:
        return self.env == Env.dev

    @property
    def is_prod(self) -> bool:
        return self.env == Env.prod

    @property
    def is_sentry_enabled(self) -> bool:
        return self.sentry_dsn is not None


Config = DynamicalConfig()
