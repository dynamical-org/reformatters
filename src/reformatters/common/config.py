import os
from enum import StrEnum

import pydantic


class Env(StrEnum):
    dev = "dev"
    prod = "prod"
    test = "test"


class DynamicalConfig(pydantic.BaseModel):
    env: Env = Env(os.getenv("DYNAMICAL_ENV", "dev"))

    # Sentry-compatible errors DSN; points at Better Stack's errors application.
    sentry_dsn: str | None = os.getenv("DYNAMICAL_SENTRY_DSN")

    betterstack_source_token: str | None = os.getenv(
        "DYNAMICAL_BETTERSTACK_SOURCE_TOKEN"
    )
    betterstack_ingesting_host: str | None = os.getenv(
        "DYNAMICAL_BETTERSTACK_INGESTING_HOST"
    )

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

    @property
    def is_betterstack_logs_enabled(self) -> bool:
        return self.betterstack_source_token is not None


Config = DynamicalConfig()
