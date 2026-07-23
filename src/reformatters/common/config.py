import os
from enum import StrEnum

import pydantic


class Env(StrEnum):
    dev = "dev"
    prod = "prod"
    test = "test"


class DynamicalConfig(pydantic.BaseModel):
    env: Env = Env(os.getenv("DYNAMICAL_ENV", "dev"))

    # Sentry-protocol DSN for the Better Stack Errors application.
    errors_dsn: str | None = os.getenv("BETTERSTACK_ERRORS_DSN")

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
    def is_error_tracking_enabled(self) -> bool:
        return self.errors_dsn is not None


Config = DynamicalConfig()
