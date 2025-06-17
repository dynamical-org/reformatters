import os
from enum import StrEnum

import pydantic


class Env(StrEnum):
    dev = "dev"
    prod = "prod"


class DynamicalConfig(pydantic.BaseModel):
    env: Env = Env(os.getenv("DYNAMICAL_ENV", "dev"))

    sentry_dsn: str | None = os.getenv("DYNAMICAL_SENTRY_DSN")

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
