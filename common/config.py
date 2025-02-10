import os
from enum import Enum

import pydantic


class Env(str, Enum):
    dev = "dev"
    prod = "prod"


class SourceCoopConfig(pydantic.BaseModel):
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None


class DynamicalConfig(pydantic.BaseModel):
    env: Env = Env(os.getenv("DYNAMICAL_ENV", "dev"))

    sentry_dsn: str | None = os.getenv("DYNAMICAL_SENTRY_DSN")

    source_coop: SourceCoopConfig = pydantic.Field(
        default_factory=lambda: SourceCoopConfig(
            aws_access_key_id=os.environ.get("DYNAMICAL_SOURCE_COOP_AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get(
                "DYNAMICAL_SOURCE_COOP_AWS_SECRET_ACCESS_KEY"
            ),
        ),
        validate_default=True,
        json_schema_extra={
            "description": "Source Coop credentials required in prod environment"
        },
    )

    @pydantic.model_validator(mode="after")
    def validate_source_coop_credentials(self) -> "DynamicalConfig":
        if self.env == Env.prod:
            if not (
                self.source_coop.aws_access_key_id
                and self.source_coop.aws_secret_access_key
            ):
                raise ValueError(
                    "aws_access_key_id and aws_secret_access_key are required in prod environment"
                )
        return self

    @property
    def is_dev(self) -> bool:
        return self.env == Env.dev

    @property
    def is_sentry_enabled(self) -> bool:
        return self.sentry_dsn is not None


Config = DynamicalConfig()
