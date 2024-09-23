import os
from enum import Enum

from pydantic import BaseModel


class Env(str, Enum):
    dev = "dev"
    prod = "prod"


class DynamicalConfig(BaseModel):
    env: Env

    def is_dev(self) -> bool:
        return self.env == Env.dev


Config = DynamicalConfig(env=Env(os.environ.get("DYNAMICAL_ENV", "dev")))
