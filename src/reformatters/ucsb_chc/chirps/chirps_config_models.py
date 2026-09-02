from typing import Literal

from reformatters.common.config_models import BaseInternalAttrs, DataVar

type ChirpsProduct = Literal["final", "preliminary"]


class UcsbChcChirpsDataVar(DataVar[BaseInternalAttrs]):
    pass
