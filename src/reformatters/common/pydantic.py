from typing import Any, TypeVar

import pydantic

B = TypeVar("B", bound=pydantic.BaseModel)


def replace(obj: B, **kwargs: Any) -> B:  # noqa: ANN401
    """Replace properties of pydantic model instances."""
    # From https://github.com/pydantic/pydantic/discussions/3352#discussioncomment-10531773
    # pydantic's model_copy(update=...) does not validate updates, this function does.
    return type(obj).model_validate(obj.model_dump() | kwargs)


class FrozenBaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        frozen=True, strict=True, revalidate_instances="always"
    )


class FrozenArbitraryBaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        frozen=True,
        strict=True,
        arbitrary_types_allowed=True,
        revalidate_instances="always",
    )
