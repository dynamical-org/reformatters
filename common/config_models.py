from collections.abc import Sequence
from typing import Annotated, Any, Generic, Literal, TypeVar

import numcodecs  # type: ignore
import pydantic

from common.types import TimedeltaUnits, TimestampUnits

B = TypeVar("B", bound=pydantic.BaseModel)


def replace(obj: B, **kwargs: Any) -> B:
    """Replace properties of pydantic model instances."""
    # From https://github.com/pydantic/pydantic/discussions/3352#discussioncomment-10531773
    # pydantic's model_copy(update=...) does not validate updates, this function does.
    return type(obj).model_validate(obj.model_dump() | kwargs)


class FrozenBaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        frozen=True, strict=True, revalidate_instances="always"
    )


class DatasetAttributes(FrozenBaseModel):
    dataset_id: str
    name: str
    description: str
    attribution: str
    spatial_domain: str
    spatial_resolution: str
    time_domain: str
    time_resolution: str
    forecast_domain: str
    forecast_resolution: str


class StatisticsApproximate(FrozenBaseModel):
    min: str | int | float
    max: str | int | float


type EnsembleStatistic = Literal["avg"]  # "spr" (spread) is also available


class DataVarAttrs(FrozenBaseModel):
    long_name: str
    short_name: str
    standard_name: str | None = None
    units: str
    step_type: Literal["instant", "accum", "avg", "min", "max"]
    ensemble_statistic: EnsembleStatistic | None = None


class CoordinateAttrs(FrozenBaseModel):
    units: TimestampUnits | TimedeltaUnits | str
    statistics_approximate: StatisticsApproximate


# numcodecs.zarr3 Codec wrappers are autogenerated and don't round trip
# in pydantic natively. Convert to dicts which is a fine format to store them.
def codecs_to_dicts(
    codecs: Sequence[numcodecs.abc.Codec],
) -> Sequence[dict[str, Any]]:
    if codecs is None:
        return None
    return [codec.to_dict() if hasattr(codec, "to_dict") else codec for codec in codecs]


class Encoding(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,  # allow numcodecs.abc.Codec values
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )

    # Could be any np.typing.DTypeLike but that type is loose and allows any string.
    # It's fine to add any valid dtype string to this literal.
    dtype: Literal["float32", "float64", "uint16", "int64", "bool"]
    chunks: tuple[int, ...] | int

    fill_value: float | int | bool

    filters: Annotated[
        Sequence[dict[str, Any]] | None,
        pydantic.BeforeValidator(codecs_to_dicts),
    ] = None
    compressors: Sequence[dict[str, Any]] | None = None

    calendar: Literal["proleptic_gregorian"] | None = None  # For timestamps only
    # The _encoded_ units, for timestamps and timedeltas only
    # Decoded units for all variables are in DataVarAttrs
    units: TimestampUnits | TimedeltaUnits | None = None

    add_offset: float | None = None
    scale_factor: float | None = None


class Coordinate(FrozenBaseModel):
    name: str
    encoding: Encoding
    attrs: CoordinateAttrs


INTERNAL_ATTRS = TypeVar("INTERNAL_ATTRS", bound=pydantic.BaseModel)


class DataVar(FrozenBaseModel, Generic[INTERNAL_ATTRS]):
    name: str
    encoding: Encoding
    attrs: DataVarAttrs
    internal_attrs: INTERNAL_ATTRS
