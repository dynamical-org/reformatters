from collections.abc import Sequence
from typing import Any, Literal, TypeVar

import numcodecs  # type: ignore
import pydantic

from common.types import TimedeltaUnits, TimestampUnits

# We pull data from 3 types of source files: `a`, `b` and `s`.
# Selected variables are available in `s` at higher resolution (0.25 vs 0.5 deg)
# but `s` stops after forecast lead time 240h at which point the variable is still in `a` or `b`.
type NoaaFileType = Literal["a", "b", "s+a", "s+b"]

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


class InternalAttrs(FrozenBaseModel):
    grib_element: str
    grib_description: str
    grib_index_level: str
    noaa_file_type: NoaaFileType
    index_position: int
    include_lead_time_suffix: bool = False


class CoordinateAttrs(FrozenBaseModel):
    units: TimestampUnits | TimedeltaUnits | str
    statistics_approximate: StatisticsApproximate


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

    filters: Sequence[numcodecs.abc.Codec] | None = None
    compressor: numcodecs.abc.Codec | None = None

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


class DataVar(FrozenBaseModel):
    name: str
    encoding: Encoding
    attrs: DataVarAttrs
    internal_attrs: InternalAttrs
