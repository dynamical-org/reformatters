from collections.abc import Hashable, Sequence
from typing import Any, Literal, NotRequired, TypedDict

import numcodecs  # type: ignore
import pydantic

# We pull data from 3 types of source files: `a`, `b` and `s`.
# Selected variables are available in `s` at higher resolution (0.25 vs 0.5 deg)
# but `s` stops after forecast lead time 240h at which point the variable is still in `a` or `b`.
type NoaaFileType = Literal["a", "b", "s+a", "s+b"]


class FrozenBaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)


class FrozenTypedDict(TypedDict):
    def __setitem__(self, _key: Hashable, _value: Any) -> None:  # type: ignore
        raise TypeError("Dict is frozen")


class DataVarAttrs(FrozenBaseModel):
    long_name: str
    standard_name: str
    units: str
    step_type: Literal["instant", "accum"]


class InternalAttrs(FrozenBaseModel):
    grib_element: str
    grib_description: str
    grib_index_level: str
    noaa_file_type: NoaaFileType


class Encoding(FrozenTypedDict):
    __pydantic_config__: NotRequired[pydantic.ConfigDict] = pydantic.ConfigDict(  # type: ignore
        extra="forbid",
        arbitrary_types_allowed=True,  # allow numcodecs.abc.Codec values
    )

    # Could be any np.typing.DTypeLike but that type is loose and allows any string.
    # It's fine to add any valid dtype string to this literal.
    dtype: Literal["float32", "float64", "uint16", "int64", "bool"]
    chunks: tuple[int, ...] | int

    filters: NotRequired[Sequence[numcodecs.abc.Codec]]
    compressor: NotRequired[numcodecs.abc.Codec]

    calendar: NotRequired[Literal["proleptic_gregorian"]]  # For timestamps only
    # The _encoded_ units, for timestamps and timedeltas only
    # Decoded units for all variables are in DataVarAttrs
    units: NotRequired[Literal["seconds", "seconds since 1970-01-01 00:00:00"]]

    add_offset: NotRequired[float]
    scale_factor: NotRequired[float]


class Coordinate(FrozenBaseModel):
    name: str
    encoding: Encoding


class DataVar(FrozenBaseModel):
    name: str
    encoding: Encoding
    attrs: DataVarAttrs
    internal_attrs: InternalAttrs
