from datetime import datetime
from enum import Enum, auto
from typing import Annotated, Any, Literal, get_args

import numpy as np
import pandas as pd
import pydantic

type DatetimeLike = pd.Timestamp | np.datetime64 | datetime | str
type Timestamp = Annotated[
    pd.Timestamp,
    pydantic.PlainValidator(pd.Timestamp),
]
type Timedelta = Annotated[
    pd.Timedelta,
    pydantic.PlainValidator(pd.Timedelta),
]
type ArrayFloat32 = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
type ArrayInt16 = np.ndarray[tuple[int, ...], np.dtype[np.int16]]
type ArrayND[D: np.generic] = np.ndarray[tuple[int, ...], np.dtype[D]]
type Array1D[D: np.generic] = np.ndarray[tuple[int], np.dtype[D]]
type Array2D[D: np.generic] = np.ndarray[tuple[int, int], np.dtype[D]]

# A numcodecs/zarr codec serialized to its config dict (Codec.to_dict()); an
# Encoding filter, compressor, or serializer.
type CodecConfig = dict[str, Any]

type TimestampUnits = Literal["seconds"]
type TimedeltaUnits = Literal["seconds since 1970-01-01 00:00:00"]


type Dim = Literal[
    "time",
    "init_time",
    "ensemble_member",
    "lead_time",
    "latitude",
    "longitude",
    "x",
    "y",
    "statistic",
    # Vertical group dimensions (a group's name equals its dimension name, see VerticalGroup).
    "pressure_level",
    "model_level",
]
type AppendDim = Literal["init_time", "time"]
assert set(get_args(AppendDim.__value__)) <= set(get_args(Dim.__value__))


class RootGroup(Enum):
    ROOT = auto()  # pure sentinel; the root zarr group has no name


ROOT = RootGroup.ROOT

# A variable on a dense, comparable vertical dimension lives in a zarr group named
# after that dimension (group name == dimension name). Expand as new types are added.
type VerticalGroup = Literal["pressure_level", "model_level"]
# A variable's group: ROOT (single-level, lives at the dataset root) or a vertical group.
type Group = VerticalGroup | RootGroup
assert set(get_args(VerticalGroup.__value__)) <= set(get_args(Dim.__value__))
