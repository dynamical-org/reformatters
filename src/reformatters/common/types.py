from typing import Annotated, Literal, get_args

import numpy as np
import pandas as pd
import pydantic

type DatetimeLike = pd.Timestamp | np.datetime64 | str
type Timestamp = Annotated[
    pd.Timestamp,
    pydantic.PlainValidator(pd.Timestamp),
]
type Timedelta = Annotated[
    pd.Timedelta,
    pydantic.PlainValidator(pd.Timedelta),
]
type ArrayFloat32 = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
type Array1D[D: np.generic] = np.ndarray[tuple[int], np.dtype[D]]
type Array2D[D: np.generic] = np.ndarray[tuple[int, int], np.dtype[D]]

type TimestampUnits = Literal["seconds"]
type TimedeltaUnits = Literal["seconds since 1970-01-01 00:00:00"]


type Dim = Literal[
    "time", "init_time", "ensemble_member", "lead_time", "latitude", "longitude"
]
type AppendDim = Literal["init_time", "time"]
assert set(get_args(AppendDim)) <= set(get_args(Dim))
