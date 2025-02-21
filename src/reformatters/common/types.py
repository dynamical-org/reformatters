from typing import Literal

import numpy as np
import pandas as pd

type DatetimeLike = pd.Timestamp | np.datetime64 | str
type ArrayFloat32 = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
type Array1D[D: np.generic] = np.ndarray[tuple[int], np.dtype[D]]
type Array2D[D: np.generic] = np.ndarray[tuple[int, int], np.dtype[D]]

type TimestampUnits = Literal["seconds"]
type TimedeltaUnits = Literal["seconds since 1970-01-01 00:00:00"]
