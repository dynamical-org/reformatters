from collections.abc import MutableMapping

import numpy as np
import pandas as pd

type StoreLike = MutableMapping[str, bytes] | str
type DatetimeLike = pd.Timestamp | np.datetime64 | str
type Array1D[D: np.generic] = np.ndarray[tuple[int], np.dtype[D]]
