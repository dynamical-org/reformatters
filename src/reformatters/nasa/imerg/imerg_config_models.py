from typing import Literal

from reformatters.common.config_models import BaseInternalAttrs, DataVar

type ImergRun = Literal["early", "late"]


class NasaImergInternalAttrs(BaseInternalAttrs):
    # GDAL HDF5 subdataset path of the source field (note the double leading slash,
    # as GDAL exposes it), e.g. "//Grid/precipitation".
    h5_path: str
    # Exact sentinel the source stores for missing pixels; masked to NaN on read.
    source_fill_value: float
    # Multiplier from source units to the variable's output units, applied on read as
    # a plain scalar multiply (equivalent to a zarr ScaleOffset codec with scale =
    # 1 / source_scale). e.g. mm/hr -> kg m-2 s-1 is 1/3600; 1.0 leaves values unchanged.
    source_scale: float = 1.0


class NasaImergDataVar(DataVar[NasaImergInternalAttrs]):
    pass
