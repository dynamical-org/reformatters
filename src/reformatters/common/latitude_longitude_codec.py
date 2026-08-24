from dataclasses import dataclass
from typing import Self

import numpy as np
from zarr.abc.codec import ArrayArrayCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import NDBuffer
from zarr.core.common import JSON, parse_named_configuration
from zarr.core.metadata.v3 import RegularChunkGridMetadata
from zarr.registry import register_codec

CODEC_NAME = "dynamicalorg.latitude_longitude"


@dataclass(frozen=True)
class LatitudeLongitudeCodec(ArrayArrayCodec):
    """Orient a whole global spatial chunk north-up and from -180 to 180 degrees."""

    is_fixed_size = True

    latitude_axis: int
    longitude_axis: int

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration = parse_named_configuration(data, CODEC_NAME)
        return cls(**configuration)  # ty: ignore[invalid-argument-type]

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": CODEC_NAME,
            "configuration": {
                "latitude_axis": self.latitude_axis,
                "longitude_axis": self.longitude_axis,
            },
        }

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: object,
        chunk_grid: object,
    ) -> None:
        del dtype
        latitude_axis = _normalize_axis(self.latitude_axis, len(shape))
        longitude_axis = _normalize_axis(self.longitude_axis, len(shape))
        if latitude_axis == longitude_axis:
            raise ValueError("latitude_axis and longitude_axis must be different")
        if shape[longitude_axis] % 2:
            raise ValueError("longitude must have an even number of points")
        if not isinstance(chunk_grid, RegularChunkGridMetadata):
            raise ValueError("latitude and longitude must use regular chunks")
        for axis in (latitude_axis, longitude_axis):
            if chunk_grid.chunk_shape[axis] != shape[axis]:
                raise ValueError(
                    "each chunk must contain the complete latitude and longitude axes"
                )

    def _decode_sync(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        return self._transform(chunk_array, chunk_spec, decode=True)

    async def _decode_single(
        self,
        chunk_data: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        return self._decode_sync(chunk_data, chunk_spec)

    def _encode_sync(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        return self._transform(chunk_array, chunk_spec, decode=False)

    async def _encode_single(
        self,
        chunk_data: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        return self._encode_sync(chunk_data, chunk_spec)

    def _transform(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
        *,
        decode: bool,
    ) -> NDBuffer:
        data = chunk_array.as_numpy_array()
        latitude_axis = _normalize_axis(self.latitude_axis, data.ndim)
        longitude_axis = _normalize_axis(self.longitude_axis, data.ndim)
        data = np.flip(data, axis=latitude_axis)
        longitude_shift = data.shape[longitude_axis] // 2
        if not decode:
            longitude_shift = -longitude_shift
        data = np.roll(data, shift=longitude_shift, axis=longitude_axis)
        return chunk_spec.prototype.nd_buffer.from_numpy_array(data)

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        del chunk_spec
        return input_byte_length


def _normalize_axis(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise ValueError(
            f"axis {axis} is out of bounds for an array with {ndim} dimensions"
        )
    return axis % ndim


register_codec(CODEC_NAME, LatitudeLongitudeCodec)
