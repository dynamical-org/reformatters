import numpy as np
import pytest
import zarr
from zarr.codecs import BloscCodec, BytesCodec, TransposeCodec
from zarr.registry import get_codec_class
from zarr.storage import MemoryStore

from reformatters.common.latitude_longitude_codec import (
    CODEC_NAME,
    LatitudeLongitudeCodec,
)


def test_metadata_round_trip_and_registration() -> None:
    codec = LatitudeLongitudeCodec(latitude_axis=-3, longitude_axis=-2)

    assert codec.from_dict(codec.to_dict()) == codec
    assert get_codec_class(CODEC_NAME) is LatitudeLongitudeCodec


def test_decodes_native_spatial_order_after_blosc_and_transpose() -> None:
    source = np.arange(24, dtype=np.float32).reshape(1, 3, 2, 4)
    canonical = np.roll(np.flip(source, axis=2), shift=2, axis=3).transpose(0, 2, 3, 1)
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

    source_store = MemoryStore()
    source_array = zarr.create_array(
        source_store,
        shape=source.shape,
        chunks=source.shape,
        dtype=source.dtype,
        serializer=BytesCodec(),
        compressors=[compressor],
    )
    source_array[:] = source

    target_store = MemoryStore()
    target_array = zarr.create_array(
        target_store,
        shape=canonical.shape,
        chunks=canonical.shape,
        dtype=canonical.dtype,
        filters=[
            LatitudeLongitudeCodec(latitude_axis=1, longitude_axis=2),
            TransposeCodec(order=(0, 3, 1, 2)),
        ],
        serializer=BytesCodec(),
        compressors=[compressor],
    )
    target_array[:] = canonical

    source_chunk = source_store.get_sync("c/0/0/0/0")
    target_chunk = target_store.get_sync("c/0/0/0/0")
    assert source_chunk is not None
    assert target_chunk is not None
    assert source_chunk.to_bytes() == target_chunk.to_bytes()
    np.testing.assert_array_equal(target_array[:], canonical)


@pytest.mark.parametrize(
    ("shape", "chunks", "latitude_axis", "longitude_axis", "message"),
    [
        ((2, 4), (1, 4), 0, 1, "complete latitude"),
        ((2, 4), (2, 2), 0, 1, "complete latitude"),
        ((2, 3), (2, 3), 0, 1, "even number"),
        ((2, 4), (2, 4), 0, 0, "must be different"),
        ((2, 4), (2, 4), 0, 2, "out of bounds"),
    ],
)
def test_rejects_incompatible_array_layouts(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    latitude_axis: int,
    longitude_axis: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        zarr.create_array(
            MemoryStore(),
            shape=shape,
            chunks=chunks,
            dtype="float32",
            filters=[
                LatitudeLongitudeCodec(
                    latitude_axis=latitude_axis,
                    longitude_axis=longitude_axis,
                )
            ],
        )
