import numpy as np
import xarray as xr

from scripts.validation.compare_spatial import align_reference_spatially
from scripts.validation.utils import load_zarr_dataset


def _reference_ds() -> xr.Dataset:
    latitude = np.arange(90, -91, -45)
    longitude = np.arange(-180, 180, 45)
    return xr.Dataset(
        {
            "temperature_2m": (
                ("latitude", "longitude"),
                np.zeros((len(latitude), len(longitude))),
            )
        },
        coords={"latitude": latitude, "longitude": longitude},
    )


def test_align_reference_spatially_keeps_full_longitude_for_geographic_xy(
    geographic_xy_store: str,
) -> None:
    ds = load_zarr_dataset(geographic_xy_store)
    reference_ds = _reference_ds()

    aligned = align_reference_spatially(ds, reference_ds)

    # The dataset's own 0 to 360 longitude bounds would have cropped this to one
    # hemisphere; latitude bounds still apply.
    np.testing.assert_array_equal(
        aligned.longitude.values, reference_ds.longitude.values
    )
    np.testing.assert_array_equal(aligned.latitude.values, reference_ds.latitude.values)
