import numpy as np

from scripts.validation.compare_spatial import align_reference_spatially
from tests.scripts.validation.grids import (
    ASCENDING_LAT,
    DESCENDING_LAT,
    LON_0_360,
    LON_180,
    global_lonlat_ds,
)


def test_align_reference_spatially_keeps_the_whole_globe_for_a_0_360_dataset() -> None:
    # Slicing a -180..180 reference by 0..355 would crop it to its eastern half.
    ds = global_lonlat_ds(LON_0_360, ASCENDING_LAT)
    reference = global_lonlat_ds(LON_180, DESCENDING_LAT)
    aligned = align_reference_spatially(ds, reference)
    assert aligned.sizes["longitude"] == reference.sizes["longitude"]
    assert aligned.sizes["latitude"] == reference.sizes["latitude"]
    np.testing.assert_array_equal(aligned.longitude.values, LON_0_360)


def test_align_reference_spatially_keeps_the_whole_globe_for_a_180_dataset() -> None:
    ds = global_lonlat_ds(LON_180, DESCENDING_LAT)
    reference = global_lonlat_ds(LON_180, DESCENDING_LAT)
    aligned = align_reference_spatially(ds, reference)
    assert aligned.sizes["longitude"] == reference.sizes["longitude"]
    assert aligned.sizes["latitude"] == reference.sizes["latitude"]


def test_align_reference_spatially_crops_to_a_regional_dataset() -> None:
    ds = global_lonlat_ds(np.arange(-100.0, -80.0, 5.0), np.arange(45.0, 30.0, -5.0))
    aligned = align_reference_spatially(ds, global_lonlat_ds(LON_180, DESCENDING_LAT))
    np.testing.assert_array_equal(
        aligned.longitude.values, np.arange(-100.0, -80.0, 5.0)
    )
    np.testing.assert_array_equal(aligned.latitude.values, np.arange(45.0, 30.0, -5.0))
