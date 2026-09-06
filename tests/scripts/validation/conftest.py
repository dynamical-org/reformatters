from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from reformatters.common.template_utils import ignore_consolidated_metadata_spec_warning


@pytest.fixture
def geographic_xy_store(tmp_path: Path) -> str:
    """A store shaped like the WeatherNext 2 products: geographic y/x dimension
    coordinates, latitude ascending and longitude spanning 0 to 360."""
    y = np.arange(-90, 91, 45)
    x = np.arange(0, 360, 45)
    store_path = tmp_path / "wn2.zarr"
    ds = xr.Dataset(
        {
            "temperature_2m": (
                ("y", "x"),
                np.arange(len(y) * len(x), dtype="float32").reshape(len(y), len(x)),
            )
        },
        coords={"y": y, "x": x},
        attrs={"dataset_id": "google-weathernext2-forecast-operational-virtual"},
    )
    # Match production stores, which carry consolidated metadata; load_zarr_dataset
    # opens a local store with consolidated=True.
    with ignore_consolidated_metadata_spec_warning():
        ds.to_zarr(store_path, zarr_format=3, consolidated=True)
    return str(store_path)
