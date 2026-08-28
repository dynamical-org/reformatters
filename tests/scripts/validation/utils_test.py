from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import xarray as xr

from scripts.validation.utils import (
    _anonymous_virtual_credentials,
    _icechunk_storage,
    choose_level,
    get_random_spatial_indices,
    get_two_random_points,
    load_zarr_dataset,
    nearest_point_index,
    parse_point_options,
    to_reference_longitude,
    var_slug,
    vertical_dims,
)


def _projected_ds() -> xr.Dataset:
    y = np.arange(20)
    x = np.arange(30)
    # 2D lat/lon like a projected (Lambert) grid: lat increases with y, lon with x.
    lat2d = np.broadcast_to((30.0 + y * 0.5)[:, None], (20, 30)).copy()
    lon2d = np.broadcast_to((-110.0 + x * 0.5)[None, :], (20, 30)).copy()
    return xr.Dataset(
        {"t": (("y", "x"), np.zeros((20, 30)))},
        coords={
            "y": y,
            "x": x,
            "latitude": (("y", "x"), lat2d),
            "longitude": (("y", "x"), lon2d),
        },
    )


def _grouped_ds() -> xr.Dataset:
    init = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    lead = pd.to_timedelta([0, 1, 2, 3], unit="h")
    y = np.arange(5)
    x = np.arange(6)
    pressure_level = np.array([1000, 850, 700, 500, 300, 100])
    return xr.Dataset(
        {
            "temperature_2m": (
                ("init_time", "lead_time", "y", "x"),
                np.zeros((3, 4, 5, 6)),
            ),
            "pressure_level/temperature": (
                ("init_time", "lead_time", "y", "x", "pressure_level"),
                np.zeros((3, 4, 5, 6, 6)),
            ),
        },
        coords={
            "init_time": init,
            "lead_time": lead,
            "y": y,
            "x": x,
            "pressure_level": pressure_level,
        },
    )


def test_var_slug() -> None:
    assert var_slug("temperature_2m") == "temperature_2m"
    assert var_slug("pressure_level/temperature") == "pressure_level__temperature"


def test_vertical_dims() -> None:
    ds = _grouped_ds()
    assert vertical_dims(ds, "temperature_2m") == []
    assert vertical_dims(ds, "pressure_level/temperature") == ["pressure_level"]


def test_choose_level_single_level_var_returns_empty() -> None:
    ds = _grouped_ds()
    assert choose_level(ds, "temperature_2m", None) == {}


def test_choose_level_default_is_middle() -> None:
    ds = _grouped_ds()
    # 6 levels -> middle index 3 -> 500
    assert choose_level(ds, "pressure_level/temperature", None) == {
        "pressure_level": 500
    }


def test_choose_level_override_selects_nearest() -> None:
    ds = _grouped_ds()
    assert choose_level(ds, "pressure_level/temperature", 720) == {
        "pressure_level": 700
    }
    assert choose_level(ds, "pressure_level/temperature", 50) == {"pressure_level": 100}


def test_parse_point_options() -> None:
    assert parse_point_options(None) == []
    assert parse_point_options([]) == []
    assert parse_point_options(["39.0,-98.5"]) == [(39.0, -98.5)]
    assert parse_point_options(["39,-98.5", "33.75,-84.4"]) == [
        (39.0, -98.5),
        (33.75, -84.4),
    ]


def test_parse_point_options_rejects_more_than_two() -> None:
    try:
        parse_point_options(["1,2", "3,4", "5,6"])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for more than two points")


def test_random_spatial_indices_within_middle_50_percent() -> None:
    ds = _projected_ds()
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    for _ in range(500):
        p1, p2 = get_random_spatial_indices(ds, "y", "x")
        for p in (p1, p2):
            assert ny // 4 <= p["y"] < ny - ny // 4
            assert nx // 4 <= p["x"] < nx - nx // 4


def test_nearest_point_index_projected_grid() -> None:
    ds = _projected_ds()
    # lat = 30 + 0.5*y, lon = -110 + 0.5*x -> (35, -100) is y=10, x=20.
    assert nearest_point_index(ds, 35.0, -100.0) == {"y": 10, "x": 20}


def test_get_two_random_points_pins_provided_points() -> None:
    ds = _projected_ds()
    # lat = 30 + 0.5*y, lon = -110 + 0.5*x.
    p1_sel, p2_sel, (lat1, lon1), (lat2, lon2) = get_two_random_points(
        ds, [(35.0, -100.0), (39.0, -96.0)]
    )
    assert p1_sel == {"y": 10, "x": 20}
    assert p2_sel == {"y": 18, "x": 28}
    assert (lat1, lon1) == (35.0, -100.0)
    assert (lat2, lon2) == (39.0, -96.0)


def test_load_zarr_dataset_keeps_geographic_xy_labels_native(
    geographic_xy_store: str,
) -> None:
    ds = load_zarr_dataset(geographic_xy_store)

    assert ds["latitude"].dims == ("y", "x")
    assert ds["longitude"].dims == ("y", "x")
    np.testing.assert_array_equal(
        ds["longitude"].values[0], [0, 45, 90, 135, 180, 225, 270, 315]
    )
    assert nearest_point_index(ds, 45.0, 225.0) == {"y": 3, "x": 5}
    _, _, (lat1, lon1), _ = get_two_random_points(ds, [(45.0, 225.0)])
    assert (lat1, lon1) == (45.0, 225.0)


def test_to_reference_longitude() -> None:
    assert to_reference_longitude(225.0) == -135.0
    assert to_reference_longitude(0.0) == 0.0
    assert to_reference_longitude(179.75) == 179.75
    assert to_reference_longitude(180.0) == -180.0
    # Already in the reference convention, so unchanged.
    assert to_reference_longitude(-110.0) == -110.0


def test_icechunk_storage_routes_by_url_scheme() -> None:
    https_url = "https://pub-abc.r2.dev/some-dataset/v0.1.0.icechunk"
    s3_url = "s3://some-bucket/some-dataset/v0.1.0.icechunk"
    assert _icechunk_storage(https_url) is not None
    assert _icechunk_storage(s3_url) is not None
    assert _icechunk_storage("s3://some-bucket/some-dataset/v0.1.0.zarr") is None
    assert _icechunk_storage("https://example.com/report.html") is None


def test_anonymous_virtual_credentials_authorize_http_container(tmp_path: Path) -> None:
    """An HTTP virtual chunk container needs HttpAccess; icechunk rejects an S3
    credential handed to one, so a wrong credential kind fails the store open."""
    container = icechunk.VirtualChunkContainer(
        "https://example.com/chunks/", icechunk.http_store()
    )
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(container)
    storage = icechunk.local_filesystem_storage(str(tmp_path / "repo.icechunk"))
    repo = icechunk.Repository.create(
        storage,
        config=config,
        authorize_virtual_chunk_access={
            container.url_prefix: icechunk.Credentials.HttpAccess()
        },
    )
    repo.save_config()

    credentials = _anonymous_virtual_credentials(storage)
    assert credentials is not None
    icechunk.Repository.open(storage, authorize_virtual_chunk_access=credentials)
