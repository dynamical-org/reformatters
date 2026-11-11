from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.template_utils import ignore_consolidated_metadata_spec_warning
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_1_degree_virtual.region_job import (
    EcmwfIfsEnsForecast15Day01DegreeVirtualRegionJob,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_1_degree_virtual.template_config import (
    EcmwfIfsEnsForecast15Day01DegreeVirtualTemplateConfig,
)
from reformatters.noaa.refs.forecast_hourly_virtual.template_config import (
    NoaaRefsForecastHourlyVirtualTemplateConfig,
)
from scripts.validation.utils import RunContext


@pytest.fixture
def statistic_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunContext:
    config = NoaaRefsForecastHourlyVirtualTemplateConfig()
    names = (
        "wind_u_10m",
        "temperature_2m",
        "derived_radar_reflectivity_1000m",
        "temperature_2m_standard_deviation",
    )
    variables = [v for v in config.data_vars if v.name in names]
    variables.extend(
        next(
            v for v in config.data_vars if v.internal_attrs.source_families == (family,)
        )
        for family in ("pmmn", "lpmm", "avrg")
    )
    wind = next(v for v in variables if v.name == "wind_u_10m")
    variables.append(
        wind.model_copy(update={"name": "wind_u", "group": "pressure_level"})
    )
    monkeypatch.setattr(
        "scripts.validation.scan_common.find_registered_dataset",
        Mock(return_value=Mock(template_config=Mock(data_vars=variables))),
    )
    monkeypatch.setattr(
        "scripts.validation.value_timeseries.restrict_plot_coverage",
        lambda ctx, var, ds, position: ds,
    )
    init = pd.date_range("2026-09-15", periods=3, freq="D")
    lead = pd.to_timedelta([0, 1, 2], unit="h")
    coords = {
        "init_time": init,
        "lead_time": lead,
        "latitude": [40.0, 30.0],
        "longitude": [-100.0, -90.0],
        "pressure_level": [1000, 500, 100],
        "statistic": np.asarray(["mean", "standard_deviation"], dtype=object),
        "valid_time": (
            ("init_time", "lead_time"),
            init.values[:, None] + lead.values[None, :],
        ),
    }
    ds = xr.Dataset(
        coords=coords, attrs={"dataset_id": config.dataset_id, "name": "Validation"}
    )
    for var in variables:
        dims = ("init_time", "lead_time", "latitude", "longitude")
        if var.group == "pressure_level":
            dims = (*dims, "pressure_level")
        da = xr.DataArray(np.full(tuple(ds.sizes[d] for d in dims), 2.0), dims=dims)
        if var.has_statistic:
            da = da.expand_dims(statistic=ds.statistic).copy(deep=True)
            da.loc[{"statistic": "standard_deviation"}] = 7.0
            if var.internal_attrs.source_families == ("mean",):
                da.loc[{"statistic": "standard_deviation"}] = np.nan
            if var.internal_attrs.source_families == ("sprd",):
                da.loc[{"statistic": "mean"}] = np.nan
        ds[var.path] = da
    reference = (
        xr.Dataset(
            {
                var: ds[var].isel(statistic=0, drop=True)
                if "statistic" in ds[var].dims
                else ds[var]
                for var in ds.data_vars
            }
        )
        .isel(init_time=0, drop=True)
        .rename({"lead_time": "time"})
    )
    reference = reference.assign_coords(time=init[0] + lead).drop_vars("valid_time")
    reference.attrs["name"] = "Reference"
    return RunContext(
        output_dir=tmp_path,
        validation_url="s3://bucket/refs/v1.icechunk",
        reference_url="s3://bucket/ref/v1.icechunk",
        validation_ds=ds,
        reference_ds=reference,
        started_at=pd.Timestamp.now(tz="UTC"),
        point1_sel={"latitude": 0, "longitude": 0},
        point2_sel={"latitude": 1, "longitude": 1},
        point1_lat=40.0,
        point1_lon=-100.0,
        point2_lat=30.0,
        point2_lon=-90.0,
        ensemble_member=None,
        variables=list(map(str, ds.data_vars)),
        init_time="2026-09-15",
        lead_time="1",
        level_override=720,
    )


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


@pytest.fixture
def sparse_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunContext:
    config = EcmwfIfsEnsForecast15Day01DegreeVirtualTemplateConfig()
    init = pd.date_range("2026-11-03", periods=2, freq="6h")
    lead = pd.to_timedelta([0, 3, 150], unit="h")
    ds = xr.Dataset(
        coords={
            "init_time": init,
            "lead_time": lead,
            "ensemble_member": [0, 25],
            "latitude": [40.0, 30.0],
            "longitude": [-100.0, -90.0],
            "pressure_level": [500],
            "valid_time": (
                ("init_time", "lead_time"),
                init.values[:, None] + lead.values[None, :],
            ),
        },
        attrs={"dataset_id": config.dataset_id, "name": "Sparse forecast"},
    )
    dims = ("init_time", "lead_time", "ensemble_member", "latitude", "longitude")
    shape = tuple(ds.sizes[d] for d in dims)
    ds["temperature_2m"] = xr.DataArray(np.full(shape, 2.0), dims=dims)
    ds["geopotential_height_surface"] = xr.DataArray(np.full(shape, np.nan), dims=dims)
    ds["geopotential_height_surface"].loc[
        {"lead_time": lead[0], "ensemble_member": 0}
    ] = 100.0
    ds["pressure_level/geopotential"] = xr.DataArray(
        np.full((*shape, 1), np.nan), dims=(*dims, "pressure_level")
    )
    ds["pressure_level/geopotential"].loc[{"ensemble_member": 0}] = 5000.0
    monkeypatch.setattr(
        "scripts.validation.scan_common.find_registered_dataset",
        lambda _: SimpleNamespace(
            region_job_class=EcmwfIfsEnsForecast15Day01DegreeVirtualRegionJob,
            template_config=config,
        ),
    )
    times = pd.date_range("2026-11-03", "2026-11-10", freq="3h")
    ref = xr.Dataset(
        {
            "temperature_2m": (
                ("time", "latitude", "longitude"),
                np.ones((len(times), 2, 2)),
            ),
            "geopotential_height_surface": (
                ("time", "latitude", "longitude"),
                np.full((len(times), 2, 2), 90.0),
            ),
            "pressure_level/geopotential": (
                ("time", "latitude", "longitude", "pressure_level"),
                np.full((len(times), 2, 2, 1), 4900.0),
            ),
        },
        coords={
            "time": times,
            "latitude": ds.latitude,
            "longitude": ds.longitude,
            "pressure_level": ds.pressure_level,
        },
    )
    return RunContext(
        output_dir=tmp_path,
        validation_url="s3://bucket/sparse/v1.icechunk",
        reference_url=None,
        validation_ds=ds,
        reference_ds=ref,
        started_at=pd.Timestamp.now(tz="UTC"),
        point1_sel={"latitude": 0, "longitude": 0},
        point2_sel={"latitude": 1, "longitude": 1},
        point1_lat=40.0,
        point1_lon=-100.0,
        point2_lat=30.0,
        point2_lon=-90.0,
        ensemble_member=25,
        variables=list(map(str, ds.data_vars)),
        is_virtual=True,
        init_time=str(init[0]),
        lead_time="150",
    )
