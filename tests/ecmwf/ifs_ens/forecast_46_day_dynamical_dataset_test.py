import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common import validation
from reformatters.common.types import DatetimeLike
from reformatters.ecmwf.archive_gribs.forecast_46_day_archiver import ECDS_VARIABLES
from reformatters.ecmwf.ifs_ens.forecast_46_day_1_5_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast46Day15DegreeDataset,
)
from tests.chunk_utils import shrink_chunks_and_shards
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG


@pytest.fixture
def dataset() -> EcmwfIfsEnsForecast46Day15DegreeDataset:
    return EcmwfIfsEnsForecast46Day15DegreeDataset(
        primary_storage_config=NOOP_STORAGE_CONFIG
    )


def test_validators_check_masked_variables_are_not_all_nan(
    dataset: EcmwfIfsEnsForecast46Day15DegreeDataset,
) -> None:
    validators = tuple(dataset.validators())

    assert len(validators) == 3
    assert isinstance(validators[1], validation.CheckRecentNans)
    assert isinstance(validators[2], validation.CheckRecentNans)
    assert validators[2].include_vars == validators[1].exclude_vars
    assert validators[2].max_nan_fraction == 0.9999
    assert validators[2].spatial_sampling == "quarter"


def test_archive_contains_every_dataset_source_variable(
    dataset: EcmwfIfsEnsForecast46Day15DegreeDataset,
) -> None:
    assert {
        data_var.internal_attrs.ecds_variable
        for data_var in dataset.template_config.data_vars
    } == set(ECDS_VARIABLES)


LEVELS = (1000, 925, 850, 700, 500, 300, 200, 100, 50, 10)


@pytest.mark.slow
def test_backfill_local_writes_pressure_levels(
    monkeypatch: pytest.MonkeyPatch, dataset: EcmwfIfsEnsForecast46Day15DegreeDataset
) -> None:
    """A pressure level variable lands on its declared axis, with real values."""
    root_var, pressure_var = "pressure_surface", "temperature"
    monkeypatch.setattr(
        type(dataset.template_config),
        "data_vars",
        [
            var
            for var in dataset.template_config.data_vars
            if var.name in (root_var, pressure_var)
        ],
    )

    orig_get_template = dataset.template_config.get_template

    def small_template(self: object, end_time: DatetimeLike) -> xr.DataTree:
        tree = orig_get_template(end_time).sel(
            lead_time=slice("0h", "24h"), ensemble_member=slice(0, 0)
        )
        # Leave init_time at production geometry: one shard per init, so
        # filter_start scopes the run to a single initialization.
        return shrink_chunks_and_shards(
            xr.DataTree.from_dict(
                {
                    "/": tree.to_dataset()[[root_var]],
                    "pressure_level": tree["pressure_level"].to_dataset()[
                        [pressure_var]
                    ],
                }
            ),
            dims=(
                "lead_time",
                "ensemble_member",
                "pressure_level",
                "latitude",
                "longitude",
            ),
        )

    monkeypatch.setattr(type(dataset.template_config), "get_template", small_template)

    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2026-08-11T00:00"),
        filter_start=pd.Timestamp("2026-08-10T00:00"),
    )

    store = dataset.store_factory.primary_store()
    root = xr.open_zarr(store, chunks=None)
    levels = xr.open_zarr(store, group="pressure_level", chunks=None)

    init = np.datetime64("2026-08-10T00:00:00")
    assert init in root.init_time.values
    assert np.isfinite(root[root_var].sel(init_time=init).values).any()

    temperature = (
        levels[pressure_var].sel(init_time=init).isel(lead_time=0, ensemble_member=0)
    )
    assert temperature.dims == ("pressure_level", "latitude", "longitude")
    assert np.isfinite(temperature.values).all()

    # Global mean per level, coldest at the 100 hPa tropopause and warming above it.
    np.testing.assert_allclose(
        [float(temperature.sel(pressure_level=level).mean()) for level in LEVELS],
        [
            10.936,
            7.364,
            4.457,
            -2.785,
            -17.292,
            -41.806,
            -55.184,
            -65.717,
            -62.399,
            -49.726,
        ],
        atol=1e-3,
        rtol=0,
    )
