from collections.abc import Sequence
from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import pandas as pd
import pytest

from reformatters.common import template_utils, validation
from reformatters.common.iterating import flatten_groups
from reformatters.common.storage import (
    DatasetFormat,
    StorageConfig,
    _repository_config_and_credentials,
)
from reformatters.common.virtual_region_job import VirtualRef
from reformatters.ecmwf.ifs_ens.forecast_6_day_0_1_degree_virtual.dynamical_dataset import (
    EcmwfIfsEnsForecast6Day01DegreeVirtualDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_6_day_0_25_degree_virtual.dynamical_dataset import (
    EcmwfIfsEnsForecast6Day025DegreeVirtualDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree_virtual.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day025DegreeVirtualDataset,
)
from reformatters.ecmwf.ifs_ens.virtual_region_job import (
    EcmwfIfsEnsForecastVirtualRegionJob,
)
from reformatters.ecmwf.ifs_ens.virtual_template_config import EcmwfIfsEnsVirtualDataVar
from tests.common.dynamical_dataset_test import (
    assert_configured_validators,
)

VARIABLES = (
    "temperature_2m",
    "soil_temperature_layer_3",
    "total_precipitation_run_total_surface",
    "temperature",
    "geopotential_height_surface",
)


@pytest.mark.slow
@pytest.mark.parametrize(
    "dataset_class",
    [
        EcmwfIfsEnsForecast6Day025DegreeVirtualDataset,
        EcmwfIfsEnsForecast6Day01DegreeVirtualDataset,
    ],
)
def test_backfill_and_update_real_source_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_class: type[
        EcmwfIfsEnsForecast6Day025DegreeVirtualDataset
        | EcmwfIfsEnsForecast6Day01DegreeVirtualDataset
    ],
) -> None:
    dataset = dataset_class(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    original = dataset.template_config.get_template
    monkeypatch.setattr(
        type(dataset.template_config),
        "get_template",
        lambda self, end_time: original(
            min(pd.Timestamp(end_time), pd.Timestamp("2026-11-06T12"))
        ).sel(
            init_time=slice("2026-11-05T18", None),
            lead_time=pd.to_timedelta([0, 3], unit="h"),
            ensemble_member=[0, 1],
            pressure_level=[500, 850],
        ),
    )
    dataset.backfill_local(
        append_dim_end=pd.Timestamp("2026-11-06T01"),
        filter_start=pd.Timestamp("2026-11-06"),
        filter_variable_names=list(VARIABLES),
    )
    ds = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    cell = ds.sel(init_time=pd.Timestamp("2026-11-06"), latitude=-5, longitude=-60)
    fine = dataset.template_config.resolution == "0p1"
    np.testing.assert_allclose(
        cell.temperature_2m.values,
        [
            [25.355676269531273, 25.405297851562523],
            [24.80896911621096, 24.219277954101585],
        ]
        if fine
        else [
            [25.440164184570335, 25.173913574218773],
            [24.625711059570335, 24.242532348632835],
        ],
    )
    np.testing.assert_allclose(
        cell.soil_temperature_layer_3.values,
        [
            [27.568338012695335, 27.822930908203148],
            [27.567239379882835, 27.822549438476585],
        ]
        if fine
        else [
            [27.64447937011721, 27.647760009765648],
            [27.64338073730471, 27.647378540039085],
        ],
    )
    np.testing.assert_array_equal(
        cell.total_precipitation_run_total_surface.values, [[0, 0], [0, 0]]
    )
    np.testing.assert_allclose(
        cell["pressure_level/temperature"].values,
        [
            [
                [-4.204077148437477, 21.36341247558596],
                [-4.40660705566404, 22.15070495605471],
            ],
            [
                [-4.123251342773415, 21.211358642578148],
                [-4.429937744140602, 21.30542907714846],
            ],
        ]
        if fine
        else [
            [
                [-4.269735717773415, 21.31934509277346],
                [-4.381735229492165, 22.06324157714846],
            ],
            [
                [-4.170492553710915, 21.119287109375023],
                [-4.32250061035154, 21.482049560546898],
            ],
        ],
    )
    np.testing.assert_allclose(
        cell.geopotential_height_surface.values,
        [[43.49505401787308 if fine else 43.52983291312528, np.nan], [np.nan, np.nan]],
        equal_nan=True,
    )
    reference = pd.Timestamp("2026-11-06T14:12" if fine else "2026-11-06T12:12")

    def now(cls: type[pd.Timestamp], tz: str | None = None) -> pd.Timestamp:
        return reference if tz is None else reference.tz_localize(tz)

    monkeypatch.setattr(pd.Timestamp, "now", classmethod(now))
    original_jobs = EcmwfIfsEnsForecastVirtualRegionJob.operational_update_jobs.__func__

    def update_jobs(
        cls: type[EcmwfIfsEnsForecastVirtualRegionJob],
        *,
        all_data_vars: Sequence[EcmwfIfsEnsVirtualDataVar],
        **kwargs: Any,  # noqa: ANN401 - forwarded to the wrapped factory
    ) -> object:
        return original_jobs(
            cls,
            all_data_vars=[v for v in all_data_vars if v.name in VARIABLES],
            **kwargs,
        )

    monkeypatch.setattr(
        EcmwfIfsEnsForecastVirtualRegionJob,
        "operational_update_jobs",
        classmethod(update_jobs),
    )
    monkeypatch.setattr(
        dataset.region_job_class, "operational_update_window", pd.Timedelta("12h")
    )
    dataset.update("test-update")
    updated = validation.open_flattened_dataset(
        dataset.store_factory.primary_store(), consolidated=False
    )
    assert updated.init_time.values[-1] == np.datetime64("2026-11-06T06")
    point = updated.sel(
        init_time=pd.Timestamp("2026-11-06T06"),
        latitude=-5,
        longitude=-60,
        lead_time=pd.Timedelta("3h"),
        ensemble_member=1,
    )
    assert -10 < float(point.temperature_2m.values) < 40
    assert_configured_validators(dataset)


def test_configured_15_day_catch_up_checks_older_inits(tmp_path: Path) -> None:
    dataset = EcmwfIfsEnsForecast15Day025DegreeVirtualDataset(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path), format=DatasetFormat.ICECHUNK
        )
    )
    inits = pd.to_datetime(["2026-11-06T00", "2026-11-06T12"])
    template = dataset.template_config.get_template("2026-11-07").sel(
        init_time=inits, lead_time=[pd.Timedelta(0)], ensemble_member=[0, 1, 50]
    )
    var = next(
        var
        for var in dataset.template_config.data_vars
        if var.name == "pressure_surface"
    )
    job = dataset.region_job_class(
        tmp_store=tmp_path / "unused.zarr",
        template_ds=template,
        data_vars=[var],
        append_dim="init_time",
        region=slice(0, 2),
        reformat_job_name="catch-up",
        processing_mode="update",
    )
    config, credentials = _repository_config_and_credentials(
        dataset.icechunk_virtual_config
    )
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "icechunk")),
        config=config,
        authorize_virtual_chunk_access=credentials,
    )
    session = repo.writable_session("main")
    template_utils.write_metadata(
        template,
        session.store,
        mode="w-",
        skip_icechunk_commit=True,
        consolidated=False,
    )
    coords = job.source_file_coords()
    refs = [
        VirtualRef(
            data_var=var, out_loc=loc, location=coord.get_url(), offset=0, length=1
        )
        for coord in coords
        for loc in job.message_locations(coord, var)
        if coord.init_time == inits[-1] or coord.family != "enfo"
    ]
    assert sum(ref.out_loc["init_time"] == inits[-1] for ref in refs) == 3
    job._emit_refs([session.store], refs)
    older_perturbed = next(
        coord
        for coord in coords
        if coord.init_time == inits[0] and coord.family == "enfo"
    )
    assert job.filter_already_present([older_perturbed], session.store) == [
        older_perturbed
    ]
    validator = dataset.validators()[1]
    assert isinstance(validator, validation.CheckVirtualManifestCompleteness)
    result = validator.check(
        validation.ValidationContext(
            store=session.store,
            ds=flatten_groups(template),
            append_dim="init_time",
            region_job=job,
        )
    )
    assert not result.passed
    assert str(inits[0]) in result.message
    assert str(inits[-1]) not in result.message
