import json
from pathlib import Path
from typing import Any

import fsspec
import icechunk
import pandas as pd
import pytest
import s3fs
import xarray as xr

from reformatters.common import kubernetes, storage, template_utils
from reformatters.common.config import Config, Env
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import ReformatCronJob
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.dynamical_dataset import (
    EcmwfIfsEnsForecast15Day025DegreeDataset,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
)
from reformatters.ecmwf.ifs_ens.profiling.config import (
    ROOT,
    ProfilingDataset,
    ProfilingStorage,
    ProfilingStoreFactory,
    dataset,
    run_root,
)
from reformatters.ecmwf.ifs_ens.profiling.template_config import (
    DATASET_ID,
    END,
    START,
    VERSION,
    ProfilingTemplateConfig,
)
from reformatters.ecmwf.ifs_ens.profiling.worker import ProfilingJob
from tests.common import (
    common_template_config_subclasses_test,
    datasets_cf_compliance_test,
    template_serialization_test,
)
from tests.dataset_helpers import IMPLEMENTED_DATASETS

RUN = "0123456789abcdef0123456789abcdef"
EXPECTED_BASE = f"{ROOT}/{RUN}/data"


@pytest.fixture(autouse=True)
def prod_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_secret(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Profiling tests must not load credentials")

    monkeypatch.setattr(kubernetes, "load_secret", forbidden_secret)
    monkeypatch.setattr(kubernetes.config, "load_kube_config", forbidden_secret)
    monkeypatch.setattr(kubernetes.config, "load_incluster_config", forbidden_secret)
    monkeypatch.setattr(s3fs.S3FileSystem, "__init__", forbidden_secret)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
    for name in (
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        storage.StoreFactory,
        "_coordination_fs",
        lambda _self: fsspec.filesystem("memory"),
    )
    monkeypatch.setattr(Config, "env", Env.prod)


@pytest.fixture
def factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ProfilingStoreFactory:
    def local_storage(*_args: object) -> icechunk.Storage:
        return icechunk.local_filesystem_storage(str(tmp_path / "repository"))

    monkeypatch.setattr(storage, "_get_icechunk_storage", local_storage)
    return dataset(RUN).store_factory


@pytest.mark.parametrize(
    "run_id", ["", "../../production", RUN.upper(), RUN[:-1], f"{RUN}0", f"p{RUN[1:]}"]
)
def test_run_root_requires_lowercase_hex_uuid(run_id: str) -> None:
    with pytest.raises(AssertionError):
        run_root(run_id)


def test_destination_paths() -> None:
    factory = dataset(RUN).store_factory
    assert factory.primary_url() == f"{EXPECTED_BASE}/{DATASET_ID}/v{VERSION}.icechunk"
    assert (
        factory._coordination_base_path() == f"{EXPECTED_BASE}/{DATASET_ID}/_internal"
    )
    assert factory.evidence_url == f"{ROOT}/{RUN}/profiles"
    assert factory.ownership_tag == f"profiling-{RUN}"
    assert factory.k8s_secret_names() == ["aws-open-data-icechunk-storage-options-key"]


def test_destination_requires_prod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Config, "env", Env.dev)
    with pytest.raises(AssertionError, match="DYNAMICAL_ENV=prod"):
        dataset(RUN)


def test_create_only_and_owned_sessions(factory: ProfilingStoreFactory) -> None:
    factory.bootstrap(0)
    factory.bootstrap(1)
    assert factory.primary_store().session.read_only
    assert not factory.primary_store(writable=True).session.read_only
    with pytest.raises(icechunk.IcechunkError):
        factory.bootstrap(0)


def test_missing_repository_reads_as_not_found(factory: ProfilingStoreFactory) -> None:
    with pytest.raises(icechunk.NotFoundError):
        factory.primary_store()
    assert dataset(RUN)._open_existing_store() is None


def test_existing_unowned_repository_is_refused(
    factory: ProfilingStoreFactory,
) -> None:
    icechunk.Repository.create(factory._storage())
    with pytest.raises(AssertionError, match="not owned"):
        factory.primary_store(writable=True)
    with pytest.raises(AssertionError, match="not owned"):
        dataset(RUN)._open_existing_store()
    with pytest.raises(icechunk.IcechunkError):
        factory.bootstrap(0)


def test_foreign_owner_is_refused(factory: ProfilingStoreFactory) -> None:
    other = factory.model_copy(update={"run_id": "f" * 32})
    repo = icechunk.Repository.create(factory._storage())
    repo.create_tag(other.ownership_tag, repo.lookup_branch("main"))
    with pytest.raises(AssertionError, match="not owned"):
        factory.primary_store()
    with pytest.raises(AssertionError, match="Unexpected owner"):
        factory.bootstrap(1, timeout_seconds=5)


def test_nonzero_worker_does_not_create(factory: ProfilingStoreFactory) -> None:
    with pytest.raises(TimeoutError):
        factory.bootstrap(1, timeout_seconds=0)
    assert not icechunk.Repository.exists(factory._storage())


def test_nonzero_worker_waits_half_an_hour_by_default() -> None:
    defaults = ProfilingStoreFactory.bootstrap.__defaults__
    assert defaults is not None
    assert defaults[-1] == 1800


@pytest.mark.parametrize(
    "mutation",
    [
        {"template_config_version": "0.1.0"},
        {"dataset_id": "ecmwf-ifs-ens-forecast-15-day-0-25-degree"},
        {"run_id": "../../production"},
        {
            "primary_storage_config": ProfilingStorage(
                base_path="s3://dynamical-ecmwf-ifs-ens"
            )
        },
        {
            "primary_storage_config": ProfilingStorage(
                base_path=EXPECTED_BASE,
                k8s_secret_name="other",  # noqa: S106
            )
        },
        {"replica_storage_configs": [ProfilingStorage(base_path=EXPECTED_BASE)]},
    ],
)
def test_bad_destination_fails_before_storage(
    monkeypatch: pytest.MonkeyPatch, mutation: dict[str, Any]
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Opened storage before validating the destination")

    monkeypatch.setattr(storage, "_get_icechunk_storage", forbidden)
    monkeypatch.setattr(storage.StorageConfig, "load_storage_options", forbidden)
    factory = dataset(RUN).store_factory.model_copy(update=mutation)
    for open_storage in (
        lambda: factory.primary_store(writable=True),
        lambda: factory.icechunk_repos(sort="primary-first"),
        lambda: factory.bootstrap(0),
        factory._coordination_fs,
    ):
        with pytest.raises(AssertionError):
            open_storage()


@pytest.mark.parametrize(
    "options",
    [
        {"endpoint_url": "https://example.invalid"},
        {"bucket": "other"},
        {"prefix": "other"},
        {"region": "us-east-1"},
    ],
)
def test_storage_options_refuse_other_destinations(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, str]
) -> None:
    monkeypatch.setattr(
        storage.StorageConfig, "load_storage_options", lambda _self: options
    )
    with pytest.raises(AssertionError):
        ProfilingStorage(base_path=EXPECTED_BASE).load_storage_options()


def test_no_operational_resources() -> None:
    assert dataset(RUN).operational_kubernetes_resources("image") == ()


def test_backfill_resource_is_the_ens_update_descriptor() -> None:
    profiling = dataset(RUN)
    resource = profiling._backfill_resource()
    production = item_reformat_cron_job(
        EcmwfIfsEnsForecast15Day025DegreeDataset(
            primary_storage_config=ProfilingStorage(base_path=EXPECTED_BASE)
        )
    )
    assert resource.name == f"{DATASET_ID}-update"
    assert resource.dataset_id == DATASET_ID
    assert resource.secret_names == ["aws-open-data-icechunk-storage-options-key"]
    assert resource.model_dump(
        exclude={"name", "dataset_id", "image"}
    ) == production.model_dump(exclude={"name", "dataset_id", "image"})


def item_reformat_cron_job(
    ens: EcmwfIfsEnsForecast15Day025DegreeDataset,
) -> ReformatCronJob:
    (resource,) = ens.operational_kubernetes_resources("placeholder")
    assert isinstance(resource, ReformatCronJob)
    return resource


def test_backfill_job_is_a_reported_profiling_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    profiling = dataset(RUN)
    resource = profiling._backfill_resource()
    command = [
        "backfill",
        END.tz_localize(None).isoformat(),
        f"--filter-start={START.tz_localize(None).isoformat()}",
        f"--filter-end={END.tz_localize(None).isoformat()}",
    ]
    image = "example.invalid/reformatters:" + "a" * 40
    job = profiling._backfill_job(command, image, 38, 2, resource)
    expected = DynamicalDataset._backfill_job(
        profiling, command, image, 38, 2, resource
    )
    assert isinstance(job, ProfilingJob)
    assert job.run_id == RUN
    assert job.model_dump(exclude={"run_id"}) == expected.model_dump()
    assert (job.dataset_id, job.workers_total, job.parallelism) == (DATASET_ID, 38, 2)
    assert RUN in summary.read_text()
    assert dataset(RUN).store_factory.primary_url() in summary.read_text()


def test_template_preserves_all_encodings() -> None:
    profiling = ProfilingTemplateConfig()
    production = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()
    assert profiling.data_vars == production.data_vars
    assert [
        (coord.name, coord.encoding.model_dump_json()) for coord in profiling.coords
    ] == [(coord.name, coord.encoding.model_dump_json()) for coord in production.coords]
    assert profiling.append_dim_coordinate_chunk_size() == 5840
    assert profiling.template_path() != production.template_path()
    assert profiling.append_dim_start == START.tz_localize(None)
    assert profiling.dataset_id == DATASET_ID
    assert profiling.version == VERSION
    assert profiling.get_template(END.tz_localize(None)).sizes["init_time"] == 2


def test_two_inits_nineteen_variables_thirty_eight_jobs() -> None:
    profiling = dataset(RUN)
    jobs = profiling.region_job_class.get_jobs(
        tmp_store=Path("unused"),
        template_ds=profiling.template_config.get_template(END.tz_localize(None)),
        append_dim=profiling.template_config.append_dim,
        all_data_vars=profiling.template_config.data_vars,
        reformat_job_name="profiling-test",
    )
    assert len(profiling.template_config.data_vars) == 19
    assert len(jobs) == 38
    assert {job.region.start for job in jobs} == {0, 1}
    assert (
        len({(job.region.start, var.name) for job in jobs for var in job.data_vars})
        == 38
    )


def test_profiling_dataset_is_unregistered() -> None:
    assert DATASET_ID not in {d.dataset_id for d in IMPLEMENTED_DATASETS}


def test_backfill_never_uses_production_factory_openers(
    factory: ProfilingStoreFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Profiling used a production store opener")

    profiling = dataset(RUN)
    monkeypatch.setattr(storage.StoreFactory, "primary_store", forbidden)
    monkeypatch.setattr(storage.StoreFactory, "icechunk_repos", forbidden)
    monkeypatch.setattr(icechunk.Repository, "open_or_create", forbidden)
    monkeypatch.setattr(
        profiling.region_job_class, "process_worker_jobs", lambda *_args, **_kwargs: {}
    )
    factory.bootstrap(0)
    profiling.backfill(
        END.tz_localize(None).to_pydatetime(),
        "local-profiling-factory-test",
        worker_index=0,
        workers_total=1,
    )
    repo = factory.open_owned_repository()
    assert len(list(repo.ancestry(branch="main"))) > 1


@pytest.fixture
def template_dataset() -> ProfilingDataset:
    return ProfilingDataset(
        run_id=RUN, primary_storage_config=ProfilingStorage(base_path=EXPECTED_BASE)
    )


def test_checked_in_template_is_current_and_round_trips(
    template_dataset: ProfilingDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_config = template_dataset.template_config
    checked_in = template_config.template_path()
    common_template_config_subclasses_test.test_template_config_structure_is_valid(
        template_dataset
    )

    updated = tmp_path / "latest.zarr"
    monkeypatch.setattr(type(template_config), "template_path", lambda _self: updated)
    template_config.update_template()
    end = template_config.dimension_coordinates()[template_config.append_dim][-1]
    roundtrip = tmp_path / "roundtrip.zarr"
    template_utils.write_metadata(
        template_config.get_template(end + pd.Timedelta(milliseconds=1)), roundtrip
    )
    expected = json.loads((checked_in / "zarr.json").read_text())
    assert json.loads((updated / "zarr.json").read_text()) == expected
    assert json.loads((roundtrip / "zarr.json").read_text()) == expected


def test_checked_in_template_serialization(
    template_dataset: ProfilingDataset,
) -> None:
    template_serialization_test.test_template_fill_values_are_correct(template_dataset)
    with xr.open_zarr(
        template_dataset.template_config.template_path(), decode_timedelta=True
    ) as stored:
        for check in (
            template_serialization_test.test_coordinates_load_without_error,
            template_serialization_test.test_timedelta_coordinates_stored_as_float,
            template_serialization_test.test_coordinates_have_single_chunk,
            template_serialization_test.test_coordinates_not_sharded,
        ):
            check(template_dataset, stored)


def test_checked_in_template_cf_compliance(
    template_dataset: ProfilingDataset,
) -> None:
    def open_template(path: Path) -> xr.Dataset:
        return xr.open_zarr(path)

    for check in (
        datasets_cf_compliance_test.test_cf_latitude_longitude_recognized,
        datasets_cf_compliance_test.test_cf_time_coordinates_recognized,
        datasets_cf_compliance_test.test_cf_ensemble_member_recognized,
        datasets_cf_compliance_test.test_cf_coordinates_have_long_name,
        datasets_cf_compliance_test.test_cf_data_variables_have_long_name,
    ):
        check(template_dataset, open_template)
    datasets_cf_compliance_test.test_cf_dimension_coordinate_axis(template_dataset)
