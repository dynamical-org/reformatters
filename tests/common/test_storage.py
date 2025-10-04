import icechunk
import pytest
import zarr

from reformatters.common.config import Config, Env
from reformatters.common.storage import (
    DatasetFormat,
    StorageConfig,
    StoreFactory,
    _get_store_path,
)


@pytest.mark.parametrize(
    "env,dataset_format,expected_base,expected_extension",
    [
        (Env.prod, DatasetFormat.ZARR3, "s3://prod-bucket/data", ".zarr"),
        (Env.prod, DatasetFormat.ICECHUNK, "s3://prod-bucket/data", ".icechunk"),
        (Env.dev, DatasetFormat.ZARR3, "local/output", ".zarr"),
        (Env.dev, DatasetFormat.ICECHUNK, "local/output", ".icechunk"),
    ],
)
def test_get_store_path(
    monkeypatch: pytest.MonkeyPatch,
    env: Env,
    dataset_format: DatasetFormat,
    expected_base: str,
    expected_extension: str,
) -> None:
    """Test _get_store_path uses correct base path and extension based on environment and format."""
    monkeypatch.setattr(
        "reformatters.common.storage._LOCAL_ZARR_STORE_BASE_PATH", "local/output"
    )
    monkeypatch.setattr(Config, "env", env)

    config = StorageConfig(
        base_path="s3://prod-bucket/data",
        format=dataset_format,
    )

    result = _get_store_path("dataset", "1.0", config)
    assert result == f"{expected_base}/dataset/v1.0{expected_extension}"


@pytest.mark.parametrize(
    "env,expected_version",
    [
        (Env.dev, "dev"),
        (Env.prod, "v1.5"),
        (Env.test, "v1.5"),
    ],
)
def test_store_factory_version(
    monkeypatch: pytest.MonkeyPatch, env: Env, expected_version: str
) -> None:
    """Test StoreFactory.version returns correct value based on environment."""
    monkeypatch.setattr(Config, "env", env)

    primary_config = StorageConfig(
        base_path="s3://bucket/data",
        format=DatasetFormat.ZARR3,
    )

    factory = StoreFactory(
        primary_storage_config=primary_config,
        dataset_id="test-dataset",
        template_config_version="v1.5",
    )

    assert factory.version == expected_version


def test_store_factory_k8s_secret_names() -> None:
    """Test StoreFactory.k8s_secret_names returns all secret names."""
    primary_config = StorageConfig(
        base_path="s3://bucket/primary",
        k8s_secret_name="primary-secret",  # noqa: S106
        format=DatasetFormat.ZARR3,
    )
    replica_config1 = StorageConfig(
        base_path="s3://bucket/replica1",
        k8s_secret_name="replica1-secret",  # noqa: S106
        format=DatasetFormat.ICECHUNK,
    )
    replica_config2 = StorageConfig(
        base_path="s3://bucket/replica2",
        k8s_secret_name="replica2-secret",  # noqa: S106
        format=DatasetFormat.ICECHUNK,
    )

    factory = StoreFactory(
        primary_storage_config=primary_config,
        replica_storage_configs=[replica_config1, replica_config2],
        dataset_id="test-dataset",
        template_config_version="v1.0",
    )

    secret_names = factory.k8s_secret_names()
    assert secret_names == ["primary-secret", "replica1-secret", "replica2-secret"]


@pytest.mark.parametrize(
    "env,expected_mode",
    [
        (Env.dev, "w"),
        (Env.prod, "w-"),
        (Env.test, "w-"),
    ],
)
def test_store_factory_mode(
    monkeypatch: pytest.MonkeyPatch, env: Env, expected_mode: str
) -> None:
    """Test StoreFactory.mode returns correct value based on environment."""
    monkeypatch.setattr(Config, "env", env)

    primary_config = StorageConfig(
        base_path="s3://bucket/data",
        format=DatasetFormat.ZARR3,
    )

    factory = StoreFactory(
        primary_storage_config=primary_config,
        dataset_id="test-dataset",
        template_config_version="v1.0",
    )

    assert factory.mode() == expected_mode


def test_store_factory_returns_correct_store_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test StoreFactory.replica_stores returns correct store types."""
    primary_config = StorageConfig(
        base_path="s3://bucket/primary",
        format=DatasetFormat.ZARR3,
    )
    factory = StoreFactory(
        primary_storage_config=primary_config,
        replica_storage_configs=[
            StorageConfig(
                base_path="s3://bucket/replica",
                format=DatasetFormat.ICECHUNK,
            ),
            StorageConfig(
                base_path="s3://bucket/replica",
                format=DatasetFormat.ZARR3,
            ),
        ],
        dataset_id="test-dataset",
        template_config_version="v1.0",
        stores_are_writable=True,
    )

    assert isinstance(factory.primary_store(), zarr.storage.LocalStore)

    replicas = factory.replica_stores()
    assert len(replicas) == 2
    assert isinstance(replicas[0], icechunk.store.IcechunkStore)
    assert isinstance(replicas[1], zarr.storage.LocalStore)
