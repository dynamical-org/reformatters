import os

import icechunk
import pytest
import s3fs

from reformatters.common import kubernetes, storage


@pytest.fixture(autouse=True)
def isolated_from_cluster_and_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Unexpected credential, cluster, or remote storage access")

    monkeypatch.setattr(kubernetes, "load_secret", forbidden)
    monkeypatch.setattr(kubernetes.config, "load_kube_config", forbidden)
    monkeypatch.setattr(kubernetes.config, "load_incluster_config", forbidden)
    monkeypatch.setattr(s3fs.S3FileSystem, "__init__", forbidden)
    monkeypatch.setattr(s3fs.S3FileSystem, "_cache", {})
    monkeypatch.setattr(storage, "_get_icechunk_storage", forbidden)
    monkeypatch.setattr(icechunk, "s3_storage", forbidden)
    monkeypatch.setattr(icechunk, "http_storage", forbidden)
    for name in [name for name in os.environ if name.startswith("AWS_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
