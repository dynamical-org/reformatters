from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from reformatters.common import template_utils
from reformatters.common.storage import DatasetFormat, StorageConfig, StoreFactory
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.deprecated_zarr_v3.region_job import (
    EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3RegionJob,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.deprecated_zarr_v3.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3TemplateConfig,
)
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.source_file_coord import (
    OpenDataSourceFileCoord,
)


def test_download_file_reads_gcs_only(monkeypatch: pytest.MonkeyPatch) -> None:
    template_config = EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3TemplateConfig()
    region_job = (
        EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3RegionJob.model_construct(
            tmp_store=Mock(),
            template_ds=template_config.get_template(pd.Timestamp("2024-04-02")),
            data_vars=template_config.data_vars,
            append_dim=template_config.append_dim,
            region=slice(0, 1),
            reformat_job_name="test",
        )
    )
    coord = OpenDataSourceFileCoord(
        init_time=pd.Timestamp("2024-04-01"),
        lead_time=pd.Timedelta("3h"),
        data_var_group=[
            var for var in template_config.data_vars if var.name == "temperature_2m"
        ],
        ensemble_member=0,
    )
    download_to_disk_mock = Mock(side_effect=FileNotFoundError("not on GCS yet"))
    monkeypatch.setattr(
        "reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.region_job.http_download_to_disk",
        download_to_disk_mock,
    )

    with pytest.raises(FileNotFoundError):
        region_job.download_file(coord)

    (url,) = [call[0][0] for call in download_to_disk_mock.call_args_list]
    assert url.startswith("https://storage.googleapis.com/ecmwf-open-data/")


def _operational_update_jobs(
    existing_start: pd.Timestamp,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    template_config = EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3TemplateConfig()
    store_factory = StoreFactory(
        primary_storage_config=StorageConfig(
            base_path=str(tmp_path / "store"), format=DatasetFormat.ZARR3
        ),
        dataset_id="test-dataset",
        template_config_version="test-version",
    )
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda *args, **kwargs: pd.Timestamp("2024-05-02T15:48")),
    )
    existing_ds = template_config.model_copy(
        update={"append_dim_start": existing_start}
    ).get_template(pd.Timestamp("2024-05-01T00:01"))
    template_utils.write_metadata(existing_ds, store_factory)

    EcmwfIfsEnsForecast15Day025DegreeDeprecatedZarrV3RegionJob.operational_update_jobs(
        primary_store=store_factory.primary_store(),
        tmp_store=tmp_path / "tmp_ds.zarr",
        get_template_fn=template_config.get_template,
        append_dim=template_config.append_dim,
        all_data_vars=template_config.data_vars,
        reformat_job_name="test_job",
    )


def test_operational_update_jobs_accepts_april_2024_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _operational_update_jobs(pd.Timestamp("2024-04-01"), monkeypatch, tmp_path)


def test_operational_update_jobs_rejects_shifted_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(AssertionError, match="do not match the template"):
        _operational_update_jobs(pd.Timestamp("2024-03-01"), monkeypatch, tmp_path)
