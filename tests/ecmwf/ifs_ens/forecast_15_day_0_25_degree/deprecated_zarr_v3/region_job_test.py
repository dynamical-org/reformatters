from unittest.mock import Mock

import pandas as pd
import pytest

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
