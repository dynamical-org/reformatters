from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from reformatters.common.pydantic import replace
from reformatters.nasa.imerg.analysis_late_v7.region_job import (
    NasaImergAnalysisLateV7RegionJob,
    NasaImergAnalysisLateV7SourceFileCoord,
)
from reformatters.nasa.imerg.analysis_late_v7.template_config import (
    NasaImergAnalysisLateV7TemplateConfig,
)
from tests.integration import require_secret


def test_get_url_late() -> None:
    coord = NasaImergAnalysisLateV7SourceFileCoord(
        time=pd.Timestamp("2024-01-15T00:00")
    )
    assert coord.get_url("V07B") == (
        "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHHL.07/"
        "2024/015/3B-HHR-L.MS.MRG.3IMERG.20240115-S000000-E002959.0000.V07B.HDF5"
    )


def test_region_job_uses_late_coord(tmp_path: Path) -> None:
    mock_ds = Mock()
    mock_ds.attrs = {"dataset_id": "nasa-imerg-analysis-late-v7"}
    region_job = NasaImergAnalysisLateV7RegionJob.model_construct(
        tmp_store=tmp_path,
        template_ds=mock_ds,
        data_vars=NasaImergAnalysisLateV7TemplateConfig().data_vars,
        append_dim="time",
        region=slice(0, 1),
        reformat_job_name="test",
    )
    assert region_job.source_file_coord_class is NasaImergAnalysisLateV7SourceFileCoord

    times = pd.date_range("2024-01-15T00:00", periods=2, freq="30min")
    coords = region_job.generate_source_file_coords(
        xr.Dataset(coords={"time": times}), region_job.data_vars
    )
    assert all(isinstance(c, NasaImergAnalysisLateV7SourceFileCoord) for c in coords)
    assert "3B-HHR-L" in coords[0].get_url("V07B")


@pytest.mark.slow
def test_download_and_read_late(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Download a real IMERG Late granule from GES DISC and read precipitation.

    Requires NASA Earthdata credentials (the nasa-earthdata secret); skipped otherwise.
    """
    require_secret(monkeypatch, "nasa-earthdata")

    config = NasaImergAnalysisLateV7TemplateConfig()
    mock_ds = Mock()
    mock_ds.attrs = {"dataset_id": config.dataset_attributes.dataset_id}
    region_job = NasaImergAnalysisLateV7RegionJob.model_construct(
        tmp_store=tmp_path,
        template_ds=mock_ds,
        data_vars=config.data_vars,
        append_dim="time",
        region=slice(0, 1),
        reformat_job_name="test",
    )
    coord = NasaImergAnalysisLateV7SourceFileCoord(
        time=pd.Timestamp("2024-01-15T00:00")
    )
    coord = replace(coord, downloaded_path=region_job.download_file(coord))

    precip = next(v for v in config.data_vars if v.name == "precipitation_surface")
    data = region_job.read_data(coord, precip)
    assert data.shape == (1800, 3600)
    valid = data[~np.isnan(data)]
    assert valid.size > 0
    assert valid.min() >= 0.0
