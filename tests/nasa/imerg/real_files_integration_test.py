"""Real-file integration tests for NASA IMERG.

These download real granules and require `DYNAMICAL_ENV=prod` plus the
`nasa-earthdata` (GES DISC) and `nasa-pps` (jsimpson) cluster secrets, e.g.:

    DYNAMICAL_ENV=prod uv run pytest tests/nasa/imerg/real_files_integration_test.py

They validate the two things the mocked pipeline test cannot: that a real
granule downloads and that GDAL's HDF5 read produces a correctly oriented,
sanely-valued (lat, lon) grid — for both the deep archive and recent granules,
and that the operational jsimpson path returns the same values as GES DISC.
"""

import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest

from reformatters.common.config import Config, Env
from reformatters.common.download import get_local_path
from reformatters.common.pydantic import replace
from reformatters.nasa.imerg.analysis_early import NasaImergAnalysisEarlyDataset
from reformatters.nasa.imerg.analysis_late import NasaImergAnalysisLateDataset
from reformatters.nasa.imerg.dynamical_dataset import (
    NasaImergAnalysisMaterializedDataset,
)
from reformatters.nasa.imerg.imerg_config_models import ImergRun
from reformatters.nasa.imerg.region_job import (
    DownloadSource,
    NasaImergAnalysisSourceFileCoord,
)
from reformatters.nasa.imerg.template_config import GRID_LAT_SIZE, GRID_LON_SIZE
from reformatters.nasa.nasa_auth import get_earthdata_session, get_pps_session
from tests.common.dynamical_dataset_test import NOOP_STORAGE_CONFIG

# Needs cluster credentials (kubeconfig/op), which CI does not have — CI runs the full
# suite, so gate on an explicit opt-in rather than the `slow` marker alone.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("NASA_CREDENTIALED_TESTS") != "1",
        reason="requires cluster credentials; set NASA_CREDENTIALED_TESTS=1 to run",
    ),
]


@pytest.fixture(autouse=True)
def _prod_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # conftest forces DYNAMICAL_ENV=test; these tests need load_secret to hit the
    # cluster, so opt into prod at runtime.
    monkeypatch.setattr(Config, "env", Env.prod)


def _make_dataset(run: ImergRun) -> NasaImergAnalysisMaterializedDataset:
    if run == "early":
        return NasaImergAnalysisEarlyDataset(primary_storage_config=NOOP_STORAGE_CONFIG)
    return NasaImergAnalysisLateDataset(primary_storage_config=NOOP_STORAGE_CONFIG)


def _recent_granule_time(days_ago: int) -> pd.Timestamp:
    return (
        pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=days_ago)
    ).normalize()


def _download_from_source(url: str, source: DownloadSource, dataset_id: str) -> Path:
    session = get_pps_session() if source == "jsimpson" else get_earthdata_session()
    response = session.get(url, timeout=30, stream=True, allow_redirects=True)
    response.raise_for_status()
    local_path = get_local_path(dataset_id, path=urlparse(url).path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        f.writelines(response.iter_content(chunk_size=8192))
    return local_path


def _read_granule(
    run: ImergRun, time: pd.Timestamp, download_source: DownloadSource
) -> dict[str, np.ndarray]:
    dataset = _make_dataset(run)
    template = dataset._get_template(time + pd.Timedelta("30min"))
    n = template.to_dataset().sizes["time"]
    with tempfile.TemporaryDirectory() as tmp:
        job = dataset.region_job_class(
            tmp_store=Path(tmp) / "tmp.zarr",
            template_ds=template,
            data_vars=list(dataset.template_config.data_vars),
            append_dim="time",
            region=slice(n - 1, n),
            reformat_job_name="itest",
            run=run,
        )
        coord = NasaImergAnalysisSourceFileCoord(run=run, time=time)
        path = _download_from_source(
            coord.get_url(download_source), download_source, dataset.dataset_id
        )
        coord = replace(coord, downloaded_path=path)
        return {v.name: job.read_data(coord, v) for v in job.data_vars}


def _assert_sane(arrays: dict[str, np.ndarray]) -> None:
    precip = arrays["precipitation_surface"]
    qi = arrays["precipitation_quality_index_surface"]
    assert precip.shape == (GRID_LAT_SIZE, GRID_LON_SIZE)
    assert qi.shape == (GRID_LAT_SIZE, GRID_LON_SIZE)

    # A real global field has substantial valid coverage and non-negative rates.
    finite = np.isfinite(precip)
    assert finite.mean() > 0.5
    assert np.nanmin(precip) >= 0.0
    # kg m-2 s-1; 200 mm/hr (IMERG max) is ~0.056, so anything above ~0.1 is wrong.
    assert np.nanmax(precip) < 0.1
    # Quality index is a 0..1 fraction.
    assert np.nanmin(qi) >= 0.0
    assert np.nanmax(qi) <= 1.0


@pytest.mark.parametrize("run", ["early", "late"])
def test_deep_archive_granule(run: ImergRun) -> None:
    _assert_sane(_read_granule(run, pd.Timestamp("2001-06-15T00:00"), "gesdisc"))


@pytest.mark.parametrize("run", ["early", "late"])
def test_recent_granule(run: ImergRun) -> None:
    days_ago = 2 if run == "early" else 3
    _assert_sane(_read_granule(run, _recent_granule_time(days_ago), "gesdisc"))


def test_jsimpson_matches_gesdisc() -> None:
    time = _recent_granule_time(2)
    via_gesdisc = _read_granule("early", time, "gesdisc")
    via_jsimpson = _read_granule("early", time, "jsimpson")
    for name in via_gesdisc:
        np.testing.assert_allclose(
            via_jsimpson[name], via_gesdisc[name], rtol=1e-5, equal_nan=True
        )
