import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from reformatters.common import template_utils
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.virtual_region_job import VirtualRegionJob
from tests.dataset_helpers import IMPLEMENTED_DATASETS

_VIRTUAL_DATASETS = [
    d for d in IMPLEMENTED_DATASETS if issubclass(d.region_job_class, VirtualRegionJob)
]
_FAST_ROUNDTRIP_IDS = {
    "noaa-mrms-conus-analysis-hourly",
    "ecmwf-aifs-single-forecast-virtual",
}
assert _FAST_ROUNDTRIP_IDS <= {d.dataset_id for d in IMPLEMENTED_DATASETS}


@pytest.fixture(scope="module")
def updated_template_path(
    dataset: DynamicalDataset[Any, Any], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    template_config = dataset.template_config
    path = tmp_path_factory.mktemp(dataset.dataset_id) / "latest.zarr"
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(type(template_config), "template_path", lambda _self: path)
        template_config.update_template()
    return path


@pytest.fixture(scope="module")
def roundtrip_template_path(
    dataset: DynamicalDataset[Any, Any], updated_template_path: Path
) -> Path:
    template_config = dataset.template_config
    dim_coords = template_config.dimension_coordinates()
    append_dim_coords = dim_coords[template_config.append_dim]
    end_time = append_dim_coords[-1] + pd.Timedelta(milliseconds=1)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            type(template_config), "template_path", lambda _self: updated_template_path
        )
        template = template_config.get_template(end_time)
    path = updated_template_path.parent / "roundtrip.zarr"
    template_utils.write_metadata(template, path)
    return path


def test_template_config_structure_is_valid(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    """Every config passes the structure validators: group dims, (group, name)
    uniqueness, no vertical dim unused by any var, uniform append-dim chunking, etc."""
    dataset.template_config._assert_valid_structure()


def test_update_template_matches_existing_template(
    dataset: DynamicalDataset[Any, Any],
    updated_template_path: Path,
) -> None:
    """
    Ensure that `uv run main <dataset-id> update-template` has been run and
    all changes to the dataset's TemplateConfig are reflected in the on-disk Zarr template.
    """
    with open(updated_template_path / "zarr.json") as f:
        updated_template = json.load(f)

    with open(dataset.template_config.template_path() / "zarr.json") as f:
        assert json.load(f) == updated_template


@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param(
            dataset,
            id=dataset.dataset_id,
            marks=(
                [] if dataset.dataset_id in _FAST_ROUNDTRIP_IDS else [pytest.mark.slow]
            ),
        )
        for dataset in IMPLEMENTED_DATASETS
    ],
    indirect=True,
    scope="module",
)
def test_update_template_round_trips_correctly(
    dataset: DynamicalDataset[Any, Any],
    roundtrip_template_path: Path,
) -> None:
    """
    Ensure that the get_template() -> write_metadata() round trip produces exactly
    the same zarr.json as already exists on disk.
    """
    with open(roundtrip_template_path / "zarr.json") as f:
        roundtrip_template = json.load(f)

    with open(dataset.template_config.template_path() / "zarr.json") as f:
        assert json.load(f) == roundtrip_template


@pytest.mark.parametrize(
    "dataset", _VIRTUAL_DATASETS, ids=[d.dataset_id for d in _VIRTUAL_DATASETS]
)
def test_virtual_serializers_flip_to_shared_orientation(
    dataset: DynamicalDataset[Any, Any],
) -> None:
    """Every GRIB-decoding virtual data var's GribberishCodec flips decoded messages into
    our shared orientation: north_up (first row = largest latitude, matching materialized
    datasets' GDAL flip) and adjust_longitude_range (monotonic -180..180 longitude)."""
    for var in dataset.template_config.data_vars:
        serializer = var.encoding.serializer
        assert serializer is not None, var.name
        if serializer["name"] != "gribberish":
            continue
        assert serializer["configuration"]["north_up"] is True, var.name
        assert serializer["configuration"]["adjust_longitude_range"] is True, var.name
