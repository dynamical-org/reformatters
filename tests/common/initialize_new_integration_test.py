import shutil
from pathlib import Path

import pytest
import typer

from reformatters.common.initialize_new_integration import (
    DatasetKind,
    initialize_new_integration,
)

PROJECT_ROOT = Path(__file__).parents[2]


@pytest.fixture
def scaffold_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for example_dirname in ("example_materialized", "example_virtual"):
        for package_root in ("src/reformatters", "tests"):
            source = PROJECT_ROOT / package_root / example_dirname
            destination = tmp_path / package_root / example_dirname
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    (
        "kind",
        "module_variant",
        "class_prefix",
        "example_dataset_class_name",
    ),
    [
        (
            DatasetKind.materialized,
            "forecast",
            "NoaaGfsForecast",
            "ExampleDataset",
        ),
        (
            DatasetKind.virtual,
            "forecast_virtual",
            "NoaaGfsForecastVirtual",
            "ExampleVirtualDataset",
        ),
    ],
)
def test_initialize_new_integration_names_follow_dataset_id(
    scaffold_root: Path,
    kind: DatasetKind,
    module_variant: str,
    class_prefix: str,
    example_dataset_class_name: str,
) -> None:
    initialize_new_integration("noaa", "gfs", "forecast", kind)

    source_path = scaffold_root / "src/reformatters/noaa/gfs" / module_variant
    test_path = scaffold_root / "tests/noaa/gfs" / module_variant
    assert source_path.is_dir()
    assert test_path.is_dir()

    generated_python = "\n".join(
        path.read_text()
        for generated_path in (source_path, test_path)
        for path in generated_path.glob("*.py")
    )
    assert f"class {class_prefix}Dataset(" in generated_python
    assert f"class {class_prefix}TemplateConfig(" in generated_python
    assert f"class {class_prefix}RegionJob(" in generated_python
    assert f"class {class_prefix}SourceFileCoord(" in generated_python
    assert f"class {example_dataset_class_name}(" not in generated_python
    assert f"reformatters.noaa.gfs.{module_variant}" in generated_python
    assert (source_path / "__init__.py").read_text() == (
        f"from .dynamical_dataset import {class_prefix}Dataset as {class_prefix}Dataset\n"
    )


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        (
            DatasetKind.materialized,
            "variant cannot use the 'virtual' suffix with --kind materialized",
        ),
        (
            DatasetKind.virtual,
            "variant must omit the 'virtual' suffix when --kind virtual is used",
        ),
    ],
)
def test_variant_rejects_virtual_suffix(
    scaffold_root: Path, kind: DatasetKind, message: str
) -> None:
    with pytest.raises(
        typer.BadParameter,
        match=message,
    ):
        initialize_new_integration("noaa", "gfs", "forecast-virtual", kind)

    assert not (scaffold_root / "src/reformatters/noaa").exists()
    assert not (scaffold_root / "tests/noaa").exists()
