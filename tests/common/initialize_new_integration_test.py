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
        "example_class_prefix",
        "dataset_id_example",
        "dataset_name_example",
    ),
    [
        (
            DatasetKind.materialized,
            "forecast",
            "NoaaGfsForecast",
            "ExampleTemporal",
            'dataset_id="producer-model-variant"',
            'name="Producer Model Variant"',
        ),
        (
            DatasetKind.virtual,
            "forecast_virtual",
            "NoaaGfsForecastVirtual",
            "ExampleSpatial",
            'dataset_id="producer-model-variant-virtual"',
            'name="Producer Model Variant, virtual"',
        ),
    ],
)
def test_initialize_new_integration_names_follow_dataset_id(
    scaffold_root: Path,
    kind: DatasetKind,
    module_variant: str,
    class_prefix: str,
    example_class_prefix: str,
    dataset_id_example: str,
    dataset_name_example: str,
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
    assert example_class_prefix not in generated_python
    assert f"reformatters.noaa.gfs.{module_variant}" in generated_python
    template_config = (source_path / "template_config.py").read_text()
    assert dataset_id_example in template_config
    assert dataset_name_example in template_config
    assert (source_path / "__init__.py").read_text() == (
        f"from .dynamical_dataset import {class_prefix}Dataset as {class_prefix}Dataset\n"
    )


def test_virtual_variant_rejects_virtual_suffix(scaffold_root: Path) -> None:
    with pytest.raises(
        typer.BadParameter,
        match="variant must omit the 'virtual' suffix when --kind virtual is used",
    ):
        initialize_new_integration(
            "noaa", "gfs", "forecast_virtual", DatasetKind.virtual
        )

    assert not (scaffold_root / "src/reformatters/noaa").exists()
    assert not (scaffold_root / "tests/noaa").exists()
