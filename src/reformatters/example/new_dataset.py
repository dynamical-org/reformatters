import re
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer


class DatasetKind(StrEnum):
    """Which teaching template to scaffold from.

    materialized: download, rechunk, and rewrite the bytes into a new zarr/icechunk
        store (the common case).
    virtual: an icechunk store of chunk references that point at source files in
        place, decoded at read time. See docs/virtual_datasets.md.
    """

    materialized = "materialized"
    virtual = "virtual"


_EXAMPLE_DIRS: dict[DatasetKind, str] = {
    DatasetKind.materialized: "example",
    DatasetKind.virtual: "example_virtual",
}


def initialize_new_integration(
    provider: Annotated[
        str, typer.Argument(help="The provider name in lowercase (e.g. 'noaa')")
    ],
    model: Annotated[
        str, typer.Argument(help="The model name in lowercase (e.g. 'gfs')")
    ],
    variant: Annotated[
        str, typer.Argument(help="The variant name in lowercase (e.g. 'forecast')")
    ],
    kind: Annotated[
        DatasetKind,
        typer.Option(
            help="Dataset kind to scaffold: 'materialized' (rechunk and rewrite the "
            "bytes) or 'virtual' (reference source files in place, decode at read time)."
        ),
    ],
) -> None:
    """Create a new dataset integration from a teaching template (materialized or virtual)."""
    # Sanitize inputs
    provider = _sanitize_identifier(provider)
    model = _sanitize_identifier(model)
    variant = _sanitize_identifier(variant)

    # Convert to PascalCase for class names
    provider_pascal = _pascal_case(provider)
    model_pascal = _pascal_case(model)
    variant_pascal = _pascal_case(variant)

    # Set up paths
    dataset_path = f"{provider}/{model}/{variant}"
    src_path = Path("src/reformatters") / dataset_path
    test_path = Path("tests") / dataset_path

    # Create directories
    src_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    # Copy from the chosen example template
    example_dirname = _EXAMPLE_DIRS[kind]
    example_src = Path("src/reformatters") / example_dirname
    example_test = Path("tests") / example_dirname

    for file in example_src.glob("*"):
        if file.is_file() and file.name != "new_dataset.py":  # Skip this file
            shutil.copy(file, src_path / file.name)

    for file in example_test.glob("*"):
        if file.is_file():
            shutil.copy(file, test_path / file.name)

    # Create empty __init__.py
    (src_path / "__init__.py").touch()
    (test_path / "__init__.py").touch()

    # Perform renames in copied files. The import-prefix key is the chosen example
    # package (reformatters.example or reformatters.example_virtual) so virtual copies
    # aren't half-renamed by a "reformatters.example" prefix match.
    example_to_actual_mappings = {
        "ExampleDataset": f"{provider_pascal}{model_pascal}{variant_pascal}Dataset",
        "ExampleTemplateConfig": f"{provider_pascal}{model_pascal}{variant_pascal}TemplateConfig",
        "ExampleRegionJob": f"{provider_pascal}{model_pascal}{variant_pascal}RegionJob",
        "ExampleDataVar": f"{provider_pascal}{model_pascal}DataVar",
        "ExampleInternalAttrs": f"{provider_pascal}{model_pascal}InternalAttrs",
        "ExampleSourceFileCoord": f"{provider_pascal}{model_pascal}{variant_pascal}SourceFileCoord",
        f"reformatters.{example_dirname}": f"reformatters.{provider}.{model}.{variant}",
    }

    # Process all Python files in both src and test directories
    for path in [src_path, test_path]:
        for file in path.glob("*.py"):
            content = file.read_text()
            for old, new in example_to_actual_mappings.items():
                content = content.replace(old, new)
            file.write_text(content)

    dataset_class_name = example_to_actual_mappings["ExampleDataset"]
    (src_path / "__init__.py").write_text(
        f"from .dynamical_dataset import {dataset_class_name} as {dataset_class_name}\n"
    )

    kind_specific_step = (
        "5. Set `icechunk_virtual_config` and use an ICECHUNK primary store "
        "(see docs/virtual_datasets.md)"
        if kind == DatasetKind.virtual
        else "5. Run `uv run main <dataset-id> update-template` to generate "
        "templates/latest.zarr"
    )
    print(  # noqa: T201
        f"Created new {kind.value} dataset integration at {src_path} and {test_path}\n\n"
        "Next steps:\n"
        "1. Register your dataset in src/reformatters/__main__.py\n"
        f"2. Implement your {example_to_actual_mappings['ExampleTemplateConfig']} subclass\n"
        f"3. Implement your {example_to_actual_mappings['ExampleRegionJob']} subclass\n"
        f"4. Implement your {example_to_actual_mappings['ExampleDataset']} subclass\n"
        f"{kind_specific_step}\n\n"
        f"Note: if shared config models already exist for this provider/model (e.g. a "
        f"<provider>/<model>_config_models.py defining "
        f"{example_to_actual_mappings['ExampleDataVar']} / "
        f"{example_to_actual_mappings['ExampleInternalAttrs']}), delete the scaffolded "
        f"copies and import the shared types instead."
    )


def _sanitize_identifier(s: str) -> str:
    """Convert string to valid Python identifier by replacing invalid chars with underscore."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", s.lower())


def _pascal_case(s: str) -> str:
    """Convert string to PascalCase."""
    return s.replace("_", " ").title().replace(" ", "")
