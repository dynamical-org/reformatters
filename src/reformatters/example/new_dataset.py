import re
import shutil
from pathlib import Path
from typing import Annotated

import typer


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
) -> None:
    """Create a new dataset integration from the example template."""
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

    # Copy example files
    example_src = Path("src/reformatters/example")
    example_test = Path("tests/example")

    for file in example_src.glob("*"):
        if file.is_file() and file.name != "new_dataset.py":  # Skip this file
            shutil.copy(file, src_path / file.name)

    for file in example_test.glob("*"):
        if file.is_file():
            shutil.copy(file, test_path / file.name)

    # Create empty __init__.py
    (src_path / "__init__.py").touch()
    (test_path / "__init__.py").touch()

    # Perform renames in copied files
    example_to_actual_mappings = {
        "ExampleDataset": f"{provider_pascal}{model_pascal}{variant_pascal}Dataset",
        "ExampleTemplateConfig": f"{provider_pascal}{model_pascal}{variant_pascal}TemplateConfig",
        "ExampleRegionJob": f"{provider_pascal}{model_pascal}{variant_pascal}RegionJob",
        "ExampleDataVar": f"{provider_pascal}{model_pascal}DataVar",
        "ExampleInternalAttrs": f"{provider_pascal}{model_pascal}InternalAttrs",
        "ExampleSourceFileCoord": f"{provider_pascal}{model_pascal}{variant_pascal}SourceFileCoord",
        "reformatters.example": f"reformatters.{provider}.{model}.{variant}",
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

    # rewrite this into 1 print AI!
    print(f"Created new dataset integration at {src_path} and {test_path}")
    print("\nNext steps:")
    print("1. Register your dataset in src/reformatters/__main__.py")
    print(
        f"2. Implement your {example_to_actual_mappings['ExampleTemplateConfig']} subclass"
    )
    print(
        f"3. Implement your {example_to_actual_mappings['ExampleRegionJob']} subclass"
    )
    print(f"4. Implement your {example_to_actual_mappings['ExampleDataset']} subclass")


def _sanitize_identifier(s: str) -> str:
    """Convert string to valid Python identifier by replacing invalid chars with underscore."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", s.lower())


def _pascal_case(s: str) -> str:
    """Convert string to PascalCase."""
    return s.replace("_", " ").title().replace(" ", "")
