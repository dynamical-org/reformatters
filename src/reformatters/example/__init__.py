import re
import shutil
from pathlib import Path


def _sanitize_identifier(s: str) -> str:
    """Convert string to valid Python identifier by replacing invalid chars with underscore."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", s)


def new(provider: str, model: str, variant: str) -> None:
    """Create a new dataset integration from the example template.

    Args:
        provider: The provider name in lowercase (e.g. 'noaa')
        model: The model name in lowercase (e.g. 'gfs')
        variant: The variant name in lowercase (e.g. 'forecast')
    """
    # Sanitize inputs
    provider = _sanitize_identifier(provider)
    model = _sanitize_identifier(model)
    variant = _sanitize_identifier(variant)

    # Convert to PascalCase for class names
    provider_pascal = provider.capitalize()
    model_pascal = model.capitalize()
    variant_pascal = variant.capitalize()

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
        if file.is_file() and file.name != "__init__.py":  # Skip __init__.py
            shutil.copy(file, src_path / file.name)

    for file in example_test.glob("*"):
        if file.is_file():
            shutil.copy(file, test_path / file.name)

    # Create empty __init__.py
    (src_path / "__init__.py").touch()
    (test_path / "__init__.py").touch()

    # Perform renames in copied files
    replacements = [
        ("ExampleDataset", f"{provider_pascal}{model_pascal}{variant_pascal}Dataset"),
        (
            "ExampleTemplateConfig",
            f"{provider_pascal}{model_pascal}{variant_pascal}TemplateConfig",
        ),
        (
            "ExampleRegionJob",
            f"{provider_pascal}{model_pascal}{variant_pascal}RegionJob",
        ),
        ("ExampleDataVar", f"{provider_pascal}{model_pascal}DataVar"),
        ("ExampleInternalAttrs", f"{provider_pascal}{model_pascal}InternalAttrs"),
        ("ExampleSourceFileCoord", f"{provider_pascal}{model_pascal}SourceFileCoord"),
        ("reformatters.example", f"reformatters.{provider}.{model}.{variant}"),
    ]

    # Process all Python files in both src and test directories
    for path in [src_path, test_path]:
        for file in path.glob("*.py"):
            content = file.read_text()
            for old, new in replacements:
                content = content.replace(old, new)
            file.write_text(content)

    print(f"Created new dataset integration at {dataset_path}")
    print("\nNext steps:")
    print("1. Implement your TemplateConfig subclass")
    print("2. Implement your RegionJob subclass")
    print("3. Implement your DynamicalDataset subclass")
    print("4. Register your dataset in src/reformatters/__main__.py")
