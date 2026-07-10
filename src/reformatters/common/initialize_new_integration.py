import re
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer


class DatasetKind(StrEnum):
    """Which teaching template to scaffold from.

    materialized: download, rechunk, and rewrite the bytes into a new zarr/icechunk
        store.
    virtual: an icechunk store of chunk references that point at source files in
        place, decoded at read time. See docs/virtual_datasets.md.
    """

    materialized = "materialized"
    virtual = "virtual"


_EXAMPLE_DIRS: dict[DatasetKind, str] = {
    DatasetKind.materialized: "example_materialized",
    DatasetKind.virtual: "example_virtual",
}

# A dataset's class names encode how it is used, not how it is built: materialized
# datasets are time-optimized (Temporal), virtual datasets are spatial-optimized
# (Spatial). The teaching templates are named to match.
_KIND_INFIX: dict[DatasetKind, str] = {
    DatasetKind.materialized: "Temporal",
    DatasetKind.virtual: "Spatial",
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

    # Create directories, with __init__.py at each new level (provider/, provider/model/)
    # so they're importable packages, not just the leaf variant/ directory.
    for base, path in (
        (Path("src/reformatters"), src_path),
        (Path("tests"), test_path),
    ):
        current = base
        for part in path.relative_to(base).parts:
            current = current / part
            current.mkdir(exist_ok=True)
            (current / "__init__.py").touch(exist_ok=True)

    # Copy from the chosen example template
    example_dirname = _EXAMPLE_DIRS[kind]
    example_src = Path("src/reformatters") / example_dirname
    example_test = Path("tests") / example_dirname

    for file in example_src.glob("*"):
        if file.is_file():
            shutil.copy(file, src_path / file.name)

    for file in example_test.glob("*"):
        if file.is_file():
            shutil.copy(file, test_path / file.name)

    # Perform renames in copied files. The dataset class names carry the kind's infix
    # (Temporal / Spatial); DataVar and InternalAttrs are model-level config models
    # shared across a model's datasets, so they stay {provider}{model}-scoped. The
    # import-prefix key uses the chosen example package so virtual copies aren't
    # half-renamed by a "reformatters.example" prefix match.
    infix = _KIND_INFIX[kind]
    dataset_class_name = (
        f"{provider_pascal}{model_pascal}{variant_pascal}{infix}DynamicalDataset"
    )
    example_to_actual_mappings = {
        f"Example{infix}DynamicalDataset": dataset_class_name,
        f"Example{infix}TemplateConfig": f"{provider_pascal}{model_pascal}{variant_pascal}{infix}TemplateConfig",
        f"Example{infix}RegionJob": f"{provider_pascal}{model_pascal}{variant_pascal}{infix}RegionJob",
        f"Example{infix}SourceFileCoord": f"{provider_pascal}{model_pascal}{variant_pascal}{infix}SourceFileCoord",
        "ExampleDataVar": f"{provider_pascal}{model_pascal}DataVar",
        "ExampleInternalAttrs": f"{provider_pascal}{model_pascal}InternalAttrs",
        f"reformatters.{example_dirname}": f"reformatters.{provider}.{model}.{variant}",
    }

    # Process all Python files in both src and test directories
    for path in [src_path, test_path]:
        for file in path.glob("*.py"):
            content = file.read_text()
            for old, new in example_to_actual_mappings.items():
                content = content.replace(old, new)
            file.write_text(content)
    (src_path / "__init__.py").write_text(
        f"from .dynamical_dataset import {dataset_class_name} as {dataset_class_name}\n"
    )

    print(  # noqa: T201
        f"Created new {kind.value} dataset integration at {src_path} and {test_path}"
    )


def _sanitize_identifier(s: str) -> str:
    """Convert string to valid Python identifier by replacing invalid chars with underscore."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", s.lower())


def _pascal_case(s: str) -> str:
    """Convert string to PascalCase."""
    return s.replace("_", " ").title().replace(" ", "")
