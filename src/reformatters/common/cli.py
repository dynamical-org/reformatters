import typer

from reformatters.common.dynamical_dataset import (
    DATA_VAR,
    SOURCE_FILE_COORD,
    DynamicalDataset,
)


def create_cli(
    dataset: DynamicalDataset[DATA_VAR, SOURCE_FILE_COORD],
) -> tuple[str, typer.Typer]:
    app = typer.Typer()
    app.command(name="update-template")(dataset.update_template)
    app.command(name="reformat-local")(dataset.reformat_local)
    app.command(name="reformat-kubernetes")(dataset.reformat_kubernetes)
    return dataset.template_config.dataset_id, app
